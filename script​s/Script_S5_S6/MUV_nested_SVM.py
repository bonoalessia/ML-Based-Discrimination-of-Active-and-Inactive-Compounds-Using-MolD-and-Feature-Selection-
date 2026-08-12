#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Leakage-controlled nested cross-validation for MUV molecular-descriptor data.

PURPOSE
-------
This script performs the leakage-controlled nested cross-validation Round I analysis for one MUV target.
It performs:

1. Outer 5-fold stratified cross-validation for unbiased performance estimation.
2. Inner 3-fold stratified cross-validation for SVM hyperparameter tuning.
3. Fold-specific variance filtering and standardization inside a scikit-learn Pipeline.
4. RBF-kernel SVM with class_weight="balanced".
5. Evaluation using ROC-AUC and EF1% from continuous model scores.
6. Decision-threshold optimization using inner-training data only.
7. Storage of fold-level metrics, selected hyperparameters, out-of-fold predictions,
   summary statistics, and the mean ROC curve.

IMPORTANT
---------
- The original active/inactive class distribution is preserved in every fold.
- No resampling or artificial class balancing is performed.
- The optimized decision threshold is NOT used for ROC-AUC or EF1%, because these
  metrics are based on continuous scores. It is used only for threshold-dependent
  classification metrics.
- This script represents the "Round I" baseline using all available descriptors.
  SHAP-based feature selection is implemented in the separate leakage-controlled Round II script.

USAGE
-----
python MUV_nested_SVM.py MUV_832

The input file must be named:
MUV_832.xlsx

and must contain a worksheet named:
MD

The first column is treated as the sample identifier, and the worksheet must contain
a binary label column named:
Class
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
EF_FRACTION = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-controlled nested-CV RBF-SVM Round I analysis for MUV molecular descriptors."
    )
    parser.add_argument(
        "file_stem",
        help="Input Excel file stem, e.g. MUV_832 for MUV_832.xlsx",
    )
    parser.add_argument(
        "--sheet",
        default="MD",
        help="Excel worksheet containing descriptors and the Class column (default: MD).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: <file_stem>_nested_svm_results",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for GridSearchCV (default: -1, all available cores).",
    )
    return parser.parse_args()


def read_data(file_path: Path, sheet_name: str) -> tuple[pd.DataFrame, pd.Series]:
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_excel(file_path, sheet_name=sheet_name, header=0, index_col=0)

    if "Class" not in df.columns:
        raise ValueError("The worksheet must contain a binary label column named 'Class'.")

    y = df["Class"].copy()
    X = df.drop(columns=["Class"]).copy()

    if y.isna().any():
        raise ValueError("The Class column contains missing values.")

    unique_labels = set(pd.unique(y))
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"The Class column must contain only 0 and 1. Found: {sorted(unique_labels)}"
        )

    # Convert all descriptor columns to numeric and stop if conversion fails.
    X = X.apply(pd.to_numeric, errors="raise")

    if X.isna().any().any():
        missing_count = int(X.isna().sum().sum())
        raise ValueError(
            f"Descriptor matrix contains {missing_count} missing values. "
            "Imputation is not included in this workflow."
        )

    X.columns = X.columns.astype(str)
    y = y.astype(int)

    return X, y


def compute_ef(
    labels: np.ndarray,
    scores: np.ndarray,
    fraction: float = EF_FRACTION,
) -> float:
    """
    Compute enrichment factor at the requested fraction.

    Stable descending sorting is used. No random shuffling is applied, ensuring
    reproducible results when scores are tied.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length.")

    active_fraction = labels.mean()
    if active_fraction <= 0:
        return np.nan

    n_top = max(1, int(np.ceil(len(labels) * fraction)))
    order = np.argsort(-scores, kind="mergesort")
    hits_top = labels[order[:n_top]].sum()

    return float((hits_top / n_top) / active_fraction)


def select_threshold_youden(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> float:
    """
    Select a decision threshold by maximizing Youden's J = sensitivity + specificity - 1.

    This function must receive predictions generated exclusively from training data
    through inner cross-validation.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr

    finite_mask = np.isfinite(thresholds)
    if not finite_mask.any():
        return 0.5

    valid_indices = np.where(finite_mask)[0]
    best_local = int(np.argmax(j[finite_mask]))
    return float(thresholds[valid_indices[best_local]])


def threshold_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            # Fitted separately within each inner/outer training partition.
            ("variance", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main() -> int:
    args = parse_args()

    file_path = Path(f"{args.file_stem}.xlsx").resolve()
    output_dir = Path(
        args.output_dir or f"{args.file_stem}_nested_svm_results"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {file_path}")
    print(f"Output directory: {output_dir}")

    X, y = read_data(file_path, args.sheet)

    class_counts = y.value_counts().sort_index()
    print(
        f"Samples: {len(y)} | Features: {X.shape[1]} | "
        f"Class 0: {class_counts.get(0, 0)} | Class 1: {class_counts.get(1, 0)}"
    )

    outer_cv = StratifiedKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Hyperparameter grid used for the nested-CV analysis.
    param_grid = {
        "svc__C": [0.1, 1.0, 10.0, 100.0],
        "svc__gamma": ["scale", 0.001, 0.01, 0.1],
    }

    fold_rows: list[dict] = []
    oof_rows: list[pd.DataFrame] = []
    roc_curves: list[tuple[np.ndarray, np.ndarray]] = []
    mean_fpr = np.linspace(0.0, 1.0, 200)

    start_total = time.perf_counter()

    for fold_number, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y),
        start=1,
    ):
        fold_start = time.perf_counter()

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        inner_cv = StratifiedKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE + fold_number,
        )

        search = GridSearchCV(
            estimator=build_pipeline(),
            param_grid=param_grid,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=args.n_jobs,
            refit=True,
            return_train_score=False,
            error_score="raise",
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # Threshold selection is confined to outer-training data.
        inner_scores = cross_val_predict(
            estimator=clone(best_model),
            X=X_train,
            y=y_train,
            cv=inner_cv,
            method="predict_proba",
            n_jobs=args.n_jobs,
        )[:, 1]
        optimized_threshold = select_threshold_youden(
            y_train.to_numpy(),
            inner_scores,
        )

        test_scores = best_model.predict_proba(X_test)[:, 1]
        fold_auc = roc_auc_score(y_test, test_scores)
        fold_ef1 = compute_ef(y_test.to_numpy(), test_scores, EF_FRACTION)
        class_metrics = threshold_metrics(
            y_test.to_numpy(),
            test_scores,
            optimized_threshold,
        )

        fpr, tpr, _ = roc_curve(y_test, test_scores)
        interpolated_tpr = np.interp(mean_fpr, fpr, tpr)
        interpolated_tpr[0] = 0.0
        roc_curves.append((mean_fpr.copy(), interpolated_tpr))

        elapsed = time.perf_counter() - fold_start

        fold_row = {
            "outer_fold": fold_number,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_actives": int(y_train.sum()),
            "test_actives": int(y_test.sum()),
            "roc_auc": float(fold_auc),
            "ef1_percent": float(fold_ef1),
            "best_inner_roc_auc": float(search.best_score_),
            "best_C": float(search.best_params_["svc__C"]),
            "best_gamma": search.best_params_["svc__gamma"],
            "runtime_seconds": float(elapsed),
            **class_metrics,
        }
        fold_rows.append(fold_row)

        fold_oof = pd.DataFrame(
            {
                "sample_id": X_test.index.astype(str),
                "outer_fold": fold_number,
                "true_label": y_test.to_numpy(),
                "score_active": test_scores,
                "optimized_threshold": optimized_threshold,
                "predicted_label": (test_scores >= optimized_threshold).astype(int),
            }
        )
        oof_rows.append(fold_oof)

        print(
            f"Fold {fold_number}/{OUTER_FOLDS} | "
            f"AUC={fold_auc:.4f} | EF1%={fold_ef1:.4f} | "
            f"C={search.best_params_['svc__C']} | "
            f"gamma={search.best_params_['svc__gamma']} | "
            f"threshold={optimized_threshold:.6f} | "
            f"time={elapsed:.1f}s"
        )

    total_elapsed = time.perf_counter() - start_total

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.concat(oof_rows, ignore_index=True)

    # Aggregate fold-level metrics.
    summary_metrics = [
        "roc_auc",
        "ef1_percent",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
    ]

    summary_rows = []
    for metric in summary_metrics:
        values = fold_df[metric].astype(float)
        summary_rows.append(
            {
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "min": values.min(),
                "max": values.max(),
            }
        )

    # Pooled out-of-fold ROC-AUC is also saved, but fold means remain the primary estimate.
    pooled_oof_auc = roc_auc_score(
        oof_df["true_label"],
        oof_df["score_active"],
    )
    pooled_oof_ef1 = compute_ef(
        oof_df["true_label"].to_numpy(),
        oof_df["score_active"].to_numpy(),
        EF_FRACTION,
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = pd.concat(
        [
            summary_df,
            pd.DataFrame(
                [
                    {
                        "metric": "pooled_oof_roc_auc",
                        "mean": pooled_oof_auc,
                        "std": np.nan,
                        "min": np.nan,
                        "max": np.nan,
                    },
                    {
                        "metric": "pooled_oof_ef1_percent",
                        "mean": pooled_oof_ef1,
                        "std": np.nan,
                        "min": np.nan,
                        "max": np.nan,
                    },
                    {
                        "metric": "total_runtime_seconds",
                        "mean": total_elapsed,
                        "std": np.nan,
                        "min": np.nan,
                        "max": np.nan,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    fold_df.to_csv(output_dir / "fold_metrics_and_hyperparameters.csv", index=False)
    oof_df.to_csv(output_dir / "out_of_fold_predictions.csv", index=False)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)

    configuration = {
        "input_file": str(file_path),
        "sheet": args.sheet,
        "n_samples": int(X.shape[0]),
        "n_input_features": int(X.shape[1]),
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
        "resampling": "none",
        "kernel": "rbf",
        "probability": True,
        "ef_fraction": EF_FRACTION,
        "threshold_method": "Youden J on inner-CV predictions from outer-training data only",
        "parameter_grid": param_grid,
    }
    with open(output_dir / "analysis_configuration.json", "w", encoding="utf-8") as fh:
        json.dump(configuration, fh, indent=2)

    # Mean ROC plot.
    mean_tpr = np.mean([curve[1] for curve in roc_curves], axis=0)
    mean_tpr[-1] = 1.0
    mean_auc_from_curve = auc(mean_fpr, mean_tpr)
    fold_auc_std = fold_df["roc_auc"].std(ddof=1)

    plt.figure(figsize=(7, 6))
    for fold_number, (fpr_grid, tpr_grid) in enumerate(roc_curves, start=1):
        plt.plot(
            fpr_grid,
            tpr_grid,
            linewidth=1,
            alpha=0.25,
            label=f"Outer fold {fold_number}",
        )

    plt.plot(
        mean_fpr,
        mean_tpr,
        linewidth=2,
        label=(
            f"Mean ROC "
            f"(AUC={fold_df['roc_auc'].mean():.3f} ± {fold_auc_std:.3f})"
        ),
    )
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Nested CV ROC curve — {args.file_stem}")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "nested_cv_mean_roc.png", dpi=300)
    plt.close()

    print("\nCompleted successfully.")
    print(
        f"Mean outer-fold ROC-AUC: "
        f"{fold_df['roc_auc'].mean():.4f} ± {fold_auc_std:.4f}"
    )
    print(
        f"Mean outer-fold EF1%: "
        f"{fold_df['ef1_percent'].mean():.4f} ± "
        f"{fold_df['ef1_percent'].std(ddof=1):.4f}"
    )
    print(f"Pooled OOF ROC-AUC: {pooled_oof_auc:.4f}")
    print(f"Total runtime: {total_elapsed:.1f} seconds")
    print(f"Results saved in: {output_dir}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
