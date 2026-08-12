#!/usr/bin/env python
# coding: utf-8

# Packages we need to have (Install)
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
import shap
import os
import time
import sys

file_stem = sys.argv[1]
print(f'Processing {file_stem}.xlsx')


def read_data(file_name, method1=None, method2=None):
    # Initialize data as None
    data = None
    
    # Conditionally load data
    if method1 is not None:
        data1 = pd.read_excel(file_name, sheet_name=method1, header=0, index_col=0)
        data = data1  # Assume data1 is the default data

    if method2 is not None:
        data2 = pd.read_excel(file_name, sheet_name=method2, header=0, index_col=0)
        if data1 is not None:  # If data1 was also loaded
            data1 = data1.drop('Class', axis=1)
            data = pd.concat([data1, data2], axis=1, ignore_index=False)
        else:
            data = data2  # If only data2 was loaded

    # If both method1 and method2 are None, you might want to raise an error or handle it differently
    if data is None:
        raise ValueError("At least one method must be specified.")
    
    return data


def compute_enrichment(x, labels, probabilities, frac_true_actives):
    '''x is the enrichment factor threshold as a fraction (e.g. 0.1 for EF10%)
       True-labels is a list of 0 or 1 for each molecule
       probabilities gives the probability for each prediction (used to sort the lists)
       frac_true_actives is the fraction of the whole testing dataset that is active
       returns the enrichment factor at the level given by 'x'

    Notes:
       EF(x%) = {Hits_found_in_subset/N_mols_in_subset} / {Hits_total/Ntotal}   
    or
       EF(x%) = frac_hits_found_in_subset_to_x% / frac_mols_in_subset_to_x%
    '''
    
    # assumes labels are 1 and 0 (for 'active' and 'inactive')    
    assert(len(labels) == len(probabilities))
    zippedlist = list(zip(probabilities, labels))
        
    random.shuffle(zippedlist)     # avoid accidental enrichment if lots of entries have same probability and list is pre-sorted
    zippedlist.sort()
    zippedlist.sort(reverse=True)
    
    nsubset = int((len(zippedlist) - 1) * x) + 1
    n_active_at_ef_threshold = len([x for x, y in zippedlist[:nsubset] if y == 1])

    # EF(x%) = {Hits_selected/N_subset} / {Hits_total/Ntotal}
    # EF(x%) = {Hits_selected/N_subset} / {frac_true_actives}
    enrichment = n_active_at_ef_threshold / (nsubset * frac_true_actives)
    
    return enrichment


from sklearn.base import BaseEstimator, ClassifierMixin

# Define a wrapper class for your SVC model to make it compatible with SHAP
class ShapSVCWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


def binary_classification_with_svm(file_name, data, label):
    # Make sure the label variable is in the right format for the next steps
    if isinstance(label, pd.Series):
        label = label.values
        
    # Prepare the cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Standardize the features
    #scaler = StandardScaler()
    #data_scaled = scaler.fit_transform(data)
    
    # Initialize lists to store metrics and SHAP values
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    shap_values_list = []
    fp_indices = []
    fn_indices = []
    enrichMent = []
    
    # Perform cross-validation
    for train, test in cv.split(data, label):
        ########### STANDARD SCALER ###########
        scaler = StandardScaler()
        X_train = scaler.fit_transform(data.iloc[train])
        X_test = scaler.transform(data.iloc[test])
        ########### STANDARD SCALER ###########

        classifier = SVC(kernel='rbf', class_weight='balanced', probability=True)
        classifier.fit(X_train, label[train])

        probas_ = classifier.predict_proba(X_test)
        
        
        y_test = label[test]
        num_of_test_active = [x for x in y_test if x != 0]
        num_of_test_inactive = [x for x in y_test if x != 1]
        frac_true_actives = len(num_of_test_active) / (len(num_of_test_inactive) + len(num_of_test_active))
        x = 0.01  # EF percentage
        e = compute_enrichment(x, y_test, probas_[:, 1], frac_true_actives)
        enrichMent.append(e)
        
        # Compute ROC curve and area of the curve
        fpr, tpr, thresholds = roc_curve(label[test], probas_[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3)
        
        # Wrap the classifier for SHAP compatibility
        shap_model_wrapper = ShapSVCWrapper(classifier)
        
        num_features = data_scaled[train].shape[1]
        min_max_evals = 2 * num_features + 1
        explainer = shap.Explainer(shap_model_wrapper.predict, data_scaled[train], max_evals=min_max_evals)
        
        shap_values = explainer(data_scaled[test])
        shap_values_list.append(shap_values)
        
        # Compute false positives and negatives
        predictions = classifier.predict(data_scaled[test])
        cm = confusion_matrix(label[test], predictions)
        fp = np.where((predictions == 1) & (label[test] == 0))[0]
        fn = np.where((predictions == 0) & (label[test] == 1))[0]
        fp_indices.append(test[fp])
        fn_indices.append(test[fn])
    
    # Plot average ROC-AUC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=0.8)
    # Add the diagonal dashed line
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    
    plt.title(f"ROC-AUC plot for {file_name}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(f"{file_name}_roc_auc.png")
    
    # Calculate and print average and standard deviation for AUC, FP, and FN
    auc_avg = np.mean(aucs)
    auc_std = np.std(aucs)
    
    fp_avg = np.mean([len(fp) for fp in fp_indices])
    fn_avg = np.mean([len(fn) for fn in fn_indices])
    print(f"AUC Avg: {auc_avg}, Std: {auc_std}")
    
    # Aggregate SHAP values across folds and select top 10 important features
    shap_values_agg = np.abs(np.concatenate([shap_values.values for shap_values in shap_values_list], axis=0)).mean(axis=0)
    # If we want to check top 10 or top 20, then change the following accordingly. 
    top_indices = np.argsort(shap_values_agg)[-25:]
    feature_names = data.columns[top_indices]
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': shap_values_agg[top_indices]})
    
    # Map FP and FN indices to original data for further investigation
    fp_mapped = np.concatenate(fp_indices)
    fn_mapped = np.concatenate(fn_indices)
    Ef_1percent = pd.DataFrame(enrichMent)
    return Ef_1percent, auc_avg, auc_std, importance_df, fp_mapped, fn_mapped


df = read_data(f'{file_stem}.xlsx', method1='MD')

# Remove the columns with ZERO variance in the Data
variance = df.var()
cols_to_keep = variance[variance != 0].index.tolist()
df_filtered = df[cols_to_keep]

# Assuming 'data' contains features and 'Class' is the label column
data = df_filtered.drop('Class', axis=1)  # Adjust 'label' to your actual label column name
label = df_filtered['Class']
label.value_counts()  # Just counting how many ligands and Decoys we have (Ligands = 1, Decoys = 0)

data.columns = data.columns.astype(str)

Ef_1percent, auc_avg, auc_std, importance_df, fp_mapped, fn_mapped = binary_classification_with_svm(f'{file_stem}', data, label)

# Sort the DataFrame by importance for better visualization
importance_df_sorted = importance_df.sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(importance_df_sorted['Feature'], importance_df_sorted['Importance'], color='skyblue')
plt.xlabel('SHAP Importance')
plt.title(f'MD Top 25 Important Features for {file_stem} Target')
# Save the figure as 'Figure1.png' before displaying it
plt.savefig('MD.png', dpi=300)
plt.show()

# Map indices back to molecule identifiers
fp_molecules = data.iloc[fp_mapped]
fn_molecules = data.iloc[fn_mapped]

# Save the false-positive molecule indices and their corresponding features
fp_molecules.to_csv('fp_molecules.csv', index=True)

# Save the false-negative molecule indices and their corresponding features
fn_molecules.to_csv('fn_molecules.csv', index=True)

# Save the top 25 important features as a CSV file
importance_df.to_csv('Top25_important_MD.csv', index=False)

# This cell prepares and saves both Metrics (AUC and EF)
EF_avg = np.mean(Ef_1percent[0])
EF_std = np.std(Ef_1percent[0])
df_summary = pd.DataFrame({'Metric': ['AUC Avg', 'AUC Std', 'EF Avg', 'EF Std'],
                           'Value': [auc_avg, auc_std, EF_avg, EF_std]})

df_summary.to_csv('MD_AUC_stat.csv')
df_summary

