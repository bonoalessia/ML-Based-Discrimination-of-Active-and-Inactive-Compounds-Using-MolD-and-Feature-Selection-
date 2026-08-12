#
#
# OVERALL DESCRIPTION:
# --------------------
# This script performs binary classification using an RBF-kernel Support Vector Machine (SVM)
# on molecular descriptor data stored in an Excel file. It applies 5-fold stratified cross-validation,
# correctly standardizes features within each fold to avoid data leakage, computes ROC-AUC metrics,
# calculates enrichment factors (EF at 1%), identifies false positives and false negatives,
# generates and saves an average ROC curve, and outputs summary statistics and misclassified samples.
#

#!/usr/bin/env python
# Specifies the interpreter path for Unix-based systems

# coding: utf-8
# Ensures UTF-8 encoding compatibility

import numpy as np                    
import pandas as pd                  
from sklearn.model_selection import StratifiedKFold  
from sklearn.metrics import roc_curve, auc, confusion_matrix s
from sklearn.svm import SVC           
from sklearn.preprocessing import StandardScaler  
import matplotlib.pyplot as plt      
import random                         
import sys                            

# --------------------
# Read input file stem from command line
# --------------------

file_stem = sys.argv[1]               # Read the first command-line argument (file name without extension)
print(f'Processing {file_stem}.xlsx') # Print which Excel file is being processed

# --------------------
# Function to read Excel data
# --------------------

def read_data(file_name, method1=None, method2=None):
    # Initialize the data variable
    data = None

    # If the first method (sheet name) is provided
    if method1 is not None:
        data1 = pd.read_excel(file_name, sheet_name=method1,
                              header=0, index_col=0)  # Read sheet into DataFrame
        data = data1                                 # Assign as default dataset

    # If the second method (sheet name) is provided
    if method2 is not None:
        data2 = pd.read_excel(file_name, sheet_name=method2,
                              header=0, index_col=0)  # Read second sheet
        if data1 is not None:
            data1 = data1.drop('Class', axis=1)       # Remove label column before concatenation
            data = pd.concat([data1, data2], axis=1,
                              ignore_index=False)     # Concatenate feature sets
        else:
            data = data2                              # Use second dataset alone

    # If no method was provided, raise an error
    if data is None:
        raise ValueError("At least one method must be specified.")
    return data

# --------------------
# Function to compute Enrichment Factor (EF)
# --------------------

def compute_enrichment(x, labels, probabilities, frac_true_actives):
    # Ensure labels and probabilities have the same length
    assert len(labels) == len(probabilities)

    # Zip probabilities and labels together
    zippedlist = list(zip(probabilities, labels))

    # Shuffle to avoid bias when probabilities are identical
    random.shuffle(zippedlist)

    # Sort in descending order of probability
    zippedlist.sort(reverse=True)

    # Determine the number of samples in the top x% subset
    nsubset = int((len(zippedlist) - 1) * x) + 1

    # Count active samples (label == 1) in the top subset
    n_active_at_ef_threshold = len(
        [x for x, y in zippedlist[:nsubset] if y == 1]
    )

    # Compute enrichment factor
    enrichment = n_active_at_ef_threshold / (nsubset * frac_true_actives)

    # Return EF value
    return enrichment
########################################
# --------------------
# Main classification function
# --------------------

def binary_classification_with_svm(file_name, data, label):
    # Convert label to NumPy array if it is a Pandas Series
    if isinstance(label, pd.Series):
        label = label.values

    # Define stratified 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize containers for metrics
    tprs = []                          # True positive rates per fold
    aucs = []                          # AUC values per fold
    mean_fpr = np.linspace(0, 1, 100)  # Common FPR grid for averaging
    fp_indices = []                    # False positive indices
    fn_indices = []                    # False negative indices
    enrichMent = []                    # Enrichment factor values

    # Cross-validation loop
    for train, test in cv.split(data, label):

        ########### STANDARD SCALER ###########
        scaler = StandardScaler()                   # Create a new scaler for this fold
        X_train = scaler.fit_transform(data.iloc[train])  # Fit on training data only
        X_test = scaler.transform(data.iloc[test])        # Transform test data using training stats
        ######################

        # Initialize SVM classifier with RBF kernel
        classifier = SVC(kernel='rbf',
                         class_weight='balanced',
                         probability=True)

        # Train the classifier
        classifier.fit(X_train, label[train])

        # Predict class probabilities for test data
        probas_ = classifier.predict_proba(X_test)

        # Extract true labels for test fold
        y_test = label[test]

        # Count active and inactive samples
        num_of_test_active = [x for x in y_test if x != 0]
        num_of_test_inactive = [x for x in y_test if x != 1]

        # Compute fraction of active samples
        frac_true_actives = len(num_of_test_active) / (
            len(num_of_test_active) + len(num_of_test_inactive)
        )

        # Compute enrichment factor at 1%
        x = 0.01
        e = compute_enrichment(x, y_test, probas_[:, 1], frac_true_actives)
        enrichMent.append(e)

        # Compute ROC curve for this fold
        fpr, tpr, thresholds = roc_curve(label[test], probas_[:, 1])

        # Interpolate TPR onto common FPR grid
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

        # Compute AUC for this fold
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Predict class labels
        predictions = classifier.predict(X_test)

        # Compute confusion matrix
        cm = confusion_matrix(label[test], predictions)

        # Identify false positives and false negatives
        fp = np.where((predictions == 1) & (label[test] == 0))[0]
        fn = np.where((predictions == 0) & (label[test] == 1))[0]

        # Store global indices of FP and FN
        fp_indices.append(test[fp])
        fn_indices.append(test[fn])

    # --------------------
    # Aggregate results across folds
    # --------------------

    mean_tpr = np.mean(tprs, axis=0)   # Average TPR
    mean_tpr[-1] = 1.0                 # Ensure curve ends at (1,1)

    mean_auc = auc(mean_fpr, mean_tpr) # Mean AUC
    std_auc = np.std(aucs)             # Standard deviation of AUC

    # Plot mean ROC curve
    plt.figure()
    plt.plot(mean_fpr, mean_tpr,
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=0.8)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC-AUC plot for {file_name}')
    plt.legend(loc="lower right")
    plt.savefig(f"{file_name}_roc_auc.png")
    plt.close()

    # Compute final statistics
    auc_avg = np.mean(aucs)
    auc_std = np.std(aucs)

    # Print summary metrics
    print(f"AUC Avg: {auc_avg}, Std: {auc_std}")

    # Concatenate FP and FN indices across folds
    fp_mapped = np.concatenate(fp_indices)
    fn_mapped = np.concatenate(fn_indices)

    # Store EF values in DataFrame
    Ef_1percent = pd.DataFrame(enrichMent)

    # Return outputs
    return Ef_1percent, auc_avg, auc_std, fp_mapped, fn_mapped

# --------------------
# Script execution starts here
# --------------------

# Read Excel data
df = read_data(f'{file_stem}.xlsx', method1='MD')

# Compute variance of each feature
variance = df.var()

# Keep only non-zero variance features
cols_to_keep = variance[variance != 0].index.tolist()
df_filtered = df[cols_to_keep]

# Separate features and labels
data = df_filtered.drop('Class', axis=1)
label = df_filtered['Class']

# Ensure feature names are strings
data.columns = data.columns.astype(str)

# Run classification
Ef_1percent, auc_avg, auc_std, fp_mapped, fn_mapped = binary_classification_with_svm(
    f'{file_stem}', data, label
)

# Extract false-positive and false-negative samples
fp_molecules = data.iloc[fp_mapped]
fn_molecules = data.iloc[fn_mapped]

# Save FP and FN samples to CSV
fp_molecules.to_csv('fp_molecules.csv', index=True)
fn_molecules.to_csv('fn_molecules.csv', index=True)

# Compute EF summary statistics
EF_avg = np.mean(Ef_1percent[0])
EF_std = np.std(Ef_1percent[0])

# Create summary table
df_summary = pd.DataFrame({
    'Metric': ['AUC Avg', 'AUC Std', 'EF Avg', 'EF Std'],
    'Value': [auc_avg, auc_std, EF_avg, EF_std]
})

# Save summary metrics to CSV
df_summary.to_csv('MD_AUC_stat.csv', index=False)
