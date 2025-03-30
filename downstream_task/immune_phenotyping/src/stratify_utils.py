import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, RocCurveDisplay
import statistics

def load_data(args):
    '''
    Load and preprocess data for immune phenotyping analysis.
    Args:
        args (argparse.Namespace): Command line arguments containing paths to data files.  
    Returns:
        df_merged (pd.DataFrame): Merged DataFrame containing predictions and ground truth data.
    '''
    os.makedirs(args.save_path, exist_ok=True)
    csv_path = glob.glob(args.saved_path_analysis + '/immune_phenotype-pred-' + '*.csv')[0]
    df_pred = pd.read_csv(csv_path, index_col=None)
    df_pred.rename(columns={'Case_ID': 'tupro_id'}, inplace=True)

    meta = pd.read_csv(args.meta_path, sep='\t', index_col=[0])[['tupro_id', 'cd8_phenotype_revised', 'pathology_immune_diagnosis']]
    df_gt = pd.ExcelFile(args.immune_gt_path).parse("Tabelle1")
    df_gt = preprocess_ground_truth(df_gt, meta)

    df_gt_labels = pd.ExcelFile(args.gt_labels_path).parse("Sheet1")
    df_gt_labels.rename(columns={'sample': 'tupro_id'}, inplace=True)
    df_gt = pd.merge(df_gt, df_gt_labels, on='tupro_id')

    df_annotations = pd.ExcelFile(args.annotation_meta_path).parse('Sheet 1 - melanoma-merged_clini')
    df_annotations = preprocess_annotations(df_annotations)
    df_gt_annotations = pd.merge(df_gt, df_annotations, on='tupro_id')

    df_merged = pd.merge(df_pred, df_gt_annotations, on='tupro_id')
    df_merged = df_merged[~(df_merged['Exclude'] == 1)]
    return df_merged


def preprocess_ground_truth(df_gt, meta):
    '''
    Preprocess the ground truth data by renaming columns, filtering rows, and calculating additional metrics.
    Args:
        df_gt (pd.DataFrame): DataFrame containing ground truth data.
        meta (pd.DataFrame): DataFrame containing metadata.
    Returns:
        df_gt (pd.DataFrame): Preprocessed DataFrame containing ground truth data.
    '''
    df_gt = df_gt[['Case_ID', 'Analysis_Region', 'Revised immune diagnosis', 'Density Tumor', 'Density Stroma total', 'Tumor Area (um²)',
                   'Positive Lymphocytes Area (um²)', 'Stroma Area (um²)', 'Tumor:_AP_Positive_Cells', 'Stroma:_AP_Positive_Cells', 'Positive_Lymphocytes:_AP_Positive_Cells']]
    df_gt['Total_Stroma:_AP_Positive_Cells'] = df_gt['Stroma:_AP_Positive_Cells'] + df_gt['Positive_Lymphocytes:_AP_Positive_Cells']
    df_gt['Total Stromal area'] = df_gt['Stroma Area (um²)'] + df_gt['Positive Lymphocytes Area (um²)']
    df_gt.drop(columns=['Stroma Area (um²)', 'Positive Lymphocytes Area (um²)', 'Stroma:_AP_Positive_Cells', 'Positive_Lymphocytes:_AP_Positive_Cells'], inplace=True)
    df_gt = df_gt[~df_gt['Analysis_Region'].isin(['IM', 'Peritumoral'])]
    df_gt.rename(columns={'Case_ID': 'tupro_id'}, inplace=True)
    return pd.merge(df_gt, meta, on='tupro_id')


def preprocess_annotations(df_annotations):
    '''
    Preprocess the annotations data by renaming columns and filtering rows.
    Args:
        df_annotations (pd.DataFrame): DataFrame containing annotations data.
    Returns:
        df_annotations (pd.DataFrame): Preprocessed DataFrame containing annotations data.
        '''
    df_annotations.columns = df_annotations.iloc[0]
    df_annotations = df_annotations[1:]
    df_annotations = df_annotations[['sample_id', 'S', 'biopsy_localisation', 'Comment', 'Exclude']]
    df_annotations.rename(columns={'sample_id': 'tupro_id', 'S': 'annotation_quality'}, inplace=True)
    df_annotations = df_annotations[~df_annotations['annotation_quality'].str.contains('*', regex=False, na=False)]
    return df_annotations


def calculate_correlations(df_merged):
    '''
    Calculate and print the Spearman and Pearson correlations between iCD8 density and tumor/stroma density.
    Args:
        df_merged (pd.DataFrame): Merged DataFrame containing predictions and ground truth data.
    Returns:
        None
    '''
    
    print("sCorr iCD8 density:", df_merged['iCD8_density_TC'].corr(df_merged['Density Tumor'], method='spearman'))
    print("pCorr iCD8 density:", df_merged['iCD8_density_TC'].corr(df_merged['Density Tumor'], method='pearson'))
    print("sCorr sCD8 density:", df_merged['sCD8_density_TC'].corr(df_merged['Density Stroma total'], method='spearman'))
    print("pCorr sCD8 density:", df_merged['sCD8_density_TC'].corr(df_merged['Density Stroma total'], method='pearson'))


def plot_boxplots(df_merged, save_path, class_type):
    '''
    Plot boxplots of CD8 density for different immune phenotypes and save the figures.
    Args:
        df_merged (pd.DataFrame): Merged DataFrame containing predictions and ground truth data.
        save_path (str): Path to save the boxplot figures.
        class_type (str): Type of classification ('3class' or '2class').
    Returns:
        None
    '''
    custom_order = ['desert', 'excluded', 'inflamed'] if class_type == '3class' else ['cold', 'hot']
    if class_type == '3class':
        custom_colors = [cm.get_cmap('terrain')(0.14), cm.get_cmap('terrain')(0.3), cm.get_cmap('jet')(0.75)]
    else:  # For '2class'
        color_cold = tuple((x + y) / 2 for x, y in zip(cm.get_cmap('terrain')(0.14), cm.get_cmap('terrain')(0.3)))
        custom_colors = {'cold': color_cold, 'hot': cm.get_cmap('jet')(0.75)}
    annotation_quality_labels = {2: 'best', 1: 'acceptable', 0: 'poor'}
    plot_col_list = ['iCD8_density_TC', 'sCD8_density_TC']

    for col in plot_col_list:
        fig, axes = plt.subplots(1, 4, figsize=(4 * 7, 5))
        sns.boxplot(x=f'cd8_phenotype_revised{"_" if class_type == "2class" else ""}', y=col, data=df_merged, order=custom_order, palette=custom_colors, ax=axes[0])
        axes[0].set_xlabel('Immune Phenotypes')
        axes[0].set_ylabel(col)

        fig_single, ax_single = plt.subplots(figsize=(5, 5))
        sns.boxplot(x=f'cd8_phenotype_revised{"_" if class_type == "2class" else ""}', y=col, data=df_merged, order=custom_order, palette=custom_colors, ax=ax_single)
        ax_single.set_ylabel('Density')
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        fig_single.savefig(os.path.join(save_path, f'{class_type}_{col}_all.svg'), bbox_inches='tight', dpi=300)
        fig_single.savefig(os.path.join(save_path, f'{class_type}_{col}_all.png'), bbox_inches='tight', dpi=300)
        plt.close(fig_single)

        for i, (key, value) in enumerate(annotation_quality_labels.items(), start=1):
            filtered_df = df_merged[df_merged['annotation_quality'] == key]
            sns.boxplot(x=f'cd8_phenotype_revised{"_" if class_type == "2class" else ""}', y=col, data=filtered_df, order=custom_order, palette=custom_colors, ax=axes[i])
            axes[i].set_title(f'annotation score: {value}')

            fig_single, ax_single = plt.subplots(figsize=(5, 5))
            sns.boxplot(x=f'cd8_phenotype_revised{"_" if class_type == "2class" else ""}', y=col, data=filtered_df, order=custom_order, palette=custom_colors, ax=ax_single)
            ax_single.set_ylabel('Density')
            plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            fig_single.savefig(os.path.join(save_path, f'{class_type}_{col}_{value}.svg'), bbox_inches='tight', dpi=300)
            fig_single.savefig(os.path.join(save_path, f'{class_type}_{col}_{value}.png'), bbox_inches='tight', dpi=300)
            plt.close(fig_single)


def map_labels(df_merged):
    '''
    Map the CD8 phenotype labels to numerical values and filter the DataFrame based on annotation quality.
    Args:
        df_merged (pd.DataFrame): Merged DataFrame containing predictions and ground truth data.
    Returns:
        df_merged (pd.DataFrame): Merged DataFrame with mapped labels and filtered data.
    '''
    annotation_quality_labels = {2: 'best', 1: 'acceptable', 0: 'poor'}
    df_merged['annotation_quality_'] = df_merged['annotation_quality'].map(annotation_quality_labels)
    df_merged['cd8_phenotype_revised__'] = df_merged['cd8_phenotype_revised_'].map({'cold': 0, 'hot': 1})
    return df_merged


def custom_stratified_splits(X, y, n_splits):
    '''
    Create custom stratified splits for cross-validation, ensuring that each fold contains at least one instance of each class.
    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target Series.
        n_splits (int): Number of splits for cross-validation.
    Returns:
        splits (list): List of tuples containing train and test indices for each fold.
    '''
    # Create indices for each class
    hot_indices = np.where(y == 1)[0]
    cold_indices = np.where(y == 0)[0]
    
    # Shuffle indices with a fixed random seed
    np.random.shuffle(hot_indices)
    np.random.shuffle(cold_indices)
    
    # Create folds manually
    splits = []
    hot_per_fold = 1  # Ensure at least one "hot" class per fold

    for i in range(n_splits):
        test_hot = hot_indices[i * hot_per_fold : (i + 1) * hot_per_fold]
        test_cold = cold_indices[i::n_splits]
        test_indices = np.concatenate([test_hot, test_cold])
        train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
        splits.append((train_indices, test_indices))
    return splits


def perform_cross_validation(X, y, splits, clf):
    '''
    Perform cross-validation using the provided splits and classifier.
    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target Series.
        splits (list): List of tuples containing train and test indices for each fold.
        clf (sklearn.base.BaseEstimator): Classifier to be used for cross-validation.
    Returns:
        accuracy_scores (list): List of accuracy scores for each fold.
        f1_scores (list): List of F1 scores for each fold.
        roc_auc_scores (list): List of ROC AUC scores for each fold.
    '''
    accuracy_scores = []
    f1_scores = []
    roc_auc_scores = []

    for train_indices, test_indices in splits:
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        
        # Train the classifier
        clf.fit(X_train, y_train)
        
        # Predictions and probabilities
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]  # Probability for class 1 ("hot")
        
        # Metrics calculation
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
        roc_auc_scores.append(roc_auc_score(y_test, y_prob, average='macro'))
    return accuracy_scores, f1_scores, roc_auc_scores


def print_metrics(accuracy_scores, f1_scores, roc_auc_scores):
    '''
    Print the mean and standard deviation of accuracy, F1 score, and ROC AUC scores.
    Args:
        accuracy_scores (list): List of accuracy scores for each fold.
        f1_scores (list): List of F1 scores for each fold.
        roc_auc_scores (list): List of ROC AUC scores for each fold.
    Returns:
        None
    '''
    print("Accuracy: ", round(statistics.mean(accuracy_scores), 3), "±", round(statistics.stdev(accuracy_scores), 3))
    print("F1 Score: ", round(statistics.mean(f1_scores), 3), "±", round(statistics.stdev(f1_scores), 3))
    print("AUC-ROC: ", round(statistics.mean(roc_auc_scores), 3), "±", round(statistics.stdev(roc_auc_scores), 3))


def run_classification(df_merged):
    '''
    Run classification using Random Forest classifier and custom stratified splits.
    Args:
        df_merged (pd.DataFrame): Merged DataFrame containing predictions and ground truth data.
    Returns:
        None
        '''
    # Map labels
    df_merged = map_labels(df_merged)

    # Filter dataset for annotation quality
    filtered_df = df_merged[df_merged['annotation_quality'] == 2]

    X = filtered_df[['Density Tumor', 'Density Stroma total']]
    y = filtered_df['cd8_phenotype_revised__']

    # Create custom splits
    n_splits = 3
    splits = custom_stratified_splits(X, y, n_splits)

    # Initialize classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=75684)

    # Perform Cross-Validation
    accuracy_scores, f1_scores, roc_auc_scores = perform_cross_validation(X, y, splits, clf)

    # Print metrics
    print_metrics(accuracy_scores, f1_scores, roc_auc_scores)