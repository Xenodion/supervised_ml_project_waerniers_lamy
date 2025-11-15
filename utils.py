# utils.py imports
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.model_selection import KFold, cross_validate, GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def psi(expected, actual, n_bins):
    """
    Compute Population Stability Index (PSI) between two distributions.
    Works for numeric or categorical data.
    """
    expected = pd.Series(expected)
    actual = pd.Series(actual)
    
    # If numeric, bin first
    if pd.api.types.is_numeric_dtype(expected):
        bin_edges = np.histogram_bin_edges(expected, bins=n_bins)
        expected_perc = np.histogram(expected, bins=bin_edges)[0] / len(expected)
        actual_perc = np.histogram(actual, bins=bin_edges)[0] / len(actual)
    else:  # categorical
        all_categories = sorted(set(expected.unique()).union(set(actual.unique())))
        expected_perc = expected.value_counts(normalize=True).reindex(all_categories, fill_value=0).values
        actual_perc = actual.value_counts(normalize=True).reindex(all_categories, fill_value=0).values

    # Avoid division by zero
    actual_perc = np.where(actual_perc == 0, 0.0001, actual_perc)
    expected_perc = np.where(expected_perc == 0, 0.0001, expected_perc)
    
    psi_val = np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))
    return psi_val

def compare_distributions(df1, df2, n_bins):
    """
    Plot side-by-side distributions for 2 DataFrames showing %
    and compute PSI to measure data drift.
    """
    if df1.columns.tolist() != df2.columns.tolist():
        print("ERROR : Dataframes do not have the same columns")
        return
    
    sns.set(style="whitegrid")

    num_cols = df1.select_dtypes(include=["number"]).columns
    cat_cols = df1.select_dtypes(include=["object", "category"]).columns
    
    results = []

    # Numeric columns
    for col in num_cols:
        plt.figure(figsize=(10, 4))
        plt.suptitle(f"Distribution of {col}", fontsize=14)
        
        # Left-hand plot
        plt.subplot(1, 2, 1)
        sns.histplot(df1[col], bins=n_bins, kde=False, stat="percent", color="steelblue")
        plt.title("df1")
        plt.xlabel(col)
        plt.ylabel("Percentage (%)")

        # Right-hand plot
        plt.subplot(1, 2, 2)
        sns.histplot(df2[col], bins=n_bins, kde=False, stat="percent", color="orange")
        plt.title("df2")
        plt.xlabel(col)
        plt.ylabel("Percentage (%)")

        plt.tight_layout()
        plt.show()

        # Print bin edges used
        bin_edges = np.histogram_bin_edges(df1[col].dropna(), bins=n_bins)
        print(f"Bins for '{col}': {bin_edges}")

    # Categorical columns
    for col in cat_cols:
        plt.figure(figsize=(10, 4))
        plt.suptitle(f"Distribution of {col}", fontsize=14)

        # Compute normalized frequencies (%)
        df1_counts = (df1[col].value_counts(normalize=True) * 100).rename("df1_%")
        df2_counts = (df2[col].value_counts(normalize=True) * 100).rename("df2_%")

        combined = pd.concat([df1_counts, df2_counts], axis=1).fillna(0)
        combined = combined.reset_index().rename(columns={"index": col})

        # Train plot
        plt.subplot(1, 2, 1)
        sns.barplot(data=combined, y=col, x="df1_%", color="steelblue")
        plt.title("df1")
        plt.xlabel("Percentage (%)")
        plt.ylabel(col)

        # Test plot
        plt.subplot(1, 2, 2)
        sns.barplot(data=combined, y=col, x="df2_%", color="orange")
        plt.title("df2")
        plt.xlabel("Percentage (%)")
        plt.ylabel(col)

        plt.tight_layout()
        plt.show()

def plot_heatmap(data: pd.DataFrame, title: str):

    sns.set(style="white", font_scale=1.1)
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=False,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation"}
    )

    plt.title(title, fontsize=16, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt((chi2 / n) / (min(k-1, r-1)))

def bool_contact(X: pd.DataFrame):
    X = X.copy()
    X.contact.map({"cellular": 0, "telephone":1})
    return X

def train_ts(X_train: pd.DataFrame,
            y_train: pd.Series,
            pipeline: Pipeline,
            n_folds: int,
            model_name: str,
            param_grids: dict,
            scoring_metrics: dict,
            refit_metric: str,
            logs: pd.DataFrame,
            pipelines_dir: str = "saved_pipelines"):

    # Create directories if needed
    os.makedirs(pipelines_dir, exist_ok=True)

    cross_val = TimeSeriesSplit(n_splits=n_folds)

    results_list = []
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    grid_params = param_grids.get(model_name, None)
    grid_use = bool(grid_params)
    start_time = time.time()

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=grid_params,
        scoring=scoring_metrics,
        cv=cross_val,
        refit=refit_metric,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    estimator_pipeline = grid.best_estimator_
        
    # Show predictions per fold
    print("\nPredictions per fold:")
    for i, (train_idx, val_idx) in enumerate(cross_val.split(X_train)):
        X_fold = X_train.iloc[val_idx]
        y_fold_true = y_train.iloc[val_idx]
        y_fold_pred = estimator_pipeline.predict(X_fold)
        
        print(f"\nFold {i}:")
        print(f"  Class:     [0 1]")
        print(f"  True:      {np.bincount(y_fold_true)}")
        print(f"  Predicted: {np.bincount(y_fold_pred)}")

        # Metrics for this fold
        acc = accuracy_score(y_fold_true, y_fold_pred)
        prec = precision_score(y_fold_true, y_fold_pred, zero_division=0)
        rec = recall_score(y_fold_true, y_fold_pred, zero_division=0)
        f1 = f1_score(y_fold_true, y_fold_pred, zero_division=0)

        # Store metrics
        accuracy_list.append(acc)
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)
        
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        
        # Check if model predicts only one class
        unique_preds = np.unique(y_fold_pred)
        if len(unique_preds) == 1:
            print(f"WARNING: Model predicts ONLY class {unique_preds[0]}!")
    
    # Extract best metrics
    precision_mean = np.mean(precision_list)
    precision_std = np.std(precision_list)
    recall_mean = np.mean(recall_list)
    recall_std = np.std(recall_list)
    f1_mean = np.mean(f1_list)
    f1_std = np.std(f1_list)
    accuracy_mean = np.mean(accuracy_list)
    accuracy_std = np.std(accuracy_list)
    compute_time = time.time() - start_time

    # Save final pipeline
    timestamp = int(time.time())
    pipeline_filename = f"{model_name.replace(' ', '_')}_pipeline_{timestamp}.pkl"
    pipeline_path = os.path.join(pipelines_dir, pipeline_filename)
    joblib.dump(estimator_pipeline, pipeline_path)

    train_results = {
        'Model': model_name,
        'Folds': n_folds,
        'Grid_search': grid_use,
        'Precision_mean': precision_mean,
        'Precision_std': precision_std,
        'Recall_mean': recall_mean,
        'Recall_std': recall_std,
        'F1_mean': f1_mean,
        'F1_std': f1_std,
        'Accuracy_mean': accuracy_mean,
        'Accuracy_std': accuracy_std,
        'Time': compute_time,
        'Pipeline_file': pipeline_path,
    }

    results_list.append(train_results)

    # Train summary
    print(f"\nModel: {model_name}")
    print(f"Grid Search: {grid_use}")
    print(f"Accuracy : {accuracy_mean:.4f} ± {accuracy_std:.4f}")
    print(f"Precision: {precision_mean:.4f} ± {precision_std:.4f}")
    print(f"Recall   : {recall_mean:.4f} ± {recall_std:.4f}")
    print(f"F1       : {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"Time     : {compute_time:.2f}s")
    print(f"Pipeline saved at: {pipeline_path}")
    print("\n" + "#" * 70 + "\n")

    # Update logs
    logs = pd.concat([logs, pd.DataFrame(results_list)], ignore_index=True)
    logs.to_csv(os.path.join(os.getcwd(), 'data', 'train_logs.csv'), index=False)

    return logs
