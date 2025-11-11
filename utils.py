# utils.py imports
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline

def compare_distributions(df1, df2, n_bins):
    """
    Plot side-by-side distributions for 2 DataFrames showing %
    """
    if df1.columns.tolist() == df2.columns.tolist():

        sns.set(style="whitegrid")

        num_cols = df1.select_dtypes(include=["number"]).columns
        cat_cols = df1.select_dtypes(include=["object", "category"]).columns

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

            # right-hand plot
            plt.subplot(1, 2, 2)
            sns.histplot(df2[col], bins=n_bins, kde=False, stat="percent", color="orange")
            plt.title("df2")
            plt.xlabel(col)
            plt.ylabel("Percentage (%)")

            plt.tight_layout()
            plt.show()

            # Print bin edges used
            bin_edges = np.histogram_bin_edges(df1[col].dropna(), bins=n_bins)
            print(f"Bins for '{col}':")
            print(bin_edges)
            print("-" * 80)

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
        
    else:
        print("ERROR : Dataframes do not have the same columns")


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
    X.contact.map({"cellular": 0, "telephone":1})
    return X


def train_k_fold(X_train: pd.DataFrame,
                y_train: pd.Series,
                pipeline: Pipeline,
                n_folds: int,
                model_name: str,
                param_grids: dict,
                scoring_metrics: dict,
                refit_metric: str,
                logs: pd.DataFrame,
                models_dir: str = "saved_models",
                pipelines_dir: str = "saved_pipelines"):


    # Create directories if needed
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(pipelines_dir, exist_ok=True)

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=777)
    results_list = []

    grid_params = param_grids.get(model_name, None)
    grid_use = bool(grid_params)
    start_time = time.time()

    # Kfold with gridsearch
    if grid_use:
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=grid_params,
            scoring=scoring_metrics,
            cv=kfold,
            refit=refit_metric,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        estimator_pipeline = grid.best_estimator_
        parameters = grid.best_params_

        # Extract best metrics
        best_index = grid.best_index_
        precision_mean = grid.cv_results_["mean_test_precision"][best_index]
        precision_std = grid.cv_results_["std_test_precision"][best_index]
        recall_mean = grid.cv_results_["mean_test_recall"][best_index]
        recall_std = grid.cv_results_["std_test_recall"][best_index]
        f1_mean = grid.cv_results_["mean_test_f1"][best_index]
        f1_std = grid.cv_results_["std_test_f1"][best_index]
        accuracy_mean = grid.cv_results_["mean_test_accuracy"][best_index]
        accuracy_std = grid.cv_results_["std_test_accuracy"][best_index]

    # Classic Kfold without gridsearch
    else:
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=kfold,
            scoring=scoring_metrics,
            return_train_score=False
        )

        # Compute mean/std metrics
        precision_mean = cv_results['test_precision'].mean()
        precision_std = cv_results['test_precision'].std()
        recall_mean = cv_results['test_recall'].mean()
        recall_std = cv_results['test_recall'].std()
        f1_mean = cv_results['test_f1'].mean()
        f1_std = cv_results['test_f1'].std()
        accuracy_mean = cv_results['test_accuracy'].mean()
        accuracy_std = cv_results['test_accuracy'].std()

        # Refit pipeline on ALL training data - Very important to get the best model out of the X-val
        estimator_pipeline = pipeline.fit(X_train, y_train)
        parameters = estimator_pipeline.named_steps['classifier'].get_params()

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
    print(f"Precision: {precision_mean:.4f} ± {precision_std:.4f}")
    print(f"Recall   : {recall_mean:.4f} ± {recall_std:.4f}")
    print(f"F1       : {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"Accuracy : {accuracy_mean:.4f} ± {accuracy_std:.4f}")
    print(f"Time     : {compute_time:.2f}s")
    print(f"Pipeline saved at: {pipeline_path}")
    print("\n" + "#" * 70 + "\n")

    # Update logs
    logs = pd.concat([logs, pd.DataFrame(results_list)], ignore_index=True)
    logs.to_csv(os.path.join(os.getcwd(), 'data', 'logs.csv'), index=False)

    return logs


