import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def calculate_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate comprehensive classification metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model for display
        
    Returns:
    --------
    dict
        Dictionary containing all metrics
    """
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_class_0': precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        'recall_class_0': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        'f1_class_0': f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        'precision_class_1': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'recall_class_1': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'f1_class_1': f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    }
    
    return metrics


def print_metrics(metrics):
    """
    Print classification metrics in a formatted way.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing metrics from calculate_metrics
    """
    print("=" * 80)
    print(f"MODEL: {metrics['model_name']}")
    print("=" * 80)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    
    print(f"\nClass 0 (Fake News) Metrics:")
    print(f"  Precision: {metrics['precision_class_0']:.4f}")
    print(f"  Recall:    {metrics['recall_class_0']:.4f}")
    print(f"  F1-Score:  {metrics['f1_class_0']:.4f}")
    
    print(f"\nClass 1 (Real News) Metrics:")
    print(f"  Precision: {metrics['precision_class_1']:.4f}")
    print(f"  Recall:    {metrics['recall_class_1']:.4f}")
    print(f"  F1-Score:  {metrics['f1_class_1']:.4f}")
    print("=" * 80)


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'], 
                yticklabels=['Fake', 'Real'],
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def print_classification_report(y_true, y_pred, model_name="Model"):
    """
    Print detailed classification report.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model
    """
    print(f"\nClassification Report - {model_name}")
    print("=" * 80)
    print(classification_report(y_true, y_pred, 
                                target_names=['Fake', 'Real'],
                                digits=4))
    print("=" * 80)


def compare_models(results_list):
    """
    Compare multiple models and create a comparison dataframe.
    
    Parameters:
    -----------
    results_list : list
        List of metric dictionaries from calculate_metrics
        
    Returns:
    --------
    pd.DataFrame
        Comparison dataframe
    """
    comparison_df = pd.DataFrame(results_list)
    comparison_df = comparison_df.set_index('model_name')
    
    # Select key metrics for comparison
    key_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    comparison_df = comparison_df[key_metrics]
    
    return comparison_df


def plot_model_comparison(results_list, metric='accuracy', save_path=None):
    """
    Plot comparison of models for a specific metric.
    
    Parameters:
    -----------
    results_list : list
        List of metric dictionaries from calculate_metrics
    metric : str
        Metric to compare (accuracy, precision, recall, f1_score)
    save_path : str, optional
        Path to save the plot
    """
    df = pd.DataFrame(results_list)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['model_name'], df[metric], color='steelblue', alpha=0.7)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(f'Model Comparison - {metric.replace("_", " ").title()}', 
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_all_metrics_comparison(results_list, save_path=None):
    """
    Plot comparison of all key metrics across models.
    
    Parameters:
    -----------
    results_list : list
        List of metric dictionaries from calculate_metrics
    save_path : str, optional
        Path to save the plot
    """
    df = pd.DataFrame(results_list)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(df))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for i, metric in enumerate(metrics):
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, df[metric], width, 
                     label=metric.replace('_', ' ').title(), alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison - All Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['model_name'], rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def perform_cross_validation(model, X, y, cv=5, scoring='f1_weighted'):
    """
    Perform cross-validation on a model.
    
    Parameters:
    -----------
    model : sklearn model
        Model to evaluate
    X : array-like
        Features
    y : array-like
        Labels
    cv : int
        Number of folds
    scoring : str
        Scoring metric
        
    Returns:
    --------
    dict
        Cross-validation results
    """
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    results = {
        'mean_score': cv_scores.mean(),
        'std_score': cv_scores.std(),
        'scores': cv_scores
    }
    
    return results


def print_cv_results(cv_results, model_name="Model"):
    """
    Print cross-validation results.
    
    Parameters:
    -----------
    cv_results : dict
        Results from perform_cross_validation
    model_name : str
        Name of the model
    """
    print(f"\nCross-Validation Results - {model_name}")
    print("=" * 80)
    print(f"Mean Score: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
    print(f"Individual Fold Scores: {cv_results['scores']}")
    print("=" * 80)

