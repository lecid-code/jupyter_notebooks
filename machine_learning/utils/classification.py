"""Core classification evaluation functions."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.axes import Axes

from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DEFAULT_TRAIN_SIZE = 0.8
DEFAULT_OUTCOME_LABEL = '1'
DEFAULT_MODEL_DESCRIPTION = 'default'

def _plot_confusion_matrix(y_true: np.ndarray | pd.Series, 
                          y_pred: np.ndarray | pd.Series, 
                          ax: Axes, title: str) -> None:
    """Plot a single confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                ax=ax, cbar=False)
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

def _plot_classification_report(y_true: np.ndarray | pd.Series,
                                y_pred: np.ndarray | pd.Series,
                                ax: Axes) -> None:
    """Plot a single classification report as text."""
    report = classification_report(y_true, y_pred)
    ax.text(0.05, 0.85, report, transform=ax.transAxes, 
            fontfamily='monospace', fontsize=10, verticalalignment='top')
    ax.axis('off')
    
def run_classifier(model: ClassifierMixin, 
                   X: np.ndarray | pd.DataFrame, 
                   y: np.ndarray | pd.Series,
                   split_random_state: int | None = None,
                   model_description: str = DEFAULT_MODEL_DESCRIPTION,
                   outcome: str = DEFAULT_OUTCOME_LABEL) -> str:
    """
    Train a classifier and return performance metrics as a CSV-formatted string.
    
    Splits the data into train/test sets (80/20), trains the model, and evaluates
    performance on both sets. Returns key metrics in a format suitable for 
    spreadsheet analysis or comparison across multiple models.
    
    Args:
        model: A scikit-learn classifier instance (must implement fit/predict).
        X: Feature matrix as numpy array or pandas DataFrame.
        y: Target vector as numpy array or pandas Series.
        split_random_state: Random seed for reproducible train/test splits. 
            If None, splits will be random.
        model_description: Human-readable name for the model, used as the first
            column in the CSV output. Defaults to 'default'.
        outcome: The target class label to report metrics for (precision, recall, f1).
            For binary classification, typically '0' or '1'. Defaults to '1'.
    
    Returns:
        A CSV-formatted string with columns:
        'model_description', train_precision, train_recall, train_f1, train_accuracy,
        test_precision, test_recall, test_f1, test_accuracy
        
        All metrics are rounded to 3 decimal places.
    
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> result = run_classifier(model, X, y, split_random_state=42, 
        ...                        model_description="Random Forest")
        >>> print(result)
        'Random Forest',0.923,0.856,0.888,0.882,0.901,0.834,0.866,0.864
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size=DEFAULT_TRAIN_SIZE, 
                                                        random_state=split_random_state)
    model = clone(model)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_report = classification_report(y_train, y_train_pred, output_dict=True)
    test_report = classification_report(y_test, y_test_pred, output_dict=True)
    
    # Extract metrics and format as CSV
    train_metrics = [train_report[outcome]['precision'], train_report[outcome]['recall'], 
                     train_report[outcome]['f1-score'], train_report['accuracy']]
    test_metrics = [test_report[outcome]['precision'], test_report[outcome]['recall'], 
                    test_report[outcome]['f1-score'], test_report['accuracy']]

    formatted_metrics = [f"{x:.3f}" for x in train_metrics + test_metrics]
    return f"'{model_description}'," + ','.join(formatted_metrics)

def print_classifier_metrics(model: ClassifierMixin, 
                             X: np.ndarray | pd.DataFrame, 
                             y: np.ndarray | pd.Series,
                             split_random_state: int | None = None) -> None:
    """
    Train a classifier and display comprehensive performance visualizations.
    
    Creates a 2x2 subplot showing confusion matrices and classification reports
    for both training and test data. Useful for detailed model evaluation and
    identifying overfitting or performance issues.
    
    The visualization includes:
    - Top row: Confusion matrices (heatmaps) for train and test data
    - Bottom row: Detailed classification reports with precision, recall, 
      f1-score, and support for each class
    
    Args:
        model: A scikit-learn classifier instance (must implement fit/predict).
        X: Feature matrix as numpy array or pandas DataFrame.
        y: Target vector as numpy array or pandas Series.
        split_random_state: Random seed for reproducible train/test splits.
            If None, splits will be random.
    
    Returns:
        None. Displays the matplotlib figure with plt.show().
    
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> print_classifier_metrics(model, X, y, split_random_state=42)
        # Displays a 2x2 subplot with confusion matrices and classification reports
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size=DEFAULT_TRAIN_SIZE, 
                                                        random_state=split_random_state)
    
    model = clone(model)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Create figure with 2 columns and 2 rows
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Classifier Performance Metrics', fontsize=16, fontweight='bold')
    
    # Plot confusion matrices (top row)
    _plot_confusion_matrix(y_train, y_train_pred, axes[0, 0], 'Training Data')
    _plot_confusion_matrix(y_test, y_test_pred, axes[0, 1], 'Testing Data')
    
    # Plot classification reports (bottom row)  
    _plot_classification_report(y_train, y_train_pred, axes[1, 0])
    _plot_classification_report(y_test, y_test_pred, axes[1, 1])
    
    plt.show()