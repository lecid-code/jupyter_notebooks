"""
ML utilities package for model evaluation and performance analysis.

This package provides convenient tools for training and evaluating 
machine learning classifiers with both numerical metrics and visualizations.
"""

from .classification import run_classifier, print_classifier_metrics

__version__ = "1.0.0"
__all__ = ['run_classifier', 'print_classifier_metrics']