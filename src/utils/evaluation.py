"""
Unified evaluation functions for all models.

This module provides consistent evaluation metrics across different models
(BERT, Random Forest, Logistic Regression, etc.).
"""
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    labels: Optional[List[int]] = None,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        labels: Label order for metrics (default: [0, 1])
        target_names: Names for labels in report (default: ["Non-Rumor", "Rumor"])
    
    Returns:
        Dictionary containing:
        - accuracy: float
        - f1: float (macro average)
        - precision: float (macro average)
        - recall: float (macro average)
        - report: str (classification report)
        - confusion_matrix: np.ndarray
    """
    if labels is None:
        labels = [0, 1]
    if target_names is None:
        target_names = ["Non-Rumor", "Rumor"]
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "report": report,
        "confusion_matrix": cm,
    }


def print_metrics(
    metrics: Dict[str, Any],
    title: str = "Evaluation Results",
    show_confusion_matrix: bool = True,
) -> None:
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary returned by calculate_metrics
        title: Title for the output section
        show_confusion_matrix: Whether to print confusion matrix
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score (Macro): {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    print("\nClassification Report:")
    print(metrics["report"])
    
    if show_confusion_matrix:
        print("\nConfusion Matrix:")
        print(metrics["confusion_matrix"])

