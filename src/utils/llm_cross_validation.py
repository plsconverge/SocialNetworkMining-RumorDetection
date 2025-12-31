"""
LLM-specific cross-validation and ensemble utilities.

This module provides functions for K-fold cross-validation and temperature-based
ensemble prediction for LLM models.
"""
from typing import List, Dict, Any, Callable, Optional, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter

from .evaluation import calculate_metrics, print_metrics


async def cross_validate_llm(
    X_train: List[Any],
    y_train: List[int],
    classify_fn: Callable[[List[Any], List[int]], Any],
    n_folds: int = 4,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Perform K-fold cross-validation for LLM models.
    
    Args:
        X_train: Training features (list of events/dicts)
        y_train: Training labels
        classify_fn: Async function that classifies a batch. Signature:
                     await classify_fn(events, labels) -> List[Dict] (results with 'label' key)
        n_folds: Number of folds (default: 4)
        random_state: Random seed for reproducibility
        verbose: Whether to print progress
    
    Returns:
        Tuple of:
        - fold_metrics: List of K metric dictionaries (one per fold)
        - cv_summary: Dictionary with mean and std of metrics across folds
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Convert to numpy array for split
    X_train_array = np.arange(len(X_train))
    
    fold_metrics = []
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"K-Fold Cross-Validation (K={n_folds})")
        print(f"{'='*80}\n")
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_array, y_train)):
        if verbose:
            print(f"Fold {fold_idx + 1}/{n_folds}")
            print("-" * 80)
        
        # Split data
        X_val_fold = [X_train[i] for i in val_idx]
        y_val_fold = [y_train[i] for i in val_idx]
        
        # Classify validation set
        if verbose:
            print(f"  Classifying {len(X_val_fold)} validation samples...")
        
        results = await classify_fn(X_val_fold, y_val_fold)
        
        # Calculate metrics
        pred_labels = [r["label"] for r in results]
        metrics = calculate_metrics(pred_labels, y_val_fold)
        fold_metrics.append(metrics)
        
        if verbose:
            print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
            print()
    
    # Calculate summary statistics
    cv_summary = {
        "accuracy": {
            "mean": np.mean([m["accuracy"] for m in fold_metrics]),
            "std": np.std([m["accuracy"] for m in fold_metrics]),
        },
        "f1": {
            "mean": np.mean([m["f1"] for m in fold_metrics]),
            "std": np.std([m["f1"] for m in fold_metrics]),
        },
        "precision": {
            "mean": np.mean([m["precision"] for m in fold_metrics]),
            "std": np.std([m["precision"] for m in fold_metrics]),
        },
        "recall": {
            "mean": np.mean([m["recall"] for m in fold_metrics]),
            "std": np.std([m["recall"] for m in fold_metrics]),
        },
    }
    
    if verbose:
        print(f"{'='*80}")
        print("Cross-Validation Summary")
        print(f"{'='*80}")
        print(f"Accuracy: {cv_summary['accuracy']['mean']:.4f} ± {cv_summary['accuracy']['std']:.4f}")
        print(f"F1 Score: {cv_summary['f1']['mean']:.4f} ± {cv_summary['f1']['std']:.4f}")
        print(f"Precision: {cv_summary['precision']['mean']:.4f} ± {cv_summary['precision']['std']:.4f}")
        print(f"Recall: {cv_summary['recall']['mean']:.4f} ± {cv_summary['recall']['std']:.4f}")
        print()
    
    return fold_metrics, cv_summary


async def ensemble_predict_temperature(
    X_test: List[Any],
    classify_fn: Callable[[List[Any], List[float], Optional[List[int]]], Any],
    temperatures: List[float] = [0.0, 0.1, 0.3, 0.5],
    y_test: Optional[List[int]] = None,
    verbose: bool = True,
    strategy: str = "hard_vote",
) -> Tuple[List[int], Optional[Dict[str, Any]], Optional[List[List[int]]], Optional[List[List[Dict[str, Any]]]]]:
    """
    Make ensemble predictions using different temperatures (hard voting).
    
    Args:
        X_test: Test features (list of events/dicts)
        classify_fn: Async function that classifies with temperature. Signature:
                     await classify_fn(events, temperatures, labels) -> List[Dict]
                     Note: temperatures is a list, one per sample (can be same for all)
        temperatures: List of temperatures to use for ensemble
        y_test: Optional test labels for evaluation
        verbose: Whether to print progress
        strategy: Ensemble strategy - "hard_vote" (majority voting)
    
    Returns:
        Tuple of:
        - ensemble_predictions: Final ensemble predictions
        - ensemble_metrics: Metrics for ensemble (if y_test provided)
        - individual_predictions: List of label predictions for each temperature
        - individual_results: List of full results (with confidence, reason, etc.) for each temperature
    """

  
    if verbose:
        print("\n" + "="*80)
        print(f"Temperature Ensemble Prediction (Temperatures: {temperatures})")
        print("="*80)
    
    all_predictions = []
    all_results = []  # Store full results with confidence, reason, etc.
    
    # For each temperature, classify all test samples
    for temp_idx, temp in enumerate(temperatures):
        if verbose:
            print(f"\nTemperature {temp_idx + 1}/{len(temperatures)}: {temp}")
            print(f"  Classifying {len(X_test)} samples...")
        
        # Create temperature list (same temperature for all samples)
        temp_list = [temp] * len(X_test)
        
        results = await classify_fn(X_test, temp_list, y_test)
        pred_labels = [r["label"] for r in results]
        all_predictions.append(pred_labels)
        all_results.append(results)  # Store full results
        
        if verbose and y_test:
            metrics = calculate_metrics(pred_labels, y_test)
            print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    
    # Ensemble predictions (hard voting)
    ensemble_preds = []
    for i in range(len(X_test)):
        votes = [preds[i] for preds in all_predictions]
        vote_counts = Counter(votes)
        ensemble_preds.append(vote_counts.most_common(1)[0][0])
    
    # Evaluate ensemble if test labels provided
    ensemble_metrics = None
    if y_test:
        ensemble_metrics = calculate_metrics(ensemble_preds, y_test)
        if verbose:
            print("\n" + "="*80)
            print("Ensemble Test Results")
            print("="*80)
            print_metrics(ensemble_metrics, title="Ensemble Test Results")
    
    return ensemble_preds, ensemble_metrics, all_predictions, all_results

