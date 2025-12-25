"""
Cross-validation and ensemble prediction utilities.

This module provides functions for K-fold cross-validation and ensemble prediction
across different model types (BERT, Random Forest, etc.).
"""
from typing import List, Dict, Any, Callable, Optional, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter

from .evaluation import calculate_metrics, print_metrics


def cross_validate(
    X_train: Any,
    y_train: List[int],
    train_fn: Callable[[Any, List[int], Any, List[int]], Any],
    evaluate_fn: Callable[[Any, Any, List[int]], Dict[str, Any]],
    n_folds: int = 4,
    random_state: int = 42,
    verbose: bool = True,
    **train_kwargs,
) -> Tuple[List[Any], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Perform K-fold cross-validation.
    
    Args:
        X_train: Training features (can be dataset, array, etc.)
        y_train: Training labels
        train_fn: Function that trains a model. Signature:
                  train_fn(X_train_fold, y_train_fold, X_val_fold, y_val_fold, **train_kwargs) -> model
        evaluate_fn: Function that evaluates a model. Signature:
                     evaluate_fn(model, X_val_fold, y_val_fold) -> metrics_dict
        n_folds: Number of folds (default: 4)
        random_state: Random seed for reproducibility
        verbose: Whether to print progress
        **train_kwargs: Additional arguments passed to train_fn
    
    Returns:
        Tuple of:
        - models: List of K trained models
        - fold_metrics: List of K metric dictionaries (one per fold)
        - cv_summary: Dictionary with mean and std of metrics across folds
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # For StratifiedKFold, we need to pass indices or array-like
    # Convert to numpy array for split if needed, but preserve original type for indexing
    is_numpy = isinstance(X_train, np.ndarray)
    if not is_numpy:
        # For list/other types, create a dummy array for split (same length)
        X_train_for_split = np.arange(len(X_train))
    else:
        X_train_for_split = X_train
    
    models = []
    fold_metrics = []
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"K-Fold Cross-Validation (K={n_folds})")
        print(f"{'='*80}\n")
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_for_split, y_train)):
        if verbose:
            print(f"Fold {fold_idx + 1}/{n_folds}")
            print("-" * 80)
        
        # Split data
        if is_numpy:
            X_train_fold = X_train[train_idx]
            X_val_fold = X_train[val_idx]
        else:
            X_train_fold = [X_train[i] for i in train_idx]
            X_val_fold = [X_train[i] for i in val_idx]
        
        y_train_fold = [y_train[i] for i in train_idx]
        y_val_fold = [y_train[i] for i in val_idx]
        
        # Train model
        if verbose:
            print(f"  Training on {len(X_train_fold)} samples...")
        model = train_fn(X_train_fold, y_train_fold, X_val_fold, y_val_fold, **train_kwargs)
        models.append(model)
        
        # Evaluate on validation set
        if verbose:
            print(f"  Evaluating on {len(X_val_fold)} samples...")
        metrics = evaluate_fn(model, X_val_fold, y_val_fold)
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
    
    return models, fold_metrics, cv_summary


def ensemble_predict(
    models: List[Any],
    X_test: Any,
    predict_fn: Callable[[Any, Any], List[int]],
    strategy: str = "hard_vote",
    return_individual: bool = False,
) -> Tuple[List[int], Optional[List[List[int]]]]:
    """
    Make ensemble predictions using multiple models.
    
    Args:
        models: List of K trained models
        X_test: Test features
        predict_fn: Function that makes predictions. Signature:
                    predict_fn(model, X_test) -> List[int]
        strategy: Ensemble strategy - "hard_vote" (majority voting) or "soft_vote" (average probabilities)
                  Note: "soft_vote" requires predict_proba_fn to be provided
        return_individual: Whether to return individual predictions from each model
    
    Returns:
        Tuple of:
        - ensemble_predictions: Final ensemble predictions
        - individual_predictions: List of predictions from each model (if return_individual=True)
    """
    all_predictions = []
    
    for model in models:
        preds = predict_fn(model, X_test)
        all_predictions.append(preds)
    
    # Ensemble predictions
    if strategy == "hard_vote":
        # Majority voting
        ensemble_preds = []
        for i in range(len(all_predictions[0])):
            votes = [preds[i] for preds in all_predictions]
            # Count votes and take majority
            vote_counts = Counter(votes)
            ensemble_preds.append(vote_counts.most_common(1)[0][0])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    if return_individual:
        return ensemble_preds, all_predictions
    else:
        return ensemble_preds, None


def cross_validate_and_ensemble(
    X_train: Any,
    y_train: List[int],
    train_fn: Callable[[Any, List[int], Any, List[int]], Any],
    evaluate_fn: Callable[[Any, Any, List[int]], Dict[str, Any]],
    predict_fn: Callable[[Any, Any], List[int]],
    X_test: Optional[Any] = None,
    y_test: Optional[List[int]] = None,
    n_folds: int = 4,
    random_state: int = 42,
    verbose: bool = True,
    **train_kwargs,
) -> Dict[str, Any]:
    """
    Perform K-fold cross-validation and ensemble prediction on test set.
    
    This is a unified function that combines cross-validation and ensemble prediction.
    Models are trained, evaluated on validation sets, and then used for ensemble
    prediction on the test set. Models are NOT saved to disk.
    
    Args:
        X_train: Training features (can be dataset, array, list of indices, etc.)
        y_train: Training labels
        train_fn: Function that trains a model. Signature:
                  train_fn(X_train_fold, y_train_fold, X_val_fold, y_val_fold, **train_kwargs) -> model
        evaluate_fn: Function that evaluates a model. Signature:
                     evaluate_fn(model, X_val_fold, y_val_fold) -> metrics_dict
        predict_fn: Function that makes predictions. Signature:
                    predict_fn(model, X_test) -> List[int]
        X_test: Optional test features for ensemble prediction
        y_test: Optional test labels for evaluating ensemble
        n_folds: Number of folds (default: 4)
        random_state: Random seed for reproducibility
        verbose: Whether to print progress
        **train_kwargs: Additional arguments passed to train_fn
    
    Returns:
        Dictionary containing:
        - fold_metrics: List of K metric dictionaries (one per fold)
        - cv_summary: Dictionary with mean and std of metrics across folds
        - ensemble_predictions: Ensemble predictions on test set (if X_test provided)
        - ensemble_metrics: Metrics for ensemble on test set (if X_test and y_test provided)
        - individual_test_metrics: List of metrics for each model on test set (if X_test and y_test provided)
        
        Note: Models are not returned as they are only used for prediction and then discarded.
    """
    # Perform cross-validation
    models, fold_metrics, cv_summary = cross_validate(
        X_train=X_train,
        y_train=y_train,
        train_fn=train_fn,
        evaluate_fn=evaluate_fn,
        n_folds=n_folds,
        random_state=random_state,
        verbose=verbose,
        **train_kwargs,
    )
    
    result = {
        "fold_metrics": fold_metrics,
        "cv_summary": cv_summary,
        "ensemble_predictions": None,
        "ensemble_metrics": None,
        "individual_test_metrics": None,
    }
    
    # Ensemble prediction on test set if provided
    if X_test is not None:
        if verbose:
            print("\n" + "="*80)
            print("Ensemble Prediction on Test Set")
            print("="*80)
        
        ensemble_preds, individual_preds = ensemble_predict(
            models=models,
            X_test=X_test,
            predict_fn=predict_fn,
            strategy="hard_vote",
            return_individual=True,
        )
        result["ensemble_predictions"] = ensemble_preds
        
        # Evaluate ensemble if test labels provided
        if y_test is not None:
            ensemble_metrics = calculate_metrics(y_test, ensemble_preds)
            result["ensemble_metrics"] = ensemble_metrics
            
            if verbose:
                print_metrics(ensemble_metrics, title="Ensemble Test Results")
            
            # Evaluate individual models on test set
            individual_test_metrics = []
            if verbose:
                print("\n" + "="*80)
                print("Individual Model Performance on Test Set")
                print("="*80)
            
            for i, preds in enumerate(individual_preds):
                metrics = calculate_metrics(y_test, preds)
                individual_test_metrics.append(metrics)
                if verbose:
                    print(f"Model {i+1}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
            
            result["individual_test_metrics"] = individual_test_metrics
        
    
    return result

