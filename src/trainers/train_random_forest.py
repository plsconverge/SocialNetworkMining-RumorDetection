import os
import sys
import json
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier

from data.data_loader import CEDDataset
from data.feature_engineer import FeatureEngineer
from utils.cross_validation import cross_validate_and_ensemble
from utils.evaluation import calculate_metrics


def train_rf_fold(
    X_train_fold: np.ndarray,
    y_train_fold: List[int],
    *args,
    max_depth: int = 3,
    random_state: int = 42,
    **kwargs
) -> RandomForestClassifier:
    """Train a Random Forest model on a fold of data."""
    rf = RandomForestClassifier(max_depth=max_depth, random_state=random_state, **kwargs)
    rf.fit(X_train_fold, y_train_fold)
    return rf


def evaluate_rf_fold(
    model: RandomForestClassifier,
    X_val_fold: np.ndarray,
    y_val_fold: List[int]
) -> Dict[str, Any]:
    """Evaluate a Random Forest model on validation data."""
    y_pred = model.predict(X_val_fold)
    return calculate_metrics(y_val_fold, y_pred.tolist())


def predict_rf_fold(
    model: RandomForestClassifier,
    X_test: np.ndarray
) -> List[int]:
    """Make predictions using a Random Forest model."""
    return model.predict(X_test).tolist()


def main():
    rootpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    datapath = os.path.join(rootpath, r'data//CED_Dataset')

    n_folds = 4
    max_depth = 3

    loader = CEDDataset(datapath)
    dataset = loader.load_all()
    train_set, test_set, y_train, y_test = loader.split_dataset(dataset)

    # Extract features
    print("Extracting features...")
    features_train = [FeatureEngineer.extract_features_advanced(event) for event in train_set]
    features_test = [FeatureEngineer.extract_features_advanced(event) for event in test_set]

    X_train = FeatureEngineer.convert_to_dataframe(features_train)
    X_test = FeatureEngineer.convert_to_dataframe(features_test)

    X_train_array = X_train.values
    X_test_array = X_test.values
    y_train_list = y_train if isinstance(y_train, list) else y_train.tolist() if hasattr(y_train, 'tolist') else list(y_train)
    y_test_list = y_test if isinstance(y_test, list) else y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)

    print(f"Train events: {len(train_set)}, Test events: {len(test_set)}")
    print(f"Feature dimension: {X_train_array.shape[1]}")

    # Cross-validation and ensemble
    start_time = datetime.now()
    results = cross_validate_and_ensemble(
        X_train=X_train_array,
        y_train=y_train_list,
        train_fn=train_rf_fold,
        evaluate_fn=evaluate_rf_fold,
        predict_fn=predict_rf_fold,
        X_test=X_test_array,
        y_test=y_test_list,
        n_folds=n_folds,
        random_state=42,
        verbose=True,
        max_depth=max_depth,
    )
    elapsed_seconds = (datetime.now() - start_time).total_seconds()
    
    # Print final summary
    print("\n" + "="*80)
    print("Final Summary")
    print("="*80)
    print(f"Cross-Validation Accuracy: {results['cv_summary']['accuracy']['mean']:.4f} ± {results['cv_summary']['accuracy']['std']:.4f}")
    print(f"Cross-Validation F1: {results['cv_summary']['f1']['mean']:.4f} ± {results['cv_summary']['f1']['std']:.4f}")
    if results['ensemble_metrics']:
        print(f"Ensemble Test Accuracy: {results['ensemble_metrics']['accuracy']:.4f}")
        print(f"Ensemble Test F1: {results['ensemble_metrics']['f1']:.4f}")
    
    # Save results
    output_dir = os.path.join(rootpath, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"random_forest_{timestamp}.json")
    
    output_data = {
        "config": {
            "model": "random_forest",
            "n_folds": n_folds,
            "train_size": len(train_set),
            "test_size": len(test_set),
            "feature_dimension": X_train_array.shape[1],
            "max_depth": max_depth,
            "random_state": 42,
        },
        "cv_summary": {
            "accuracy": {
                "mean": float(results['cv_summary']['accuracy']['mean']),
                "std": float(results['cv_summary']['accuracy']['std']),
            },
            "f1": {
                "mean": float(results['cv_summary']['f1']['mean']),
                "std": float(results['cv_summary']['f1']['std']),
            },
            "precision": {
                "mean": float(results['cv_summary']['precision']['mean']),
                "std": float(results['cv_summary']['precision']['std']),
            },
            "recall": {
                "mean": float(results['cv_summary']['recall']['mean']),
                "std": float(results['cv_summary']['recall']['std']),
            },
        },
        "ensemble_metrics": {
            "accuracy": float(results['ensemble_metrics']['accuracy']),
            "f1": float(results['ensemble_metrics']['f1']),
            "precision": float(results['ensemble_metrics']['precision']),
            "recall": float(results['ensemble_metrics']['recall']),
        } if results['ensemble_metrics'] else None,
        "fold_metrics": [
            {
                "accuracy": float(m['accuracy']),
                "f1": float(m['f1']),
                "precision": float(m['precision']),
                "recall": float(m['recall']),
            }
            for m in results['fold_metrics']
        ],
        "individual_test_metrics": [
            {
                "accuracy": float(m['accuracy']),
                "f1": float(m['f1']),
                "precision": float(m['precision']),
                "recall": float(m['recall']),
            }
            for m in results['individual_test_metrics']
        ] if results['individual_test_metrics'] else None,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed_seconds,
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()