import os
import sys
import json
from typing import Dict, Any, List
from datetime import datetime

import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.gcn_dataset import GCNDataset
from models.gnn_classifier import GCNModel, GCNFNModel
from utils.cross_validation import cross_validate_and_ensemble
from utils.evaluation import calculate_metrics


def train_epoch(
    model: GCNModel | GCNFNModel,
    dataloader: PyGDataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
    for batch in progress_bar:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)
        loss = torch.nn.functional.nll_loss(out, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate(model: GCNModel | GCNFNModel, dataloader: PyGDataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    all_labels = []
    all_preds = []

    for batch in dataloader:
        batch = batch.to(device)
        out = model(batch)
        preds = torch.argmax(out, dim=-1)

        all_labels.extend(batch.y.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    return calculate_metrics(all_labels, all_preds)


@torch.no_grad()
def predict(model: GCNModel | GCNFNModel, dataloader: PyGDataLoader, device: torch.device) -> List[int]:
    """Make predictions on a dataset."""
    model.eval()
    all_preds = []

    for batch in dataloader:
        batch = batch.to(device)
        out = model(batch)
        preds = torch.argmax(out, dim=-1)
        all_preds.extend(preds.cpu().tolist())

    return all_preds


def train_gcn_fold(
    train_dataset: GCNDataset,
    val_dataset: GCNDataset,
    device: torch.device,
    num_features: int,
    num_hidden: int = 128,
    num_classes: int = 2,
    batch_size: int = 32,
    num_epochs: int = 100,
    lr: float = 0.001,
) -> GCNModel | GCNFNModel:
    """Train a GCN model on a fold of data."""
    train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    args = {
        'num_features': num_features,
        'num_hidden': num_hidden,
        'num_classes': num_classes
    }
    model = GCNModel(args, cat=False).to(device)
    # model = GCNFNModel(args).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train_epoch(model, train_loader, optimizer, device, epoch, num_epochs)

    return model


def evaluate_gcn_fold(model: GCNModel | GCNFNModel, val_dataset: GCNDataset, y_val: List[int],
                      device: torch.device, batch_size: int = 32) -> Dict[str, Any]:
    """Evaluate a GCN model on validation data."""
    val_loader = PyGDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return evaluate(model, val_loader, device)


def predict_gcn_fold(model: GCNModel | GCNFNModel, test_dataset: GCNDataset,
                      device: torch.device, batch_size: int = 32) -> List[int]:
    """Make predictions using a GCN model."""
    test_loader = PyGDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return predict(model, test_loader, device)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train GCN model with cross-validation")
    parser.add_argument("--quick-test", action="store_true",
                       help="Quick test mode: use small dataset and 2 folds for fast validation")
    args = parser.parse_args()

    rootpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    datapath = os.path.join(rootpath, "data")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Quick test mode: use fewer folds and smaller dataset
    if args.quick_test:
        n_folds = 2
        print("\n" + "="*80)
        print("QUICK TEST MODE - Using 2 folds and limited samples for fast validation")
        print("="*80 + "\n")
    else:
        n_folds = 4

    # Load dataset
    # PyG automatically checks if processed data exists:
    # - If CED_Processed/CED_data_features.pt exists, it loads directly
    # - If not, it calls process() which decides whether to run preprocess()
    print("Loading GCN dataset...")
    dataset = GCNDataset(root=datapath, empty=False, pre_process=False)
    print(f"Total samples: {len(dataset)}")

    # Extract labels for stratified splitting
    y = [data.y.item() for data in dataset]

    # Quick test mode: limit dataset size
    if args.quick_test:
        max_samples = 100  # Use only 100 samples
        indices = list(range(min(max_samples, len(dataset))))
        dataset = dataset[indices]
        y = y[:max_samples]
        print(f"⚠ Quick test mode: Using {len(dataset)} samples\n")

    from collections import Counter
    print("Label distribution:", dict(Counter(y)))

    # Calculate num_features from first graph
    sample_graph = dataset[0]
    num_features = sample_graph.x.shape[1]
    print(f"Number of features: {num_features}")

    # Split dataset using the static method
    print("Splitting dataset...")
    train_val_dataset, test_dataset, y_train_val, y_test = GCNDataset.split_dataset(
        dataset, test_size=0.2, random_state=42
    )

    print("\nData Split:")
    print(f"  Train+Val size: {len(train_val_dataset)} (for {n_folds}-fold CV)")
    print(f"  Test size     : {len(test_dataset)} (hold-out)")
    from collections import Counter
    print(f"  Test label dist: {dict(Counter(y_test))}\n")

    # Prepare wrapper functions
    # We pass indices relative to the train_val_dataset
    train_indices = list(range(len(train_val_dataset)))

    def train_fn(X_train_fold, y_train_fold, X_val_fold, y_val_fold, **kwargs):
        # Indices are relative to train_val_dataset
        train_subset = train_val_dataset[X_train_fold]
        val_subset = train_val_dataset[X_val_fold]

        return train_gcn_fold(
            train_dataset=train_subset,
            val_dataset=val_subset,
            device=device,
            num_features=num_features,
            **kwargs
        )

    def evaluate_fn(model, X_val_fold, y_val_fold):
        val_subset = train_val_dataset[X_val_fold]
        return evaluate_gcn_fold(
            model, val_subset, y_val_fold, device
        )

    def predict_fn(model, X_test):
        # X_test here will be passed from cross_validate_and_ensemble
        # When predicting on the hold-out test set, X_test will be the test_dataset itself
        # (or indices if we passed indices, but we'll pass the dataset object directly for the test part)
        
        # However, cross_validate_and_ensemble treats X_test as the data input.
        # If we pass test_dataset as X_test to cross_validate_and_ensemble,
        # it will be passed here.
        return predict_gcn_fold(model, X_test, device)

    # Cross-validation and ensemble
    start_time = datetime.now()
    results = cross_validate_and_ensemble(
        X_train=train_indices,
        y_train=y_train_val,
        train_fn=train_fn,
        evaluate_fn=evaluate_fn,
        predict_fn=predict_fn,
        X_test=test_dataset,  # Pass the dataset object directly
        y_test=y_test,
        n_folds=n_folds,
        random_state=42,
        verbose=True,
        batch_size=32,
        num_epochs=100,
        lr=0.001,
        num_hidden=128,
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
    output_file = os.path.join(output_dir, f"gcn_{timestamp}.json")

    output_data = {
        "config": {
            "model": "gcn",
            "n_folds": n_folds,
            "dataset_size": len(dataset),
            "num_features": num_features,
            "batch_size": 32,
            "num_epochs": 100,
            "lr": 0.001,
            "num_hidden": 128,
            "num_classes": 2,
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

    # Validation checks
    print("\n" + "="*80)
    print("Validation Checks")
    print("="*80)

    # Check 1: Results structure
    required_keys = ['config', 'cv_summary', 'ensemble_metrics', 'fold_metrics', 'timestamp', 'elapsed_seconds']
    missing_keys = [k for k in required_keys if k not in output_data]
    if missing_keys:
        print(f"❌ Missing keys in results: {missing_keys}")
    else:
        print("✓ All required keys present in results")

    # Check 2: CV summary structure
    cv_keys = ['accuracy', 'f1', 'precision', 'recall']
    cv_valid = all(k in output_data['cv_summary'] for k in cv_keys)
    if cv_valid:
        print("✓ CV summary structure valid")
    else:
        print("❌ CV summary structure invalid")

    # Check 3: Number of folds
    if len(output_data['fold_metrics']) == n_folds:
        print(f"✓ Correct number of fold metrics: {n_folds}")
    else:
        print(f"❌ Expected {n_folds} fold metrics, got {len(output_data['fold_metrics'])}")

    # Check 4: Ensemble metrics
    if output_data['ensemble_metrics']:
        ensemble_keys = ['accuracy', 'f1', 'precision', 'recall']
        if all(k in output_data['ensemble_metrics'] for k in ensemble_keys):
            print("✓ Ensemble metrics structure valid")
        else:
            print("❌ Ensemble metrics structure invalid")
    else:
        print("⚠ No ensemble metrics (test set may not have been provided)")

    # Check 5: Metrics are in valid range [0, 1]
    all_metrics_valid = True
    for fold_metric in output_data['fold_metrics']:
        for metric_name in ['accuracy', 'f1', 'precision', 'recall']:
            value = fold_metric[metric_name]
            if not (0 <= value <= 1):
                print(f"❌ Invalid metric value: {metric_name}={value} (should be in [0, 1])")
                all_metrics_valid = False
    if all_metrics_valid:
        print("✓ All metrics in valid range [0, 1]")

    # Check 6: File was created
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"✓ Results file created: {file_size} bytes")
    else:
        print(f"❌ Results file not found: {output_file}")

    print("="*80)

    if args.quick_test:
        print("\n" + "="*80)
        print("QUICK TEST COMPLETE - All checks passed!")
        print("You can now run without --quick-test for full training")
        print("="*80)


if __name__ == "__main__":
    main()
