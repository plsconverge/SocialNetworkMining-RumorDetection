import os
import sys
import json
from typing import Dict, Any, List
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    BertTokenizerFast,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import CEDDataset
from data.bert_dataset import CEDBertDataset
from models.bert_classifier import BertRumorClassifier
from utils.cross_validation import cross_validate_and_ensemble
from utils.evaluation import calculate_metrics


def collate_fn_builder(tokenizer, include_numeric: bool = False):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    def collate_fn(batch):
        # data_collator expects a list of dicts with input_ids / attention_mask
        features = [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch]
        collated = data_collator(features)

        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        collated["labels"] = labels

        if include_numeric and "numeric_features" in batch[0]:
            numeric = torch.tensor([b["numeric_features"] for b in batch], dtype=torch.float)
            collated["numeric_features"] = numeric

        return collated

    return collate_fn


def train_epoch(
    model: BertRumorClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    num_epochs: int,
):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
    for step, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            numeric_features=batch.get("numeric_features"),
            labels=batch["labels"],
        )
        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate(model: BertRumorClassifier, dataloader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    all_labels = []
    all_preds = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            numeric_features=batch.get("numeric_features"),
            labels=None,
        )
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1)

        all_labels.extend(batch["labels"].cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    return calculate_metrics(all_labels, all_preds)


@torch.no_grad()
def predict(model: BertRumorClassifier, dataloader: DataLoader, device: torch.device) -> List[int]:
    """Make predictions on a dataset."""
    model.eval()
    all_preds = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            numeric_features=batch.get("numeric_features"),
            labels=None,
        )
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())

    return all_preds


def train_bert_fold(
    train_events: List[Dict],
    val_events: List[Dict],
    tokenizer: BertTokenizerFast,
    device: torch.device,
    include_numeric: bool,
    numeric_keys: List[str],
    batch_size: int = 8,
    num_epochs: int = 1,
    lr: float = 2e-5,
) -> BertRumorClassifier:
    """Train a BERT model on a fold of data."""
    train_set = CEDBertDataset(
        events=train_events,
        tokenizer=tokenizer,
        k_front=10,
        m_back=5,
        max_length=512,
        include_numeric=include_numeric,
        numeric_keys=numeric_keys,
        return_text=False,
    )

    collate_fn = collate_fn_builder(tokenizer, include_numeric=include_numeric)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


    num_numeric_features = len(numeric_keys) if include_numeric else 0
    model = BertRumorClassifier(
        pretrained_model_name="bert-base-chinese",
        num_numeric_features=num_numeric_features,
        hidden_size=256,
        num_labels=2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(num_epochs):
        train_epoch(model, train_loader, optimizer, scheduler, device, epoch, num_epochs)

    return model


def evaluate_bert_fold(model: BertRumorClassifier, val_events: List[Dict], y_val: List[int],
                       tokenizer: BertTokenizerFast, device: torch.device,
                       include_numeric: bool, numeric_keys: List[str]) -> Dict[str, Any]:
    """Evaluate a BERT model on validation data."""
    val_set = CEDBertDataset(
        events=val_events,
        tokenizer=tokenizer,
        k_front=10,
        m_back=5,
        max_length=512,
        include_numeric=include_numeric,
        numeric_keys=numeric_keys,
        return_text=False,
    )
    collate_fn = collate_fn_builder(tokenizer, include_numeric=include_numeric)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, collate_fn=collate_fn)
    return evaluate(model, val_loader, device)


def predict_bert_fold(model: BertRumorClassifier, test_events: List[Dict],
                      tokenizer: BertTokenizerFast, device: torch.device,
                      include_numeric: bool, numeric_keys: List[str]) -> List[int]:
    """Make predictions using a BERT model."""
    test_set = CEDBertDataset(
        events=test_events,
        tokenizer=tokenizer,
        k_front=10,
        m_back=5,
        max_length=512,
        include_numeric=include_numeric,
        numeric_keys=numeric_keys,
        return_text=False,
    )
    collate_fn = collate_fn_builder(tokenizer, include_numeric=include_numeric)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, collate_fn=collate_fn)
    return predict(model, test_loader, device)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train BERT model with cross-validation")
    parser.add_argument("--quick-test", action="store_true", 
                       help="Quick test mode: use small dataset and 2 folds for fast validation")
    args = parser.parse_args()
    
    rootpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    datapath = os.path.join(rootpath, "data", "CED_Dataset")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    include_numeric = True
    numeric_keys = ["likes", "followers"]
    
    # Quick test mode: use fewer folds and smaller dataset
    if args.quick_test:
        n_folds = 2
        print("\n" + "="*80)
        print("QUICK TEST MODE - Using 2 folds and limited samples for fast validation")
        print("="*80 + "\n")
    else:
        n_folds = 4

    # Load data
    base_loader = CEDDataset(datapath)
    all_events = base_loader.load_all()
    train_events, test_events, y_train, y_test = CEDDataset.split_dataset(all_events)
    
    # Quick test mode: limit dataset size
    if args.quick_test:
        max_train_samples = 200  # Use only 200 training samples
        max_test_samples = 50    # Use only 50 test samples
        train_events = train_events[:max_train_samples]
        y_train = y_train[:max_train_samples]
        test_events = test_events[:max_test_samples]
        y_test = y_test[:max_test_samples]
        print(f"⚠ Quick test mode: Using {len(train_events)} train samples, {len(test_events)} test samples\n")

    from collections import Counter
    print("Train label distribution:", dict(Counter(e["label"] for e in train_events)))
    print("Test label distribution :", dict(Counter(e["label"] for e in test_events)))
    print(f"Train events: {len(train_events)}, Test events: {len(test_events)}")

    # Prepare wrapper functions
    train_indices = list(range(len(train_events)))

    def train_fn(X_train_fold, y_train_fold, X_val_fold, y_val_fold, **kwargs):
        train_events_fold = [train_events[i] for i in X_train_fold]
        val_events_fold = [train_events[i] for i in X_val_fold]
        
        return train_bert_fold(
            train_events=train_events_fold,
            val_events=val_events_fold,
            tokenizer=tokenizer,
            device=device,
            include_numeric=include_numeric,
            numeric_keys=numeric_keys,
            **kwargs  # 这里会自动包含 cross_validate_and_ensemble 传来的 batch_size, lr, num_epochs
        )

    def evaluate_fn(model, X_val_fold, y_val_fold):
        val_events_fold = [train_events[i] for i in X_val_fold]
        return evaluate_bert_fold(
            model, val_events_fold, y_val_fold,
            tokenizer, device, include_numeric, numeric_keys
        )

    def predict_fn(model, X_test):
        return predict_bert_fold(model, X_test, tokenizer, device, include_numeric, numeric_keys)

    # Cross-validation and ensemble
    start_time = datetime.now()
    results = cross_validate_and_ensemble(
        X_train=train_indices,
        y_train=y_train,
        train_fn=train_fn,
        evaluate_fn=evaluate_fn,
        predict_fn=predict_fn,
        X_test=test_events,
        y_test=y_test,
        n_folds=n_folds,
        random_state=42,
        verbose=True,
        batch_size=8,
        num_epochs=3,
        lr=2e-5,
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
    output_file = os.path.join(output_dir, f"bert_{timestamp}.json")
    
    output_data = {
        "config": {
            "model": "bert",
            "n_folds": n_folds,
            "train_size": len(train_events),
            "test_size": len(test_events),
            "batch_size": 8,
            "num_epochs": 3,
            "lr": 2e-5,
            "include_numeric": include_numeric,
            "numeric_keys": numeric_keys,
            "pretrained_model": "bert-base-chinese",
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


