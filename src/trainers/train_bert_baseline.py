import os
import sys
from typing import Dict, Any

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
    from sklearn.metrics import accuracy_score, f1_score, classification_report

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

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    # Explicitly fix label order to 0/1 so that even如果某一类在y_true中缺失也不会报错
    report = classification_report(
        all_labels,
        all_preds,
        labels=[0, 1],
        target_names=["Non-Rumor", "Rumor"],
        zero_division=0,
    )

    return {"accuracy": acc, "f1": f1, "report": report}


def main():
    rootpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    datapath = os.path.join(rootpath, "data", "CED_Dataset")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

    include_numeric = True
    numeric_keys = ["likes", "followers"]  # simple, strong features

    # Load events and use original stratified split from CEDDataset
    base_loader = CEDDataset(datapath)
    all_events = base_loader.load_all()
    train_events, test_events, y_train, y_test = CEDDataset.split_dataset(all_events)

    # Quick sanity check: label distribution in train/test after split
    from collections import Counter

    train_label_counts = Counter(e["label"] for e in train_events)
    test_label_counts = Counter(e["label"] for e in test_events)
    print("Train label distribution:", dict(train_label_counts))
    print("Test label distribution :", dict(test_label_counts))

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

    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_numeric_features = len(numeric_keys) if include_numeric else 0
    model = BertRumorClassifier(
        pretrained_model_name="bert-base-chinese",
        num_numeric_features=num_numeric_features,
        hidden_size=256,
        num_labels=2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_epochs = 1
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, num_epochs)
        print(f"Train loss: {train_loss:.4f}")

        metrics = evaluate(model, test_loader, device)
        print(f"Eval accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        print("Classification report:")
        print(metrics["report"])


if __name__ == "__main__":
    main()


