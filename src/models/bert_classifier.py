import os
import sys
from typing import Optional

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import BertModel, PreTrainedTokenizerBase


class BertRumorClassifier(nn.Module):
    """
    BERT-based classifier for rumor detection.

    - Uses [CLS] representation from BERT.
    - Optionally concatenates numeric features (e.g., followers, likes).
    """

    def __init__(
        self,
        pretrained_model_name: str = "bert-base-chinese",
        num_numeric_features: int = 0,
        hidden_size: int = 256,
        num_labels: int = 2,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        bert_dim = self.bert.config.hidden_size

        input_dim = bert_dim + num_numeric_features if num_numeric_features > 0 else bert_dim

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels),
        )

        self.num_numeric_features = num_numeric_features

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numeric_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            numeric_features: (batch, num_numeric_features), optional
            labels: (batch,), optional
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        cls_repr = self.dropout(cls_repr)

        if self.num_numeric_features > 0 and numeric_features is not None:
            x = torch.cat([cls_repr, numeric_features], dim=-1)
        else:
            x = cls_repr

        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels.long())

        return {"loss": loss, "logits": logits}



