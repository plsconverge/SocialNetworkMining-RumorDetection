import os
import sys
from typing import List, Dict, Any, Optional, Sequence

from transformers import PreTrainedTokenizerBase

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import CEDDataset


class CEDBertDataset:
    """
    Prepare text inputs for BERT-style models.

    - Supports concatenating root + early comments + late comments.
    - If concat is disabled, only the root text is used.
    - Uses tokenizer to handle special tokens / truncation.
    """

    def __init__(
        self,
        events: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        k_front: int = 10,
        m_back: int = 5,
        max_length: int = 512,
        include_numeric: bool = False,
        numeric_keys: Optional[Sequence[str]] = None,
        return_text: bool = False,
    ):
        """
        Args:
            events: List of event dicts from CEDDataset.load_all().
            tokenizer: HuggingFace tokenizer (e.g., BertTokenizerFast).
            k_front: number of earliest comments to include.
            m_back: number of latest comments to include.
            max_length: tokenizer max_length (includes special tokens).
            include_numeric: whether to return numeric features from root_info.
            numeric_keys: which numeric keys to extract from root_info.
            return_text: if True, __getitem__ also returns the concatenated raw text.
        """
        self.events = events
        self.tokenizer = tokenizer
        self.k_front = k_front
        self.m_back = m_back
        self.max_length = max_length
        self.include_numeric = include_numeric
        self.numeric_keys = numeric_keys or [
            "likes",
            "comments",
            "reposts",
            "pics",
            "followers",
            "friends",
            "messages",
        ]
        self.return_text = return_text

    def __len__(self):
        return len(self.events)

    def _select_comment_indices(self, n: int) -> List[int]:
        """Return indices for front K and back M, deduplicated."""
        idx_front = list(range(min(self.k_front, n)))
        idx_back_start = max(self.k_front, n - self.m_back)
        idx_back = list(range(idx_back_start, n))

        seen = set()
        result = []
        for i in idx_front + idx_back:
            if i not in seen:
                result.append(i)
                seen.add(i)
        return result

    def _build_text(self, event: Dict[str, Any]) -> str:
        """Build concatenated text according to concat_comments flag."""
        root_text = event["root"].get("text", "") or ""
        reposts = event["reposts"]
        # ensure chronological order by date (string sort; dataset dates are ISO-like)
        if reposts:
            reposts = sorted(reposts, key=lambda r: r.get("date", ""))
        else:
            return root_text.strip()

        indices = self._select_comment_indices(len(reposts))
        segments = [root_text.strip()]

        for i in indices:
            c_text = reposts[i].get("text", "") or ""
            if not c_text.strip():
                continue
            segments.append(c_text.strip())

        # Join with tokenizer.sep_token to mimic multi-segment input
        sep = self.tokenizer.sep_token or "[SEP]"
        return f" {sep} ".join(segments)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        event = self.events[idx]
        label = event["label"]

        text = self._build_text(event)

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # padding should be handled by a collator
            return_attention_mask=True,
        )

        item = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": label,
        }

        if self.include_numeric:
            root_info = event["root"]
            numeric_feats = []
            for key in self.numeric_keys:
                val = root_info.get(key, 0)
                numeric_feats.append(float(val) if isinstance(val, (int, float)) else 0.0)
            item["numeric_features"] = numeric_feats

        if self.return_text:
            item["text"] = text

        return item


def load_ced_for_bert(
    datapath: str,
    tokenizer: PreTrainedTokenizerBase,
    k_front: int = 10,
    m_back: int = 5,
    max_length: int = 512,
    include_numeric: bool = False,
    numeric_keys: Optional[Sequence[str]] = None,
    return_text: bool = False,
):
    """
    Convenience helper: load CED dataset and wrap with CEDBertDataset.
    """
    loader = CEDDataset(datapath)
    events = loader.load_all()

    return CEDBertDataset(
        events=events,
        tokenizer=tokenizer,
        k_front=k_front,
        m_back=m_back,
        max_length=max_length,
        include_numeric=include_numeric,
        numeric_keys=numeric_keys,
        return_text=return_text,
    )


if __name__ == "__main__":
    from transformers import BertTokenizerFast

    import os
    rootpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    example_datapath = os.path.join(rootpath, "data", "CED_Dataset")

    # Load the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

    # Call the loader function
    try:
        dataset = load_ced_for_bert(
            datapath=example_datapath,
            tokenizer=tokenizer,
            k_front=10,
            m_back=5,
            max_length=512,
            include_numeric=True,
            numeric_keys=["likes", "comments"],
            return_text=True
        )
        print(f"CEDBertDataset loaded with {len(dataset)} samples.")
        # Print a sample item
        print("Sample item:", dataset[0])
        print(len(dataset[0]["input_ids"]))  # should <= max_length
    except Exception as e:
        print(f"Smoke test failed: {e}")

