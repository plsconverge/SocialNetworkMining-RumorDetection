import os
import sys
import random
from typing import List

# make `src` importable 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import BertTokenizerFast

from data.data_loader import CEDDataset


def sample_events(dataset: List[dict], max_events: int = 500, seed: int = 42) -> List[dict]:
    """Randomly sample a subset of events for quick statistics."""
    if len(dataset) <= max_events:
        return dataset
    random.seed(seed)
    return random.sample(dataset, max_events)


def compute_token_lengths(
    events: List[dict],
    tokenizer: BertTokenizerFast,
    k_front: int = 8,
    m_back: int = 4,
) -> None:
    """
    Compute token length statistics for:
      - source weibo (root)
      - individual comments
      - concatenated sequence with [CLS] root [SEP] comments...[SEP]
    """
    root_lengths: List[int] = []
    comment_lengths: List[int] = []
    concat_lengths: List[int] = []

    for ev in events:
        root = ev["root"]
        reposts = ev["reposts"]

        root_text = root.get("text", "") or ""
        root_len = len(tokenizer.tokenize(root_text))
        root_lengths.append(root_len)

        # all comments
        for r in reposts:
            c_text = r.get("text", "") or ""
            if not c_text.strip():
                continue
            c_len = len(tokenizer.tokenize(c_text))
            comment_lengths.append(c_len)

        # front K + back M comments (for a hypothetical concatenation)
        if not reposts:
            concat_lengths.append(1 + root_len + 1)  # [CLS] root [SEP]
            continue

        n = len(reposts)
        idx_front = list(range(min(k_front, n)))
        idx_back_start = max(k_front, n - m_back)
        idx_back = list(range(idx_back_start, n))
        # avoid duplicate indices if n < k_front + m_back
        indices = []
        for i in idx_front + idx_back:
            if i not in indices:
                indices.append(i)

        total = 1 + root_len + 1  # [CLS] + root + [SEP]
        for i in indices:
            c_text = reposts[i].get("text", "") or ""
            if not c_text.strip():
                continue
            c_len = len(tokenizer.tokenize(c_text))
            # +1 for each [SEP] between segments
            total += c_len + 1
        concat_lengths.append(total)

    def describe(name: str, values: List[int]) -> None:
        if not values:
            print(f"{name}: no data")
            return
        import numpy as np

        arr = np.array(values)
        print(f"\n=== {name} ===")
        print(f"count: {len(arr)}")
        print(f"min  : {arr.min():.2f}")
        print(f"max  : {arr.max():.2f}")
        print(f"mean : {arr.mean():.2f}")
        for q in [50, 75, 90, 95, 99]:
            print(f"p{q:>2}: {np.percentile(arr, q):.2f}")

    describe("Root token length", root_lengths)
    describe("Comment token length", comment_lengths)

    # For concatenated sequence, also report how many exceed BERT's 512 limit
    if concat_lengths:
        import numpy as np

        arr = np.array(concat_lengths)
        describe(f"Concat length (K={k_front}, M={m_back})", concat_lengths)
        over_512 = (arr > 512).sum()
        print(f"\nSequences > 512 tokens: {over_512} / {len(arr)} "
              f"({over_512 / len(arr) * 100:.2f}%)")


def main():
    rootpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    datapath = os.path.join(rootpath, r"data", "CED_Dataset")

    print(f"Loading dataset from: {datapath}")
    loader = CEDDataset(datapath)
    all_events = loader.load_all()
    print(f"Total events loaded: {len(all_events)}")

    # sample a subset for quick analysis
    events = sample_events(all_events, max_events=500, seed=42)
    print(f"Sampled events for analysis: {len(events)}")

    # initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

    # you can tweak k_front / m_back here
    k_front = 10
    m_back = 5
    print(f"\nAnalyzing token lengths with K={k_front}, M={m_back} ...")
    compute_token_lengths(events, tokenizer, k_front=k_front, m_back=m_back)


if __name__ == "__main__":
    main()


