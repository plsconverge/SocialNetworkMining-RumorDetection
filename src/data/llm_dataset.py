import os
import sys
from typing import List, Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import CEDDataset


class CEDLLMDataset:
    """
    Prepare text inputs for LLM models (GPT, Gemini, etc.).
    
    - Supports concatenating root + early comments + late comments.
    - Uses natural language format (not tokenized).
    - Includes numeric features in the prompt.
    """

    def __init__(
        self,
        events: List[Dict[str, Any]],
        k_front: int = 20,
        m_back: int = 10,
        include_numeric: bool = True,
        numeric_keys: Optional[List[str]] = None,
    ):
        """
        Args:
            events: List of event dicts from CEDDataset.load_all().
            k_front: number of earliest comments to include.
            m_back: number of latest comments to include.
            include_numeric: whether to include numeric features in the prompt.
            numeric_keys: which numeric keys to extract from root_info.
        """
        self.events = events
        self.k_front = k_front
        self.m_back = m_back
        self.include_numeric = include_numeric
        self.numeric_keys = numeric_keys or ["likes", "followers"]
        
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
        return sorted(result)  # Keep chronological order
    
    def _get_numeric_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract numeric features from root_info."""
        root_info = event.get("root", {})
        user_info = root_info.get("user", {})
        
        # Handle case where user_info may be a string (e.g., 'empty')
        if not isinstance(user_info, dict):
            user_info = {}
        
        features = {}
        
        # Extract numeric features
        if "likes" in self.numeric_keys:
            features["likes"] = root_info.get("likes", 0)
        if "followers" in self.numeric_keys:
            features["followers"] = user_info.get("followers", 0)
        
        # Verified status (if include_numeric is True, always include)
        if self.include_numeric:
            features["verified"] = user_info.get("verified", False)
        
        return features
    
    def _build_text(self, event: Dict[str, Any]) -> str:
        """
        Build natural language text for LLM prompt.
        
        Format:
        作者信息：粉丝数[X]，点赞数[Y]，认证状态[是/否]
        
        源博文：[文本内容]
        
        评论：
        1. [评论1]
        2. [评论2]
        ...
        """
        root_text = event["root"].get("text", "") or ""
        reposts = event.get("reposts", [])
        
        # Build author info part
        parts = []
        
        if self.include_numeric:
            features = self._get_numeric_features(event)
            author_info_parts = []
            
            if "followers" in features:
                author_info_parts.append(f"粉丝数{features['followers']}")
            if "likes" in features:
                author_info_parts.append(f"点赞数{features['likes']}")
            if "verified" in features:
                verified_str = "是" if features["verified"] else "否"
                author_info_parts.append(f"认证状态{verified_str}")
            
            if author_info_parts:
                parts.append(f"作者信息：{', '.join(author_info_parts)}")
                parts.append("")  # Blank line separator
        
        # Source microblog
        parts.append(f"源博文：{root_text.strip()}")
        parts.append("")  # Blank line separator
        
        # Comments part
        if reposts:
            # Sort by time
            reposts = sorted(reposts, key=lambda r: r.get("date", ""))
            
            # Select front K and back M comments
            indices = self._select_comment_indices(len(reposts))
            
            # Filter non-empty comments and collect
            valid_comments = []
            for i in indices:
                c_text = reposts[i].get("text", "") or ""
                if c_text.strip():  # Only include non-empty comments
                    valid_comments.append(c_text.strip())
            
            if valid_comments:
                parts.append("评论：")
                for idx, comment in enumerate(valid_comments, 1):
                    parts.append(f"{idx}. {comment}")
        
        return "\n".join(parts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        event = self.events[idx]
        label = event["label"]
        
        text = self._build_text(event)
        
        return {
            "text": text,
            "label": label,
            "event": event,  # Preserve original event for debugging
        }


def load_ced_for_llm(
    datapath: str,
    k_front: int = 20,
    m_back: int = 10,
    include_numeric: bool = True,
    numeric_keys: Optional[List[str]] = None,
) -> CEDLLMDataset:
    """
    Convenience helper: load CED dataset and wrap with CEDLLMDataset.
    
    Args:
        datapath: Path to CED_Dataset directory.
        k_front: Number of earliest comments to include.
        m_back: Number of latest comments to include.
        include_numeric: Whether to include numeric features.
        numeric_keys: Which numeric keys to extract.
    
    Returns:
        CEDLLMDataset instance.
    """
    loader = CEDDataset(datapath)
    events = loader.load_all()
    
    return CEDLLMDataset(
        events=events,
        k_front=k_front,
        m_back=m_back,
        include_numeric=include_numeric,
        numeric_keys=numeric_keys,
    )


if __name__ == "__main__":
    import os
    
    rootpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    example_datapath = os.path.join(rootpath, "data", "CED_Dataset")
    
    # Test loading
    try:
        print("Loading dataset...")
        dataset = load_ced_for_llm(
            datapath=example_datapath,
            k_front=20,
            m_back=10,
            include_numeric=True,
            numeric_keys=["likes", "followers"],
        )
        print(f"CEDLLMDataset loaded with {len(dataset)} samples.")
        
        # Test several samples
        print("\n" + "="*80)
        print("Sample 1 (first event):")
        print("="*80)
        sample = dataset[0]
        print(f"Label: {sample['label']} ({'Rumor' if sample['label'] == 1 else 'Non-rumor'})")
        print(f"\nText:\n{sample['text']}")
        print("\n" + "="*80)
        print(f"Text length: {len(sample['text'])} characters")
        
        # Test another sample
        if len(dataset) > 1:
            print("\n" + "="*80)
            print("Sample 2:")
            print("="*80)
            sample2 = dataset[1]
            print(f"Label: {sample2['label']} ({'Rumor' if sample2['label'] == 1 else 'Non-rumor'})")
            print(f"\nText (first 500 chars):\n{sample2['text'][:500]}...")
            print(f"\nFull text length: {len(sample2['text'])} characters")
        
        # Print statistics for several samples
        print("\n" + "="*80)
        print("Statistics (first 10 samples):")
        print("="*80)
        num_samples = min(10, len(dataset))
        text_lengths = [len(dataset[i]['text']) for i in range(num_samples)]
        print(f"Average text length: {sum(text_lengths) / len(text_lengths):.0f} chars")
        print(f"Min length: {min(text_lengths)} chars")
        print(f"Max length: {max(text_lengths)} chars")
        
        # Count number of comments
        comment_counts = []
        for i in range(num_samples):
            event = dataset[i]['event']
            reposts = event.get('reposts', [])
            valid_comments = [r for r in reposts if r.get('text', '').strip()]
            comment_counts.append(len(valid_comments))
        
        if comment_counts:
            print(f"\nAverage valid comments per event: {sum(comment_counts) / len(comment_counts):.1f}")
            print(f"Min comments: {min(comment_counts)}")
            print(f"Max comments: {max(comment_counts)}")
        
    except Exception as e:
        print(f"Smoke test failed: {e}")
        import traceback
        traceback.print_exc()

