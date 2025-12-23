"""
Train and evaluate LLM-based rumor classifier using Gemini API.

This script loads data, builds prompts, and classifies using async IO.
"""
import os
import sys
import asyncio
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import CEDDataset
from data.llm_dataset import CEDLLMDataset
from prompts.rumor_detection_prompt import build_prompt
from models.llm_classifier import classify_batch, calculate_metrics

try:
    from google import genai
except ImportError:
    print("Error: google-genai is required.")
    print("Install it with: uv pip install google-genai")
    sys.exit(1)


def setup_gemini_api() -> genai.Client:
    """Setup Gemini API with API key from environment variable and return client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        print("\nTo set it:")
        print("  Windows PowerShell: $env:GEMINI_API_KEY='your-api-key'")
        print("  Linux/Mac: export GEMINI_API_KEY='your-api-key'")
        print("\nGet your API key from: https://aistudio.google.com/app/apikey")
        sys.exit(1)
    
    client = genai.Client(api_key=api_key)
    print(f"✓ Gemini API configured (key: {api_key[:10]}...)")
    return client


def progress_callback(current: int, total: int):
    """Print progress during batch classification."""
    percentage = (current / total) * 100
    print(f"\rProgress: {current}/{total} ({percentage:.1f}%)", end="", flush=True)


async def main():
    """Main function to run LLM-based rumor detection."""
    # Setup
    client = setup_gemini_api()
    
    # Configuration
    rootpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    datapath = os.path.join(rootpath, "data", "CED_Dataset")
    
    # Model configuration
    MODEL_NAME = "gemini-2.5-flash-lite"
    MAX_CONCURRENT = 15     
    MAX_RETRIES = 4         
    TEMPERATURE = 0.0       # deterministic
    MAX_OUTPUT_TOKENS = 256 # allow longer JSON and avoid truncation
    
    # Data configuration
    K_FRONT = 25
    M_BACK = 15
    TEST_SAMPLE_SIZE = None # None = use all test set, or set to 50/100 for POC
    
    print("\n" + "="*80)
    print("LLM-based Rumor Detection (Gemini)")
    print("="*80)
    print(f"Model: {MODEL_NAME}")
    print(f"Max concurrent: {MAX_CONCURRENT}")
    print(f"Data: {datapath}")
    print("="*80 + "\n")
    
    # Load dataset - use same split as BERT baseline
    print("Loading dataset...")
    base_loader = CEDDataset(datapath)
    all_events = base_loader.load_all()
    # Use same split parameters as BERT: test_size=0.2, seed=42 (default)
    train_events, test_events, y_train, y_test = CEDDataset.split_dataset(all_events)
    
    # Quick sanity check: label distribution in train/test after split (same as BERT)
    from collections import Counter
    train_label_counts = Counter(e["label"] for e in train_events)
    test_label_counts = Counter(e["label"] for e in test_events)
    print("Train label distribution:", dict(train_label_counts))
    print("Test label distribution :", dict(test_label_counts))
    
    print(f"\nTotal events: {len(all_events)}")
    print(f"Train events: {len(train_events)}")
    print(f"Test events: {len(test_events)}")
    
    # Sample test set if needed (for POC)
    if TEST_SAMPLE_SIZE and TEST_SAMPLE_SIZE < len(test_events):
        import random
        random.seed(42) 
        indices = random.sample(range(len(test_events)), TEST_SAMPLE_SIZE)
        test_events = [test_events[i] for i in indices]
        y_test = [y_test[i] for i in indices]
        print(f"\n⚠ Using sample of {TEST_SAMPLE_SIZE} events for testing (POC mode)")
    
    # Prepare LLM dataset
    print("\nPreparing LLM dataset...")
    test_dataset = CEDLLMDataset(
        events=test_events,
        k_front=K_FRONT,
        m_back=M_BACK,
        include_numeric=True,
        numeric_keys=["likes", "followers"],
    )
    
    # Build prompts
    print("Building prompts...")
    prompts = []
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        prompt = build_prompt(
            sample['text'],
            use_few_shot=True,  # add compact few-shot to steady outputs
        )
        prompts.append(prompt)
    
    print(f"Built {len(prompts)} prompts")
    print(f"Average prompt length: {sum(len(p) for p in prompts) / len(prompts):.0f} characters")
    
    # Classify
    print(f"\nStarting classification (concurrency: {MAX_CONCURRENT})...")
    start_time = datetime.now()
    
    results = await classify_batch(
        prompts=prompts,
        client=client,
        model_name=MODEL_NAME,
        max_concurrent=MAX_CONCURRENT,
        max_retries=MAX_RETRIES,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        progress_callback=progress_callback,
    )
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print(f"\n\nClassification complete! Time used: {elapsed:.1f} seconds")
    print(f"Average per sample: {elapsed / len(results):.2f} seconds")
    
    # Calculate metrics
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    
    metrics = calculate_metrics(results, y_test)
    
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score (Macro): {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    print("\nClassification Report:")
    print(metrics['report'])
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    print("\nStatistics:")
    stats = metrics['statistics']
    print(f"  Total: {stats['total']}")
    print(f"  Parse Errors: {stats['parse_errors']} ({stats['parse_error_rate']*100:.1f}%)")
    print(f"  API Errors: {stats['api_errors']} ({stats['api_error_rate']*100:.1f}%)")
    print(f"  Avg. Confidence: {stats['avg_confidence']:.3f}")
    
    # API error diagnostics
    api_error_details = [
        (r.get("api_error_type"), r.get("api_error"))
        for r in results
        if r.get("api_error")
    ]
    if api_error_details:
        from collections import Counter
        err_counter = Counter(api_error_details)
        print("\nTop API errors (type, message):")
        for (err_type, err_msg), cnt in err_counter.most_common(5):
            short_msg = (err_msg or "").strip().split("\n")[0][:120]
            print(f"  {cnt}x  {err_type}: {short_msg}")
    
    # Save results
    output_dir = os.path.join(rootpath, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"llm_results_{timestamp}.json")
    
    output_data = {
        "config": {
            "model": MODEL_NAME,
            "max_concurrent": MAX_CONCURRENT,
            "k_front": K_FRONT,
            "m_back": M_BACK,
            "test_size": len(test_events),
        },
        "metrics": {
            "accuracy": float(metrics['accuracy']),
            "f1": float(metrics['f1']),
            "precision": float(metrics['precision']),
            "recall": float(metrics['recall']),
        },
        "statistics": stats,
        "elapsed_seconds": elapsed,
    }
    
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print misclassified cases for quick diagnosis
    mis_idx = [i for i, r in enumerate(results) if r["label"] != y_test[i]]
    if mis_idx:
        print("\n" + "="*80)
        print(f"Misclassified ({len(mis_idx)} cases)")
        print("="*80)
        for i in mis_idx:
            r = results[i]
            print(f"\nIdx {i+1}: true={y_test[i]}, pred={r['label']}, conf={r['confidence']:.3f}")
            if r.get("second_pass"):
                print("  Note: second-pass result")
            print(f"  Reason: {r['reason']}")
            # Show root microblog text instead of full prompt
            root_text = ""
            try:
                root_text = test_events[i]["root"].get("text", "") or ""
            except Exception:
                pass
            root_preview = root_text.replace("\n", " ")
            print(f"  Root text: {root_preview[:200]}{'...' if len(root_preview) > 200 else ''}")


if __name__ == "__main__":
    asyncio.run(main())

