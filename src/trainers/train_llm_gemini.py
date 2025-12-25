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
from utils.llm_cross_validation import cross_validate_llm

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
    """Main function to run LLM-based rumor detection with cross-validation."""
    import json
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LLM model with cross-validation")
    parser.add_argument("--quick-test", action="store_true",
                       help="Quick test mode: use small dataset, 2 folds, and 2 temperatures for fast validation")
    args = parser.parse_args()
    
    # Setup
    client = setup_gemini_api()
    
    # Configuration
    rootpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    datapath = os.path.join(rootpath, "data", "CED_Dataset")
    
    # Model configuration
    MODEL_NAME = "gemini-2.5-flash-lite"
    MAX_CONCURRENT = 15     
    MAX_RETRIES = 4         
    MAX_OUTPUT_TOKENS = 256
    
    # Data configuration
    K_FRONT = 25
    M_BACK = 15
    
    # Quick test mode: use fewer folds and smaller dataset
    if args.quick_test:
        n_folds = 2
        print("\n" + "="*80)
        print("QUICK TEST MODE - Using 2 folds and limited samples for fast validation")
        print("="*80 + "\n")
    else:
        n_folds = 4
    
    # Use best single temperature (0.1) based on analysis
    TEST_TEMPERATURE = 0.1
    
    print("\n" + "="*80)
    print("LLM-based Rumor Detection (Gemini) with Cross-Validation")
    print("="*80)
    print(f"Model: {MODEL_NAME}")
    print(f"Max concurrent: {MAX_CONCURRENT}")
    print(f"Cross-validation folds: {n_folds}")
    print(f"Test temperature: {TEST_TEMPERATURE} (best single temperature)")
    print(f"Data: {datapath}")
    print("="*80 + "\n")
    
    # Load dataset
    print("Loading dataset...")
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
    
    # Helper function to build prompts from events
    def build_prompts_from_events(events):
        """Build prompts from a list of events."""
        dataset = CEDLLMDataset(
            events=events,
            k_front=K_FRONT,
            m_back=M_BACK,
            include_numeric=True,
            numeric_keys=["likes", "followers"],
        )
        prompts = []
        for i in range(len(dataset)):
            sample = dataset[i]
            prompt = build_prompt(sample['text'], use_few_shot=True)
            prompts.append(prompt)
        return prompts
    
    # Cross-validation: classify validation sets
    async def classify_fn(events, labels):
        """Classify events using fixed temperature (0.0 for CV)."""
        prompts = build_prompts_from_events(events)
        results = await classify_batch(
            prompts=prompts,
            client=client,
            model_name=MODEL_NAME,
            max_concurrent=MAX_CONCURRENT,
            max_retries=MAX_RETRIES,
            temperature=0.0,  # Deterministic for CV
            max_output_tokens=MAX_OUTPUT_TOKENS,
            progress_callback=progress_callback,
        )
        return results
    
    # Cross-validation
    start_time = datetime.now()
    fold_metrics, cv_summary = await cross_validate_llm(
        X_train=train_events,
        y_train=y_train,
        classify_fn=classify_fn,
        n_folds=n_folds,
        random_state=42,
        verbose=True,
    )
    cv_elapsed = (datetime.now() - start_time).total_seconds()
    
    # Test set prediction using best single temperature
    print("\n" + "="*80)
    print(f"Test Set Prediction (Temperature: {TEST_TEMPERATURE})")
    print("="*80)
    test_start = datetime.now()
    
    test_prompts = build_prompts_from_events(test_events)
    test_results = await classify_batch(
        prompts=test_prompts,
        client=client,
        model_name=MODEL_NAME,
        max_concurrent=MAX_CONCURRENT,
        max_retries=MAX_RETRIES,
        temperature=TEST_TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        progress_callback=progress_callback,
    )
    
    test_preds = [r["label"] for r in test_results]
    test_metrics = calculate_metrics(test_results, y_test) if y_test else None
    
    test_elapsed = (datetime.now() - test_start).total_seconds()
    total_elapsed = (datetime.now() - start_time).total_seconds()
    
    if test_metrics:
        print("\n" + "="*80)
        print("Test Set Results")
        print("="*80)
        print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"F1:        {test_metrics['f1']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall:    {test_metrics['recall']:.4f}")
        print("="*80)
    
    # Print final summary
    print("\n" + "="*80)
    print("Final Summary")
    print("="*80)
    print(f"Cross-Validation Accuracy: {cv_summary['accuracy']['mean']:.4f} ± {cv_summary['accuracy']['std']:.4f}")
    print(f"Cross-Validation F1: {cv_summary['f1']['mean']:.4f} ± {cv_summary['f1']['std']:.4f}")
    if test_metrics:
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"\nTime: CV={cv_elapsed:.1f}s, Test={test_elapsed:.1f}s, Total={total_elapsed:.1f}s")
    
    # Save results
    output_dir = os.path.join(rootpath, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"llm_{timestamp}.json")
    
    # Collect misclassified cases
    misclassified_cases = []
    if test_preds and y_test and test_results:
        # Find misclassified cases
        for i, (true_label, pred_label) in enumerate(zip(y_test, test_preds)):
            if true_label != pred_label:
                result = test_results[i]
                
                # Get root text
                root_text = ""
                try:
                    root_text = test_events[i]["root"].get("text", "") or ""
                except Exception:
                    pass
                
                misclassified_cases.append({
                    "index": i,
                    "true_label": int(true_label),
                    "predicted_label": int(pred_label),
                    "confidence": float(result["confidence"]),
                    "reason": result.get("reason", ""),
                    "root_text": root_text[:500],  # Limit text length
                })
    
    output_data = {
        "config": {
            "model": MODEL_NAME,
            "n_folds": n_folds,
            "train_size": len(train_events),
            "test_size": len(test_events),
            "max_concurrent": MAX_CONCURRENT,
            "k_front": K_FRONT,
            "m_back": M_BACK,
            "test_temperature": TEST_TEMPERATURE,
        },
        "cv_summary": {
            "accuracy": {
                "mean": float(cv_summary['accuracy']['mean']),
                "std": float(cv_summary['accuracy']['std']),
            },
            "f1": {
                "mean": float(cv_summary['f1']['mean']),
                "std": float(cv_summary['f1']['std']),
            },
            "precision": {
                "mean": float(cv_summary['precision']['mean']),
                "std": float(cv_summary['precision']['std']),
            },
            "recall": {
                "mean": float(cv_summary['recall']['mean']),
                "std": float(cv_summary['recall']['std']),
            },
        },
        "test_metrics": {
            "accuracy": float(test_metrics['accuracy']),
            "f1": float(test_metrics['f1']),
            "precision": float(test_metrics['precision']),
            "recall": float(test_metrics['recall']),
        } if test_metrics else None,
        "fold_metrics": [
            {
                "accuracy": float(m['accuracy']),
                "f1": float(m['f1']),
                "precision": float(m['precision']),
                "recall": float(m['recall']),
            }
            for m in fold_metrics
        ],
        "misclassified_cases": misclassified_cases,
        "misclassified_count": len(misclassified_cases),
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": {
            "cv": cv_elapsed,
            "test": test_elapsed,
            "total": total_elapsed,
        },
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    if misclassified_cases:
        print(f"Misclassified cases: {len(misclassified_cases)} (saved to results file)")
    
    if args.quick_test:
        print("\n" + "="*80)
        print("QUICK TEST COMPLETE - All checks passed!")
        print("You can now run without --quick-test for full training")
        print("="*80)


if __name__ == "__main__":
    asyncio.run(main())

