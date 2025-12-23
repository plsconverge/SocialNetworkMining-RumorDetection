"""
LLM-based rumor classifier using Gemini API with async IO.

This module provides async functions to classify rumors using Gemini API.
"""
import os
import sys
import json
import re
import asyncio
from typing import List, Dict, Any, Optional

try:
    from google import genai
except ImportError:
    print("Error: google-genai is required.")
    print("Install it with: uv pip install google-genai")
    sys.exit(1)


def parse_llm_response(text: str) -> Dict[str, Any]:
    """
    Parse LLM response to extract JSON result.
    Tries to be lenient: strips markdown fences and partial JSON.
    """
    # Clean markdown fences if any
    clean_text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    clean_text = re.sub(r"```", "", clean_text)
    clean_text = clean_text.strip()

    # Pattern 0: try the broadest { ... } slice to survive pre/suffix chatter
    try:
        start = clean_text.find("{")
        end = clean_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = clean_text[start : end + 1]
            result = json.loads(json_str)
            return {
                "label": int(result.get("label", 0)),
                "confidence": float(result.get("confidence", 0.5)),
                "reason": result.get("reason", "成功解析"),
                "parse_error": None,
            }
    except Exception:
        pass

    # Pattern 1: Look for JSON object
    json_match = re.search(r'\{[^{}]*"label"[^{}]*\}', clean_text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            return {
                "label": int(result.get("label", 0)),
                "confidence": float(result.get("confidence", 0.5)),
                "reason": result.get("reason", ""),
                "parse_error": None,
            }
        except (json.JSONDecodeError, ValueError):
            pass

    # Pattern 2: broader JSON
    json_match = re.search(r'\{.*?"label".*?\}', clean_text, re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group()
            json_str = re.sub(r'```json\s*', '', json_str)
            json_str = re.sub(r'```\s*', '', json_str)
            result = json.loads(json_str)
            return {
                "label": int(result.get("label", 0)),
                "confidence": float(result.get("confidence", 0.5)),
                "reason": result.get("reason", ""),
                "parse_error": None,
            }
        except (json.JSONDecodeError, ValueError):
            pass

    # Pattern 3: regex fields
    label_match = re.search(r'"label"\s*:\s*(\d+)', clean_text)
    confidence_match = re.search(r'"confidence"\s*:\s*([\d.]+)', clean_text)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', clean_text)

    if label_match:
        label = int(label_match.group(1))
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        reason = reason_match.group(1) if reason_match else "解析失败，使用默认值"
        return {
            "label": label,
            "confidence": confidence,
            "reason": reason,
            "parse_error": "Partial parse (regex)",
        }

    # Pattern 4: minimal label
    label_match = re.search(r'\b(label|标签)[\s:：]*(\d)\b', clean_text, re.IGNORECASE)
    if label_match:
        label = int(label_match.group(2))
        return {
            "label": label,
            "confidence": 0.0,
            "reason": "解析失败，仅提取到label",
            "parse_error": "Minimal parse",
        }

    return {
        "label": 0,
        "confidence": 0.0,
        "reason": "无法解析输出",
        "parse_error": f"Failed to parse: {clean_text[:100]}",
    }


async def classify_single(
    prompt: str,
    client: Optional["genai.Client"] = None,
    model_name: str = "models/gemini-1.5",
    semaphore: Optional[asyncio.Semaphore] = None,
    max_retries: int = 4,
    temperature: float = 0.0,
    max_output_tokens: int = 256,
    system_prompt: str = (
        "你是谣言检测专家，只能输出一行 JSON。"
        "标准：label=1 为缺权威来源/被辟谣/夸大捏造/阴谋论；"
        "label=0 为有可信来源或仅个人表达。无把握时仍需给出标签与置信度。"
        'JSON 模板：{"label":0或1,"confidence":0.0-1.0,"reason":"<=30字中文理由"}。'
        "不要输出解释、前后缀、Markdown 或反引号。"
    ),
    rumor_bias: bool = False,
    rumor_bias_threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    Classify a single event using Gemini API (async).
    
    Args:
        prompt: Complete prompt string
        model_name: Gemini model name (default: gemini-2.0-flash-exp)
        semaphore: Semaphore to control concurrency
        max_retries: Maximum number of retry attempts
        temperature: Generation temperature (0.0-1.0)
        max_output_tokens: Maximum output tokens
        system_prompt: System instruction for the model
        rumor_bias: If True, low-confidence label 0 will be nudged to 1
        rumor_bias_threshold: threshold for confidence to trigger rumor bias
    
    Returns:
        Dictionary with classification result and metadata
    """
    if semaphore:
        await semaphore.acquire()
    
    if client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {
                "label": 0,
                "confidence": 0.0,
                "reason": "API错误: GEMINI_API_KEY 未设置",
                "parse_error": None,
                "raw_response": "",
                "api_error": "no_api_key",
                "retries": 0,
            }
        client = genai.Client(api_key=api_key)

    try:
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model_name,
                    contents=prompt,  # 直接传递字符串
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        response_mime_type="application/json",
                    ),
                )
                # Extract raw text safely
                raw_text = ""
                try:
                    raw_text = response.text
                except Exception:
                    if hasattr(response, "candidates") and response.candidates:
                        for cand in response.candidates:
                            if hasattr(cand, "content") and cand.content and hasattr(cand.content, "parts"):
                                texts = []
                                for p in cand.content.parts:
                                    if hasattr(p, "text") and p.text:
                                        texts.append(p.text)
                                if texts:
                                    raw_text = "\n".join(texts)
                                    break

                result = parse_llm_response(raw_text)
                result["raw_response"] = raw_text
                result["api_error"] = None
                result["api_error_type"] = None
                # Optional post-processing: bias toward rumor for low-confidence 0
                if rumor_bias and result["label"] == 0 and result["confidence"] < rumor_bias_threshold:
                    result["label"] = 1
                    result["reason"] = "低置信度，偏向标为谣言；" + (result.get("reason") or "")
                # Add debug meta if available
                finish_reason = None
                safety = None
                try:
                    if response.candidates:
                        finish_reason = response.candidates[0].finish_reason
                        safety = response.candidates[0].safety_ratings
                except Exception:
                    pass
                result["finish_reason"] = finish_reason
                result["safety_ratings"] = safety
                result["retries"] = attempt
                
                return result
                
            except Exception as e:
                error_msg = str(e)
                error_type = e.__class__.__name__
                # Check if it's a rate limit error
                if "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    # Rate limit: wait longer
                    wait_time = min(60, (2 ** attempt) * 6)  # slower backoff for quota
                    if attempt < max_retries - 1:
                        await asyncio.sleep(wait_time)
                        continue
                
                # Other errors: exponential backoff
                if attempt < max_retries - 1:
                    wait_time = min(20, 2 + (2 ** attempt))
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Final attempt failed
                    return {
                        "label": 0,
                        "confidence": 0.0,
                        "reason": f"API错误: {error_msg}",
                        "parse_error": None,
                        "raw_response": "",
                        "api_error": error_msg,
                        "api_error_type": error_type,
                        "retries": attempt,
                    }
    
    finally:
        if semaphore:
            semaphore.release()


async def classify_batch(
    prompts: List[str],
    client: Optional["genai.Client"] = None,
    model_name: str = "models/gemini-2.5-flash-lite",
    max_concurrent: int = 5,
    max_retries: int = 4,
    temperature: float = 0.0,
    max_output_tokens: int = 256,
    system_prompt: str = (
        "你是一个专业的谣言检测专家。"
        "请根据输入文本判断是否为谣言，并仅用 JSON 返回结果："
        '{"label":0或1,"confidence":0到1之间的小数,"reason":"简要理由"}。'
        "不要输出除 JSON 以外的任何内容。"
    ),
    progress_callback: Optional[callable] = None,
) -> List[Dict[str, Any]]:
    """
    Classify a batch of events using async IO.
    
    Args:
        prompts: List of prompt strings
        model_name: Gemini model name
        max_concurrent: Maximum concurrent requests
        max_retries: Maximum retry attempts per request
        temperature: Generation temperature
        max_output_tokens: Maximum output tokens
        system_prompt: System instruction for the model
        progress_callback: Optional callback function(current, total)
    
    Returns:
        List of classification results
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    total = len(prompts)
    
    async def classify_with_progress(idx: int, prompt: str):
        result = await classify_single(
            prompt=prompt,
            client=client,
            model_name=model_name,
            semaphore=semaphore,
            max_retries=max_retries,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_prompt=system_prompt,
        )
        if progress_callback:
            progress_callback(idx + 1, total)
        return result
    
    # Create tasks
    tasks = [
        classify_with_progress(i, prompt)
        for i, prompt in enumerate(prompts)
    ]
    
    # Execute all tasks
    results = await asyncio.gather(*tasks)
    
    return results


def calculate_metrics(
    results: List[Dict[str, Any]],
    true_labels: List[int],
) -> Dict[str, Any]:
    """
    Calculate classification metrics.
    
    Args:
        results: List of classification results from classify_batch
        true_labels: List of true labels (0 or 1)
    
    Returns:
        Dictionary with metrics and statistics
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        classification_report,
        confusion_matrix,
    )
    
    # Extract predictions
    pred_labels = [r["label"] for r in results]
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average="macro")
    precision = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
    recall = recall_score(true_labels, pred_labels, average="macro", zero_division=0)
    
    # Classification report - same parameters as BERT baseline
    report = classification_report(
        true_labels,
        pred_labels,
        labels=[0, 1],
        target_names=["Non-Rumor", "Rumor"],
        zero_division=0,
    )
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
    
    # Statistics
    parse_errors = sum(1 for r in results if r.get("parse_error"))
    api_errors = sum(1 for r in results if r.get("api_error"))
    avg_confidence = sum(r["confidence"] for r in results) / len(results) if results else 0.0
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "report": report,
        "confusion_matrix": cm,
        "statistics": {
            "total": len(results),
            "parse_errors": parse_errors,
            "api_errors": api_errors,
            "avg_confidence": avg_confidence,
            "parse_error_rate": parse_errors / len(results) if results else 0.0,
            "api_error_rate": api_errors / len(results) if results else 0.0,
        },
    }

