"""
Rumor detection prompt for LLM models.

This module provides prompt templates and functions to build complete prompts
for rumor detection tasks using LLM APIs (Gemini, GPT, etc.).
"""
from typing import Optional


# System prompt (with clear criteria)
SYSTEM_PROMPT = """你是一个专业的谣言检测专家，只能输出 JSON。

判定标准：
- label=1（谣言）：缺乏可信来源/权威报道；已被辟谣；夸大、捏造、阴谋论；评论中有明确辟谣或质疑且无可靠佐证；涉及公共安全/医疗/食品/药品/政策/商业指控/求助但无法核实时，倾向判为谣言。
- label=0（非谣言）：有权威或可信来源支撑；为纯个人情绪/日常表达且无事实主张；内容客观可验证且未被反驳。
- 边界：若无把握且缺权威来源，请偏向 label=1，但必须给出置信度。

输出要求（只允许一行 JSON，不要解释、不用 Markdown/反引号）：
{"label": 0或1, "confidence": 0.0-1.0, "reason": "不超过30字的中文理由"}"""


# Few-shot examples（短小，覆盖正负与边界）
FEW_SHOT_EXAMPLES = [
    {
        "input": "源博文：中央气象台发布台风登陆预警，附官方链接。\n评论：1. 已在央视看到同样消息\n2. 早点准备物资\n3. 这是官方发布的",
        "output": {"label": 0, "confidence": 0.92, "reason": "有官方来源，评论支持，可信度高"},
    },
    {
        "input": "源博文：听说XX医院倒闭了，大家别去看病了！\n评论：1. 医生辟谣：医院正常接诊\n2. 官方微博未发布相关消息\n3. 别信谣言",
        "output": {"label": 1, "confidence": 0.94, "reason": "无权威来源且被辟谣，属于谣言"},
    },
    {
        "input": "源博文：月球其实是空心的，是外星人基地。\n评论：1. 又来阴谋论\n2. 没证据别乱说\n3. 纯属想象吧",
        "output": {"label": 1, "confidence": 0.88, "reason": "典型阴谋论，无证据支持"},
    },
]


def _format_few_shot_examples(examples):
    """Render few-shot examples as compact text."""
    formatted = []
    for i, ex in enumerate(examples, 1):
        formatted.append(
            f"示例{i}：\n输入：{ex['input']}\n输出：{ex['output']}"
        )
    return "\n\n".join(formatted)


def build_prompt(
    content_text: str,
    system_prompt: Optional[str] = None,
    use_few_shot: bool = True,
    few_shot_examples=None,
) -> str:
    """
    Build complete prompt, including system prompt and content to be classified.
    
    Args:
        content_text: Content text to be classified (built by llm_dataset, including author information, source blog, and comments)
        system_prompt: System prompt, if None then use default SYSTEM_PROMPT
    
    Returns:
        Complete prompt string
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT
    if few_shot_examples is None:
        few_shot_examples = FEW_SHOT_EXAMPLES
    
    examples_block = _format_few_shot_examples(few_shot_examples) if use_few_shot else ""
    examples_section = f"\n\n以下是示例（参考格式和判定标准）：\n{examples_block}\n" if use_few_shot else ""
    
    # Build complete prompt (content as user message)
    full_prompt = (
        f"{system_prompt}"
        f"{examples_section}"
        "\n待分类内容：\n"
        f"{content_text}\n\n"
        "请只输出一行 JSON，不要输出解释、分析、自然语言、Markdown、代码块或反引号。\n"
        'JSON 模板：{"label": 0或1, "confidence": 0.0-1.0, "reason": "不超过30字的中文理由"}\n'
        "请输出上述 JSON："
    )
    
    return full_prompt




