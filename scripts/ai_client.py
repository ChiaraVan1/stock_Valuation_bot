"""
AI客户端工厂 —— Claude优先，失败时自动fallback到DeepSeek/Qwen
用法：
    from ai_client import get_text_client, claude_completion, text_completion
"""

import os
from openai import OpenAI

# ── 环境变量 ────────────────────────────────────────────────
CLAUDE_API_KEY  = os.environ.get("CLAUDE_API_KEY", "")
CLAUDE_BASE_URL = "https://openai.qiniu.com/v1"
CLAUDE_MODEL    = "claude-4.5-sonnet"

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
QWEN_API_KEY     = os.environ.get("QWEN_API_KEY", "")


# ── 客户端构造 ──────────────────────────────────────────────

def _claude_client() -> OpenAI:
    if not CLAUDE_API_KEY:
        raise ValueError("CLAUDE_API_KEY 未设置")
    return OpenAI(api_key=CLAUDE_API_KEY, base_url=CLAUDE_BASE_URL)


def _deepseek_client() -> OpenAI:
    """DeepSeek官方或阿里云额度，自动选优"""
    # 先试阿里云DeepSeek额度
    if QWEN_API_KEY:
        try:
            client = OpenAI(
                api_key=QWEN_API_KEY,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            client.chat.completions.create(
                model="deepseek-v3.2", max_tokens=1,
                messages=[{"role": "user", "content": "hi"}]
            )
            print("  [ai_client] 使用阿里云DeepSeek额度")
            return client
        except Exception:
            pass
    # fallback到DeepSeek官方
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY 和 QWEN_API_KEY 均未设置")
    print("  [ai_client] 使用DeepSeek官方")
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


def _qwen_vl_client() -> OpenAI:
    if not QWEN_API_KEY:
        raise ValueError("QWEN_API_KEY 未设置")
    return OpenAI(
        api_key=QWEN_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


# ── 核心调用函数 ────────────────────────────────────────────

def claude_completion(
    messages: list,
    system: str = "",
    max_tokens: int = 1000,
    temperature: float = 0,
) -> str:
    """
    直接调用Claude，不fallback。
    调用方可自行决定是否try/except后走fallback。
    """
    client = _claude_client()
    kwargs = dict(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
    )
    # Claude通过system角色消息传system prompt
    if system:
        kwargs["messages"] = [{"role": "system", "content": system}] + messages
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content.strip()


def deepseek_completion(
    messages: list,
    system: str = "",
    max_tokens: int = 1000,
    temperature: float = 0,
    model: str = "deepseek-v3.2",
) -> str:
    """直接调用DeepSeek，不fallback。"""
    client = _deepseek_client()
    all_messages = []
    if system:
        all_messages.append({"role": "system", "content": system})
    all_messages.extend(messages)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=all_messages,
    )
    return resp.choices[0].message.content.strip()


def text_completion(
    messages: list,
    system: str = "",
    max_tokens: int = 1000,
    temperature: float = 0,
    task_label: str = "",        # 用于日志，如 "OCR校验" / "估值倍数提取"
    deepseek_model: str = "deepseek-v3.2",
    claude_max_tokens: int = 0,  # 若为0则复用max_tokens；大任务可单独指定
) -> tuple[str, str]:
    """
    Claude优先，失败时自动fallback到DeepSeek。
    返回 (content, provider)，provider为 "claude" 或 "deepseek"。
    """
    label = f"[{task_label}] " if task_label else ""

    # ── 尝试Claude ──
    if CLAUDE_API_KEY:
        try:
            print(f"  {label}→ 尝试Claude...")
            content = claude_completion(
                messages=messages,
                system=system,
                max_tokens=claude_max_tokens or max_tokens,
                temperature=temperature,
            )
            print(f"  {label}✅ Claude成功（{len(content)}字）")
            return content, "claude"
        except Exception as e:
            print(f"  {label}⚠️  Claude失败（{e}），fallback到DeepSeek...")
    else:
        print(f"  {label}CLAUDE_API_KEY未设置，直接使用DeepSeek")

    # ── Fallback到DeepSeek ──
    content = deepseek_completion(
        messages=messages,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        model=deepseek_model,
    )
    print(f"  {label}✅ DeepSeek成功（{len(content)}字）")
    return content, "deepseek"


def get_qwen_vl_client() -> OpenAI:
    """图片OCR专用，始终用千问VL（Claude暂不支持本地base64图片走该接口）"""
    return _qwen_vl_client()
