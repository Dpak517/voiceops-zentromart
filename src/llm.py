"""Groq wrapper: client, retries, JSON parsing and a small tool-calling loop."""
import json
import os
import re
import time

from dotenv import load_dotenv
from groq import APIConnectionError, BadRequestError, Groq, RateLimitError

load_dotenv()

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
REWRITE_MODEL = os.getenv("GROQ_REWRITE_MODEL") or DEFAULT_MODEL
_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")
        _client = Groq(api_key=key)
    return _client


def _kwargs(messages, model, temperature, max_tokens, reasoning_effort=None, **extra):
    kw = dict(model=model or DEFAULT_MODEL, messages=messages,
              temperature=temperature, max_tokens=max_tokens)
    if reasoning_effort and kw["model"].startswith("openai/gpt-oss"):
        kw["extra_body"] = {"reasoning_effort": reasoning_effort}   # works on any SDK version
    kw.update(extra)
    return kw


def _create(kwargs, retries=6):
    """Call Groq with exponential backoff on rate limits / network errors."""
    delay = 2.0
    for attempt in range(retries):
        try:
            return _get_client().chat.completions.create(**kwargs)
        except (RateLimitError, APIConnectionError):
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30)


def chat(messages, *, model=None, temperature=0.0, json_mode=False,
         max_tokens=2000, reasoning_effort=None) -> str:
    """Plain chat call. Returns the assistant text."""
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = _create(_kwargs(messages, model, temperature, max_tokens, reasoning_effort, **extra))
    return resp.choices[0].message.content or ""


def chat_with_tools(messages, tools, impls, *, model=None, temperature=0.0,
                    max_tokens=4500, max_rounds=5, reasoning_effort=None):
    """Chat with function calling. Runs the tools the model asks for until it answers.

    tools: OpenAI-style tool specs.  impls: {tool_name: python_function(**kwargs) -> dict}.
    Returns (final_text, trace) where trace lists every tool call made (for logging).
    """
    msgs = list(messages)
    trace = []
    for _ in range(max_rounds):
        try:
            resp = _create(_kwargs(msgs, model, temperature, max_tokens, reasoning_effort,
                                   tools=tools, tool_choice="auto"))
        except BadRequestError:
            break  # model produced a malformed tool call -> answer without tools below
        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            if (msg.content or "").strip():
                return msg.content, trace
            break  # empty answer (e.g. thinking used all tokens) -> retry below with a lighter setting
        msgs.append({"role": "assistant", "content": msg.content or "",
                     "tool_calls": [{"id": c.id, "type": "function",
                                     "function": {"name": c.function.name,
                                                  "arguments": c.function.arguments}} for c in calls]})
        for c in calls:
            try:
                args = json.loads(c.function.arguments or "{}")
                out = impls[c.function.name](**args)
            except Exception as exc:  # report the error to the model instead of crashing
                args, out = {}, {"error": str(exc)}
            trace.append({"tool": c.function.name, "args": args, "result": out})
            msgs.append({"role": "tool", "tool_call_id": c.id,
                         "content": json.dumps(out, ensure_ascii=False)})
    # tool budget used up, tool call failed, or empty answer: force a final answer with no tools,
    # low reasoning effort and a bigger token budget so the answer cannot be cut off
    resp = _create(_kwargs(msgs, model, temperature, max(max_tokens, 4000), "low"))
    return resp.choices[0].message.content or "", trace


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences / extra text."""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text or "", flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}
