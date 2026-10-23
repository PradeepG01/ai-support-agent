"""LLM provider abstraction.

All providers take the same provider-neutral message list and return an `LLMResponse`, so the agent loop
does not care which model is behind it:

    {"role": "system" | "user" | "assistant" | "tool", "content": str,
     "tool_calls": [ToolCall, ...]  # assistant only
     "tool_call_id": str, "name": str}  # tool only

Providers:
  * MockLLM      - deterministic, offline; used for tests, CI and local demos without an API key.
  * OpenAIChat   - OpenAI Chat Completions API with function calling.
  * AnthropicChat- Anthropic Messages API with tool use.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from app.config import Settings
from app.tools import BOOKING_REF, ToolSpec


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


class LLMClient(Protocol):
    name: str

    def chat(self, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> LLMResponse: ...


class LLMError(RuntimeError):
    pass


def _post_with_retry(url: str, headers: dict, payload: dict, timeout: float, attempts: int = 3) -> dict:
    """POST with exponential backoff on 429/5xx and network errors."""
    delay = 1.0
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code in (429, 500, 502, 503, 529) and attempt < attempts:
                time.sleep(delay)
                delay *= 2
                continue
            response.raise_for_status()
            return response.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(delay)
            delay *= 2
    raise LLMError(f"LLM request failed after {attempts} attempts: {last_error}")


# --------------------------------------------------------------------------- OpenAI
class OpenAIChat:
    name = "openai"
    URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_key: str, model: str, timeout: float) -> None:
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        self.api_key, self.model, self.timeout = api_key, model or "gpt-4o-mini", timeout

    def _to_wire(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        wire = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                wire.append(
                    {
                        "role": "assistant",
                        "content": m.get("content") or None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                            }
                            for tc in m["tool_calls"]
                        ],
                    }
                )
            elif m["role"] == "tool":
                wire.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            else:
                wire.append({"role": m["role"], "content": m["content"]})
        return wire

    def chat(self, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> LLMResponse:
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": self._to_wire(messages),
            "tools": [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in tools
            ],
        }
        data = _post_with_retry(self.URL, {"Authorization": f"Bearer {self.api_key}"}, payload, self.timeout)
        message = data["choices"][0]["message"]
        calls = [
            ToolCall(tc["id"], tc["function"]["name"], json.loads(tc["function"]["arguments"] or "{}"))
            for tc in message.get("tool_calls") or []
        ]
        usage = data.get("usage", {})
        return LLMResponse(
            message.get("content"),
            calls,
            {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)},
        )


# ------------------------------------------------------------------------ Anthropic
class AnthropicChat:
    name = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, model: str, timeout: float) -> None:
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        if not model:
            raise LLMError("LLM_MODEL must be set to an Anthropic model id")
        self.api_key, self.model, self.timeout = api_key, model, timeout

    @staticmethod
    def _to_wire(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        wire: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "system":
                continue
            if m["role"] == "assistant":
                blocks: list[dict[str, Any]] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls") or []:
                    blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
                wire.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                result = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                # Consecutive tool results belong in a single user turn.
                if wire and wire[-1]["role"] == "user" and isinstance(wire[-1]["content"], list):
                    wire[-1]["content"].append(result)
                else:
                    wire.append({"role": "user", "content": [result]})
            else:
                wire.append({"role": "user", "content": m["content"]})
        return system, wire

    def chat(self, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> LLMResponse:
        system, wire = self._to_wire(messages)
        payload = {
            "model": self.model,
            "max_tokens": 512,
            "system": system,
            "messages": wire,
            "tools": [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools],
        }
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        data = _post_with_retry(self.URL, headers, payload, self.timeout)
        text = "".join(b["text"] for b in data["content"] if b["type"] == "text") or None
        calls = [ToolCall(b["id"], b["name"], b.get("input") or {}) for b in data["content"] if b["type"] == "tool_use"]
        return LLMResponse(text, calls, data.get("usage", {}))


# ----------------------------------------------------------------------------- Mock
class MockLLM:
    """Rule-based stand-in for an LLM. It follows the same protocol (tool calls, then a final answer)
    so the whole agent loop, API and eval suite run offline and deterministically."""

    name = "mock"
    HUMAN = re.compile(r"\b(human|real person|manager|speak to (someone|a person)|agent)\b", re.I)
    COMPLAINT = re.compile(
        r"\b(broken|not working|doesn'?t work|leak|dirty|noisy|refund|overcharged|complain|terrible|cold)\b", re.I
    )

    def chat(self, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> LLMResponse:
        last = messages[-1]
        if last["role"] == "tool":
            return LLMResponse(self._answer_from_tool(last["name"], json.loads(last["content"])))

        text = last["content"]
        tool_names = {t.name for t in tools}
        if self.HUMAN.search(text) and "escalate_to_human" in tool_names:
            return self._call("escalate_to_human", {"reason": "Guest asked for a human"})
        if (match := BOOKING_REF.search(text.upper())) and "lookup_booking" in tool_names:
            return self._call("lookup_booking", {"reference": match.group(1)})
        if self.COMPLAINT.search(text) and "create_ticket" in tool_names:
            category = "billing" if re.search(r"refund|overcharged|bill", text, re.I) else "maintenance"
            return self._call("create_ticket", {"category": category, "summary": text[:200], "priority": "high"})
        return LLMResponse(self._answer_from_context(messages[0]["content"], text))

    @staticmethod
    def _call(name: str, args: dict[str, Any]) -> LLMResponse:
        return LLMResponse(None, [ToolCall(f"call_{uuid.uuid4().hex[:8]}", name, args)])

    @staticmethod
    def _answer_from_tool(name: str, result: dict[str, Any]) -> str:
        if "error" in result:
            return "Sorry, I couldn't complete that. Could you double-check the details?"
        if name == "lookup_booking":
            if not result.get("found"):
                return f"I couldn't find a booking with reference {result['reference']}. Could you check it?"
            return (
                f"Your booking {result['reference']} is {result['status']}: a {result['room_type']} room "
                f"from {result['check_in']} to {result['check_out']}."
            )
        if name == "create_ticket":
            return (
                f"I'm sorry about that. I've raised ticket #{result['ticket_id']} and our team will follow up shortly."
            )
        if name == "escalate_to_human":
            return f"I'm connecting you with a member of our team. Expected wait: {result['eta_minutes']} minutes."
        return "Done."

    @staticmethod
    def _answer_from_context(system_prompt: str, question: str) -> str:
        context = system_prompt.split("<context>")[-1].split("</context>")[0]
        if "no relevant knowledge-base articles" in context:
            return "I'm not sure about that. Would you like me to connect you with our team?"

        def stems(text: str) -> set[str]:
            return {w.rstrip("s") for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 3}

        q_words = stems(question)
        asks_price = bool(re.search(r"\b(how much|cost|price|fee|charge)\b", question, re.I))
        sentences = re.split(r"(?<=[.!?])\s+|\n+", context)
        scored = []
        for s in sentences:
            clean = re.sub(r"^\[\d+\]\s*\([^)]*\)\s*", "", s).strip(" -*")
            if len(clean) < 15 or clean.startswith("#"):  # skip markdown headings
                continue
            overlap = len(q_words & stems(clean))
            if asks_price and re.search(r"INR|\bcosts?\b|\bcharged?\b|\bfee\b", clean):
                overlap += 1
            scored.append((overlap, clean))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        if not scored or scored[0][0] == 0:
            return "I'm not sure about that. Would you like me to connect you with our team?"
        return " ".join(s for _, s in scored[:2])


def get_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "openai":
        return OpenAIChat(settings.openai_api_key, settings.llm_model, settings.llm_timeout_seconds)
    if settings.llm_provider == "anthropic":
        return AnthropicChat(settings.anthropic_api_key, settings.llm_model, settings.llm_timeout_seconds)
    return MockLLM()
