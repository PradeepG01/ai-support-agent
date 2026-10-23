"""The agent loop: retrieve context -> call LLM -> execute tool calls -> repeat until a final answer.

Guardrails:
  * a hard cap on reasoning steps (prevents runaway tool loops and cost blow-ups),
  * unknown tools / bad arguments are returned to the model as errors instead of crashing,
  * if the step budget is exhausted the conversation is escalated to a human.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.llm import LLMClient
from app.models import Conversation, Message
from app.prompts import PROMPT_VERSION, build_system_prompt
from app.retrieval import KnowledgeBase
from app.tools import TOOLS, execute_tool

log = logging.getLogger("agent")

FALLBACK_REPLY = "I'm having trouble with that request, so I've passed it to our team. Someone will reply shortly."


@dataclass
class AgentResult:
    reply: str
    conversation_id: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    steps: int = 0
    latency_ms: int = 0
    escalated: bool = False


class SupportAgent:
    def __init__(self, llm: LLMClient, knowledge: KnowledgeBase, max_steps: int = 4, top_k: int = 3) -> None:
        self.llm, self.knowledge, self.max_steps, self.top_k = llm, knowledge, max_steps, top_k

    def _history(self, conversation: Conversation, limit: int = 10) -> list[dict[str, Any]]:
        turns = [m for m in conversation.messages if m.role in ("user", "assistant")][-limit:]
        return [{"role": m.role, "content": m.content} for m in turns]

    def run(self, session: Session, conversation: Conversation, user_text: str) -> AgentResult:
        started = time.perf_counter()
        results = self.knowledge.search(user_text, top_k=self.top_k)
        messages: list[dict[str, Any]] = [{"role": "system", "content": build_system_prompt(results)}]
        messages += self._history(conversation)
        messages.append({"role": "user", "content": user_text})
        session.add(Message(conversation_id=conversation.id, role="user", content=user_text))

        result = AgentResult(reply=FALLBACK_REPLY, conversation_id=conversation.id)
        result.sources = sorted({r.chunk.doc_id for r in results})
        tools = list(TOOLS.values())

        for step in range(1, self.max_steps + 1):
            result.steps = step
            response = self.llm.chat(messages, tools)
            if not response.tool_calls:
                result.reply = (response.content or "").strip() or FALLBACK_REPLY
                break
            messages.append({"role": "assistant", "content": response.content, "tool_calls": response.tool_calls})
            for call in response.tool_calls:
                output = execute_tool(session, conversation.id, call.name, call.arguments)
                result.tool_calls.append({"name": call.name, "arguments": call.arguments, "result": output})
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": json.dumps(output)}
                )
                session.add(
                    Message(
                        conversation_id=conversation.id,
                        role="tool",
                        content=json.dumps(output),
                        meta={"tool": call.name, "arguments": call.arguments},
                    )
                )
        else:
            execute_tool(session, conversation.id, "escalate_to_human", {"reason": "agent step budget exhausted"})
            log.warning("step budget exhausted", extra={"conversation_id": conversation.id})

        result.escalated = conversation.status == "escalated"
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        session.add(
            Message(
                conversation_id=conversation.id,
                role="assistant",
                content=result.reply,
                meta={
                    "prompt_version": PROMPT_VERSION,
                    "model": self.llm.name,
                    "sources": result.sources,
                    "steps": result.steps,
                    "latency_ms": result.latency_ms,
                },
            )
        )
        session.commit()
        log.info(
            "agent reply",
            extra={
                "conversation_id": conversation.id,
                "steps": result.steps,
                "tools": [t["name"] for t in result.tool_calls],
                "latency_ms": result.latency_ms,
            },
        )
        return result
