"""Tools the agent can call. Each tool has a JSON schema (sent to the LLM) and a Python handler.

In a real deployment these handlers are the "integration glue" to customer systems: a PMS for bookings,
a helpdesk/CRM for tickets, and a live-agent queue for escalations.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Booking, Conversation, Ticket

BOOKING_REF = re.compile(r"\b([A-Z]{2}\d{4})\b")


class ToolError(Exception):
    """Raised for invalid tool arguments; the message is returned to the LLM so it can recover."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[Session, int, dict[str, Any]], dict[str, Any]]


def _lookup_booking(session: Session, conversation_id: int, args: dict[str, Any]) -> dict[str, Any]:
    reference = str(args.get("reference", "")).strip().upper()
    if not BOOKING_REF.fullmatch(reference):
        raise ToolError("reference must be two letters followed by four digits, e.g. SG1234")
    booking = session.scalar(select(Booking).where(Booking.reference == reference))
    if booking is None:
        return {"found": False, "reference": reference}
    return {
        "found": True,
        "reference": booking.reference,
        "guest_name": booking.guest_name,
        "room_type": booking.room_type,
        "check_in": booking.check_in.isoformat(),
        "check_out": booking.check_out.isoformat(),
        "status": booking.status,
    }


def _create_ticket(session: Session, conversation_id: int, args: dict[str, Any]) -> dict[str, Any]:
    category = str(args.get("category", "")).strip().lower()
    summary = str(args.get("summary", "")).strip()
    priority = str(args.get("priority", "normal")).strip().lower()
    if category not in {"maintenance", "housekeeping", "billing", "complaint", "other"}:
        raise ToolError("category must be one of maintenance, housekeeping, billing, complaint, other")
    if priority not in {"low", "normal", "high"}:
        raise ToolError("priority must be low, normal or high")
    if len(summary) < 5:
        raise ToolError("summary is required")
    ticket = Ticket(conversation_id=conversation_id, category=category, summary=summary[:500], priority=priority)
    session.add(ticket)
    session.flush()
    return {"ticket_id": ticket.id, "category": category, "priority": priority}


def _escalate_to_human(session: Session, conversation_id: int, args: dict[str, Any]) -> dict[str, Any]:
    conversation = session.get(Conversation, conversation_id)
    if conversation is not None:
        conversation.status = "escalated"
    return {"escalated": True, "reason": str(args.get("reason", "guest request"))[:200], "eta_minutes": 5}


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in [
        ToolSpec(
            name="lookup_booking",
            description="Fetch a guest's reservation from the property management system by booking reference.",
            parameters={
                "type": "object",
                "properties": {"reference": {"type": "string", "description": "Booking reference, e.g. SG1234"}},
                "required": ["reference"],
            },
            handler=_lookup_booking,
        ),
        ToolSpec(
            name="create_ticket",
            description="Open a support ticket for complaints, maintenance issues or billing/refund requests.",
            parameters={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["maintenance", "housekeeping", "billing", "complaint", "other"],
                    },
                    "summary": {"type": "string", "description": "One-sentence description of the issue"},
                    "priority": {"type": "string", "enum": ["low", "normal", "high"]},
                },
                "required": ["category", "summary"],
            },
            handler=_create_ticket,
        ),
        ToolSpec(
            name="escalate_to_human",
            description="Hand the conversation to a human staff member.",
            parameters={
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
            handler=_escalate_to_human,
        ),
    ]
}


def execute_tool(session: Session, conversation_id: int, name: str, args: dict[str, Any]) -> dict[str, Any]:
    spec = TOOLS.get(name)
    if spec is None:
        return {"error": f"unknown tool '{name}'"}
    try:
        return spec.handler(session, conversation_id, args)
    except ToolError as exc:
        return {"error": str(exc)}
