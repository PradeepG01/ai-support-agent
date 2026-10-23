"""FastAPI service exposing the support agent over REST and a signed webhook."""

from __future__ import annotations

import logging
import time
from collections import Counter
from contextlib import asynccontextmanager
from datetime import date

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import __version__
from app.agent import SupportAgent
from app.config import get_settings
from app.db import SessionLocal, get_session, init_db
from app.extraction import ExtractionError, extract_booking_request
from app.llm import get_llm
from app.logging_config import configure_logging
from app.models import Booking, Conversation, WebhookEvent
from app.retrieval import KnowledgeBase
from app.webhooks import verify

log = logging.getLogger("api")
METRICS: Counter = Counter()


def seed_demo_bookings(session: Session) -> None:
    if session.scalar(select(Booking).limit(1)) is not None:
        return
    session.add_all(
        [
            Booking(
                reference="SG1234",
                guest_name="Asha Rao",
                room_type="Deluxe Sea View",
                check_in=date(2026, 11, 2),
                check_out=date(2026, 11, 5),
            ),
            Booking(
                reference="SG5678",
                guest_name="Rahul Mehta",
                room_type="Standard Twin",
                check_in=date(2026, 12, 20),
                check_out=date(2026, 12, 23),
                status="pending_payment",
            ),
        ]
    )
    session.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging()
    init_db()
    with SessionLocal() as session:
        seed_demo_bookings(session)
    knowledge = KnowledgeBase()
    chunks = knowledge.load_directory(settings.knowledge_dir)
    app.state.knowledge = knowledge
    app.state.agent = SupportAgent(get_llm(settings), knowledge, settings.agent_max_steps, settings.retrieval_top_k)
    log.info("startup complete", extra={"chunks": chunks, "llm": settings.llm_provider})
    yield


app = FastAPI(title="AI Support Agent", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    METRICS[f'http_requests_total{{path="{request.url.path}",status="{response.status_code}"}}'] += 1
    METRICS["http_request_duration_ms_sum"] += int((time.perf_counter() - started) * 1000)
    return response


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if x_api_key != get_settings().api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-API-Key")


# ----------------------------------------------------------------------------- schemas
class ChatRequest(BaseModel):
    customer_ref: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: int | None = None
    channel: str = "api"


class ChatResponse(BaseModel):
    conversation_id: int
    reply: str
    tool_calls: list[dict]
    sources: list[str]
    steps: int
    latency_ms: int
    escalated: bool


class KnowledgeDoc(BaseModel):
    doc_id: str = Field(pattern=r"^[a-z0-9\-_]{2,64}$")
    content: str = Field(min_length=20)


class InboundMessage(BaseModel):
    event_id: str = Field(min_length=1, max_length=128)
    sender: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=2000)
    channel: str = "whatsapp"


# ---------------------------------------------------------------------------- helpers
def _get_or_create_conversation(session: Session, customer_ref: str, channel: str, conversation_id: int | None):
    if conversation_id is not None:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None or conversation.customer_ref != customer_ref:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found")
        return conversation
    conversation = session.scalar(
        select(Conversation)
        .where(Conversation.customer_ref == customer_ref, Conversation.status == "open")
        .order_by(Conversation.id.desc())
    )
    if conversation is None:
        conversation = Conversation(customer_ref=customer_ref, channel=channel)
        session.add(conversation)
        session.flush()
    return conversation


def _run_agent(request: Request, session: Session, conversation: Conversation, text: str) -> ChatResponse:
    result = request.app.state.agent.run(session, conversation, text)
    METRICS["agent_replies_total"] += 1
    METRICS["agent_tool_calls_total"] += len(result.tool_calls)
    if result.escalated:
        METRICS["agent_escalations_total"] += 1
    return ChatResponse(**result.__dict__)


# ---------------------------------------------------------------------------- routes
@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "version": __version__, "knowledge_chunks": len(request.app.state.knowledge)}


@app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
def chat(body: ChatRequest, request: Request, session: Session = Depends(get_session)) -> ChatResponse:
    conversation = _get_or_create_conversation(session, body.customer_ref, body.channel, body.conversation_id)
    return _run_agent(request, session, conversation, body.message)


@app.get("/v1/conversations/{conversation_id}", dependencies=[Depends(require_api_key)])
def get_conversation(conversation_id: int, session: Session = Depends(get_session)) -> dict:
    conversation = session.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found")
    return {
        "id": conversation.id,
        "status": conversation.status,
        "channel": conversation.channel,
        "messages": [{"role": m.role, "content": m.content, "meta": m.meta} for m in conversation.messages],
    }


@app.post("/v1/knowledge", status_code=201, dependencies=[Depends(require_api_key)])
def ingest_document(doc: KnowledgeDoc, request: Request) -> dict:
    chunks = request.app.state.knowledge.add_document(doc.doc_id, doc.content)
    return {"doc_id": doc.doc_id, "chunks": chunks}


@app.get("/v1/knowledge", dependencies=[Depends(require_api_key)])
def list_documents(request: Request) -> dict:
    return {"documents": request.app.state.knowledge.documents}


@app.post("/v1/webhooks/messages")
async def inbound_webhook(
    request: Request,
    x_signature: str | None = Header(default=None),
    x_timestamp: str | None = Header(default=None),
    session: Session = Depends(get_session),
) -> dict:
    raw = await request.body()
    if not verify(get_settings().webhook_secret, x_timestamp, x_signature, raw):
        METRICS["webhook_rejected_total"] += 1
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid signature")
    event = InboundMessage.model_validate_json(raw)

    duplicate = session.get(WebhookEvent, event.event_id)
    if duplicate is not None:  # provider retried a delivery we already handled
        return {"event_id": event.event_id, "duplicate": True, "reply": duplicate.reply}

    conversation = _get_or_create_conversation(session, event.sender, event.channel, None)
    response = _run_agent(request, session, conversation, event.text)
    session.add(WebhookEvent(event_id=event.event_id, conversation_id=conversation.id, reply=response.reply))
    session.commit()
    return {"event_id": event.event_id, "duplicate": False, "reply": response.reply}


class ExtractRequest(BaseModel):
    email: str = Field(min_length=10, max_length=5000)


@app.post("/v1/extract/booking-request", dependencies=[Depends(require_api_key)])
def extract_booking(body: ExtractRequest, request: Request) -> dict:
    try:
        booking = extract_booking_request(request.app.state.agent.llm, body.email)
    except ExtractionError as exc:
        METRICS["extraction_failures_total"] += 1
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"could not extract a valid booking: {exc}") from exc
    return booking.model_dump(mode="json")


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return "\n".join(f"{name} {value}" for name, value in sorted(METRICS.items())) + "\n"
