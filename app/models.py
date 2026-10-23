"""ORM models: conversations, messages, bookings, support tickets and processed webhook events."""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import JSON, Date, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    channel: Mapped[str] = mapped_column(String(32), default="api")  # api | whatsapp | sms | web
    customer_ref: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), default="open")  # open | escalated | closed
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.id"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))  # user | assistant | tool
    content: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class Booking(Base):
    """Reservation record, standing in for a property-management / booking system integration."""

    __tablename__ = "bookings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    reference: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    guest_name: Mapped[str] = mapped_column(String(128))
    room_type: Mapped[str] = mapped_column(String(32))
    check_in: Mapped[date] = mapped_column(Date)
    check_out: Mapped[date] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(16), default="confirmed")


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int | None] = mapped_column(ForeignKey("conversations.id"), nullable=True)
    category: Mapped[str] = mapped_column(String(32))
    summary: Mapped[str] = mapped_column(Text)
    priority: Mapped[str] = mapped_column(String(8), default="normal")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class WebhookEvent(Base):
    """Stores processed webhook event ids so retried deliveries are handled idempotently."""

    __tablename__ = "webhook_events"

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(Integer)
    reply: Mapped[str] = mapped_column(Text)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
