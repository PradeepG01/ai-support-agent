"""LLM document extraction: turn an unstructured guest email into a validated booking request.

Pattern: prompt with an explicit JSON schema -> parse -> validate with pydantic -> if invalid, send the
validation errors back to the model once so it can correct itself. Output is never trusted unvalidated.
"""

from __future__ import annotations

import json
import re
from datetime import date

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.llm import LLMClient

ROOM_TYPES = ["Standard Twin", "Deluxe Sea View", "Suite"]


class BookingRequest(BaseModel):
    guest_name: str = Field(min_length=2)
    check_in: date
    check_out: date
    guests: int = Field(ge=1, le=8)
    room_type: str | None = None
    special_requests: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_dates(self) -> BookingRequest:
        if self.check_out <= self.check_in:
            raise ValueError("check_out must be after check_in")
        if self.room_type is not None and self.room_type not in ROOM_TYPES:
            raise ValueError(f"room_type must be one of {ROOM_TYPES} or null")
        return self


EXTRACTION_PROMPT = """Extract a hotel booking request from the email below.
Return ONLY a JSON object with exactly these keys:
  guest_name (string), check_in (YYYY-MM-DD), check_out (YYYY-MM-DD), guests (integer),
  room_type (one of {room_types} or null), special_requests (array of short strings).
Assume the year is {year} when it is not stated. Do not invent values that are not in the email.

<email>
{email}
</email>"""


class ExtractionError(Exception):
    pass


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ExtractionError("model did not return JSON")
    return json.loads(text[start : end + 1])


def _rule_based(email: str, year: int) -> dict:
    """Offline extractor used with the mock LLM (tests/CI). Handles the common '12 Nov' style dates."""
    months = "jan feb mar apr may jun jul aug sep oct nov dec".split()
    dates = []
    for day, mon in re.findall(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3})[a-z]*\b", email):
        if mon.lower() in months:
            dates.append(date(year, months.index(mon.lower()) + 1, int(day)).isoformat())
    name = re.search(r"(?:regards|thanks|sincerely)[,\s]+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", email, re.I)
    guests = re.search(r"\b(\d)\s+(?:adults|guests|people|persons)\b", email, re.I)
    room = next((r for r in ROOM_TYPES if r.split()[0].lower() in email.lower()), None)
    requests = [r.strip() for r in re.findall(r"(?:please|also)\s+(?:arrange|provide|need)\s+([^.]+)", email, re.I)]
    return {
        "guest_name": name.group(1) if name else "",
        "check_in": dates[0] if dates else None,
        "check_out": dates[1] if len(dates) > 1 else None,
        "guests": int(guests.group(1)) if guests else 1,
        "room_type": room,
        "special_requests": requests,
    }


def extract_booking_request(llm: LLMClient, email: str, year: int | None = None) -> BookingRequest:
    year = year or date.today().year
    if llm.name == "mock":
        try:
            return BookingRequest.model_validate(_rule_based(email, year))
        except ValidationError as exc:
            raise ExtractionError(str(exc)) from exc

    messages = [
        {"role": "system", "content": "You are a precise information-extraction engine."},
        {"role": "user", "content": EXTRACTION_PROMPT.format(room_types=ROOM_TYPES, year=year, email=email)},
    ]
    for attempt in range(2):  # one self-correction round
        reply = llm.chat(messages, tools=[]).content or ""
        try:
            return BookingRequest.model_validate(_parse_json(reply))
        except (ValidationError, ValueError, ExtractionError) as exc:
            if attempt == 1:
                raise ExtractionError(f"invalid extraction after retry: {exc}") from exc
            messages += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": f"That output was invalid: {exc}. Return corrected JSON only."},
            ]
    raise ExtractionError("unreachable")
