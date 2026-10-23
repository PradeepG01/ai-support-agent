from datetime import date

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db import init_db
from app.extraction import ExtractionError, extract_booking_request
from app.llm import LLMResponse, MockLLM
from app.models import Booking
from scripts.import_bookings import import_bookings, read_rows

EMAIL = """Hi team,
We'd like a Deluxe room for 2 adults from 12 Nov to 15 Nov. Please arrange an airport pickup.
Thanks, Meera Iyer"""


def test_extract_booking_offline():
    booking = extract_booking_request(MockLLM(), EMAIL, year=2026)
    assert booking.guest_name == "Meera Iyer"
    assert (booking.check_in, booking.check_out) == (date(2026, 11, 12), date(2026, 11, 15))
    assert booking.guests == 2 and booking.room_type == "Deluxe Sea View"
    assert booking.special_requests == ["an airport pickup"]


def test_extract_rejects_impossible_dates():
    with pytest.raises(ExtractionError):
        extract_booking_request(MockLLM(), "Room from 15 Nov to 12 Nov please. Thanks, Ravi Kumar", year=2026)


class ScriptedLLM:
    """Fake real-provider LLM: first answer is invalid, second is corrected (tests the self-correction loop)."""

    name = "scripted"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, tools):
        self.calls.append(messages)
        return LLMResponse(self.replies.pop(0))


def test_extract_self_corrects_after_validation_error():
    bad = '{"guest_name": "Meera", "check_in": "2026-11-15", "check_out": "2026-11-12", "guests": 2}'
    good = (
        '```json\n{"guest_name": "Meera Iyer", "check_in": "2026-11-12", "check_out": "2026-11-15", "guests": 2,'
        ' "room_type": "Deluxe Sea View", "special_requests": []}\n```'
    )
    llm = ScriptedLLM([bad, good])
    booking = extract_booking_request(llm, EMAIL, year=2026)
    assert booking.check_out == date(2026, 11, 15)
    assert "invalid" in llm.calls[1][-1]["content"]


def test_extract_endpoint(client, auth):
    res = client.post("/v1/extract/booking-request", headers=auth, json={"email": EMAIL})
    assert res.status_code == 200
    assert res.json()["guest_name"] == "Meera Iyer"


def test_import_bookings_from_csv(tmp_path):
    csv_file = tmp_path / "bookings.csv"
    csv_file.write_text(
        "Reference,Guest_Name,Room_Type,Check_In,Check_Out,Status\n"
        "SG9001,Kiran Das,Suite,2026-12-01,2026-12-04,confirmed\n"
        "SG9002,Neha Shah,Standard Twin,05/12/2026,07/12/2026,\n"
        "bad,Someone,Suite,2026-12-01,2026-12-02,\n"
        "SG9003,Late Guest,Suite,2026-12-05,2026-12-01,\n"
        "SG9001,Kiran Das,Suite,2026-12-01,2026-12-04,confirmed\n"
    )
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    with sessionmaker(bind=engine)() as session:
        report = import_bookings(session, read_rows(csv_file))
        assert (report.inserted, report.updated) == (2, 0)
        assert len(report.errors) == 3 and report.errors[0].startswith("row 4")
        assert session.scalar(select(Booking).where(Booking.reference == "SG9002")).check_in == date(2026, 12, 5)

        rerun = import_bookings(session, read_rows(csv_file)[:2])  # idempotent re-run updates, never duplicates
        assert (rerun.inserted, rerun.updated) == (0, 2)


def test_import_bookings_from_excel(tmp_path):
    from openpyxl import Workbook

    wb = Workbook()
    wb.active.append(["reference", "guest_name", "room_type", "check_in", "check_out"])
    wb.active.append(["SG9100", "Excel Guest", "Suite", date(2026, 12, 10), date(2026, 12, 12)])
    path = tmp_path / "bookings.xlsx"
    wb.save(path)
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    with sessionmaker(bind=engine)() as session:
        assert import_bookings(session, read_rows(path)).inserted == 1
