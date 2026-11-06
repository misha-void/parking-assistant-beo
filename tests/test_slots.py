"""Tests for reservation slots and their validation helpers."""

from parking_assistant.graph.slots import (
    ReservationSlots,
    format_confirmation,
    is_valid_name,
    is_valid_plate,
    parse_iso,
)


def _filled_slots() -> ReservationSlots:
    return ReservationSlots(
        name="John",
        surname="Doe",
        car_number="BG123AB",
        location_id=1,
        start_iso="2030-01-01T10:00:00",
        end_iso="2030-01-01T18:00:00",
    )


def test_new_slots_are_all_missing():
    slots = ReservationSlots()
    assert not slots.is_complete()
    assert slots.missing_slots() == [
        "name", "surname", "car_number", "location_id", "start_iso", "end_iso",
    ]


def test_slots_are_complete_once_every_field_is_set():
    slots = _filled_slots()
    assert slots.is_complete()
    assert slots.missing_slots() == []


def test_missing_slots_follow_the_asking_order():
    assert ReservationSlots(name="John").missing_slots()[0] == "surname"


def test_plate_validation():
    assert is_valid_plate("BG123AB")
    assert is_valid_plate("ABC-123")
    assert not is_valid_plate("invalid")  # letters only, no digit
    assert not is_valid_plate("X")  # too short


def test_name_validation_rejects_sentences():
    # A question used to be stored as the user's name when the interpreter failed open.
    assert is_valid_name("Mika")
    assert is_valid_name("Anna Maria")
    assert not is_valid_name("Do you have EV chargers?")
    assert not is_valid_name("")


def test_parse_iso_drops_the_timezone():
    # A timezone-aware reply must not poison later comparisons against naive values.
    assert parse_iso("2030-12-25T10:00:00Z").tzinfo is None
    assert parse_iso("not a date") is None
    assert parse_iso(None) is None


def test_confirmation_lists_the_booking_details():
    summary = format_confirmation(_filled_slots(), "Test Garage", "Price: 100 RSD/hour")
    for expected in ("John", "Doe", "BG123AB", "Test Garage", "100 RSD/hour"):
        assert expected in summary
