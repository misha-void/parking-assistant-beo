"""Tests for reservation slot collection."""

import pytest
from parking_assistant.graph.slot_collector import SlotCollector, ReservationSlots


def test_slots_initially_empty():
    """Test new ReservationSlots starts empty."""
    slots = ReservationSlots()
    
    assert not slots.is_complete()
    assert len(slots.missing_slots()) == 6


def test_slots_fill_progressively():
    """Test filling slots one by one."""
    slots = ReservationSlots()
    
    slots.name = "John"
    assert "name" not in slots.missing_slots()
    assert not slots.is_complete()
    
    slots.surname = "Doe"
    slots.car_number = "BG123AB"
    slots.location_id = 1
    slots.start_date = "2024-01-01T10:00:00"
    slots.end_date = "2024-01-01T18:00:00"
    
    assert slots.is_complete()
    assert len(slots.missing_slots()) == 0


def test_collector_validates_car_number():
    """Test car number validation."""
    collector = SlotCollector()
    
    assert collector.validate_car_number("BG123AB")
    assert collector.validate_car_number("ABC-123")
    assert not collector.validate_car_number("invalid")
    assert not collector.validate_car_number("X")


def test_collector_prompts():
    """Test collector returns correct prompts."""
    collector = SlotCollector()
    slots = ReservationSlots()
    
    # Should ask for name first
    prompt = collector.get_next_prompt(slots)
    assert "name" in prompt.lower()
    
    # Fill name, should ask for surname
    slots.name = "John"
    prompt = collector.get_next_prompt(slots)
    assert "surname" in prompt.lower()


def test_confirmation_format():
    """Test confirmation message formatting."""
    collector = SlotCollector()
    slots = ReservationSlots(
        name="John",
        surname="Doe",
        car_number="BG123AB",
        location_id=1,
        start_date="2024-01-01T10:00:00",
        end_date="2024-01-01T18:00:00"
    )
    
    confirmation = collector.format_confirmation(slots, "Test Garage")
    
    assert "John" in confirmation
    assert "Doe" in confirmation
    assert "BG123AB" in confirmation
    assert "Test Garage" in confirmation
