"""Tests for SQL database models and queries."""

import pytest
from parking_assistant.db.models import ParkingLocation, ParkingType, ParkingZone, PriceRule, Availability
from sqlalchemy import func


def test_parking_locations_exist(db_session):
    """Test that parking locations are seeded."""
    locations = db_session.query(ParkingLocation).all()
    assert len(locations) > 0, "No parking locations found in database"


def test_filter_by_type(db_session):
    """Test filtering parking locations by type."""
    garages = db_session.query(ParkingLocation).filter(
        ParkingLocation.type == ParkingType.GARAGE
    ).all()
    
    assert len(garages) > 0, "No garages found"
    for garage in garages:
        assert garage.type == ParkingType.GARAGE


def test_filter_by_zone(db_session):
    """Test filtering by parking zone."""
    green_zone = db_session.query(ParkingLocation).filter(
        ParkingLocation.zone == ParkingZone.GREEN_3
    ).all()
    
    # May or may not have zone 3 locations depending on seed data
    for loc in green_zone:
        assert loc.zone == ParkingZone.GREEN_3


def test_price_rules_complete(db_session):
    """Test that all zones have price rules."""
    price_rules = db_session.query(PriceRule).all()
    
    assert len(price_rules) == 5, "Should have 5 price rules (one per zone)"
    
    zones_covered = {rule.zone for rule in price_rules}
    expected_zones = {
        ParkingZone.PURPLE_A,
        ParkingZone.RED_1,
        ParkingZone.WHITE_B,
        ParkingZone.YELLOW_2,
        ParkingZone.GREEN_3
    }
    
    assert zones_covered == expected_zones


def test_availability_join(db_session):
    """Test joining locations with availability data."""
    # Get latest availability per location
    subquery = db_session.query(
        Availability.location_id,
        func.max(Availability.timestamp).label('max_timestamp')
    ).group_by(Availability.location_id).subquery()
    
    results = db_session.query(
        ParkingLocation.name,
        ParkingLocation.capacity,
        Availability.available_spots
    ).join(
        Availability, ParkingLocation.id == Availability.location_id
    ).join(
        subquery,
        (Availability.location_id == subquery.c.location_id) &
        (Availability.timestamp == subquery.c.max_timestamp)
    ).all()
    
    assert len(results) > 0, "No availability data found"
    
    for name, capacity, available in results:
        assert available <= capacity, f"Available spots ({available}) exceed capacity ({capacity})"


def test_complex_query(db_session):
    """Test complex filtering query."""
    # Find garages with >100 available spots
    results = db_session.query(
        ParkingLocation.name,
        Availability.available_spots
    ).join(
        Availability, ParkingLocation.id == Availability.location_id
    ).filter(
        ParkingLocation.type == ParkingType.GARAGE,
        Availability.available_spots > 100
    ).all()
    
    # Result depends on seed data, just check structure
    for name, available in results:
        assert available > 100
