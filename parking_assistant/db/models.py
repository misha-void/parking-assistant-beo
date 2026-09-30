"""SQLAlchemy models for dynamic parking data."""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum, Boolean, Time
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

from parking_assistant.db.database import Base


class ParkingType(enum.Enum):
    """Types of parking facilities in Belgrade."""
    GARAGE = "garage"                    # Indoor/covered parking structure
    CAR_PARK = "car_park"                # Outdoor parking lot
    RESERVED_GARAGE = "reserved_garage"  # Garage with pre-bookable spots


class ParkingZone(enum.Enum):
    """Belgrade parking zones with time restrictions."""
    PURPLE_A = "purple_a"  # Zone A - 30 min max
    RED_1 = "red_1"        # Zone 1 - 60 min max
    WHITE_B = "white_b"    # Zone B - 120 min max
    YELLOW_2 = "yellow_2"  # Zone 2 - 120 min max
    GREEN_3 = "green_3"    # Zone 3 - 180 min max


class ParkingLocation(Base):
    """
    Parking locations in Belgrade.
    
    Represents physical parking facilities (garages, car parks, reserved garages)
    with their static attributes.
    """
    __tablename__ = "parking_location"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False, index=True)
    type = Column(Enum(ParkingType), nullable=False, index=True)
    zone = Column(Enum(ParkingZone), nullable=True, index=True)  # Null for garages not in street zones
    address = Column(String(500), nullable=False)
    capacity = Column(Integer, nullable=False)  # Total number of parking spots
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    description = Column(String(1000), nullable=True)  # Optional additional info
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    availability_records = relationship("Availability", back_populates="location", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ParkingLocation(id={self.id}, name='{self.name}', type={self.type.value}, zone={self.zone})>"


class PriceRule(Base):
    """
    Pricing rules per zone.
    
    Stores hourly rates, maximum parking duration, and extension times
    for each parking zone in Belgrade.
    """
    __tablename__ = "price_rule"

    id = Column(Integer, primary_key=True, index=True)
    zone = Column(Enum(ParkingZone), nullable=False, unique=True, index=True)
    hourly_rate_rsd = Column(Integer, nullable=False)  # Price in Serbian Dinars per hour
    max_duration_minutes = Column(Integer, nullable=False)  # Maximum continuous parking time
    extended_time_minutes = Column(Integer, nullable=True)  # Additional time allowed after max duration
    cooldown_minutes = Column(Integer, default=30)  # Time before you can park in same zone again
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<PriceRule(zone={self.zone.value}, rate={self.hourly_rate_rsd} RSD/h, max={self.max_duration_minutes}min)>"


class WorkingHours(Base):
    """
    Parking control and payment working hours per zone.
    
    Defines when parking control applies and payment is required
    for each zone and day of week.
    """
    __tablename__ = "working_hours"

    id = Column(Integer, primary_key=True, index=True)
    zone = Column(Enum(ParkingZone), nullable=False, index=True)
    day_of_week = Column(Integer, nullable=False)  # 0=Monday, 1=Tuesday, ..., 6=Sunday
    open_time = Column(Time, nullable=False)  # When parking control starts
    close_time = Column(Time, nullable=False)  # When parking control ends
    is_charged = Column(Boolean, default=True)  # Whether payment is required during these hours
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        return f"<WorkingHours(zone={self.zone.value}, {days[self.day_of_week]} {self.open_time}-{self.close_time})>"


class Availability(Base):
    """
    Real-time parking availability snapshots.

    Tracks available and occupied spots for each parking location
    at different timestamps. Simulated data for learning project
    (no real-time API integration).
    """
    __tablename__ = "availability"

    id = Column(Integer, primary_key=True, index=True)
    location_id = Column(Integer, ForeignKey("parking_location.id"), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    available_spots = Column(Integer, nullable=False)
    occupied_spots = Column(Integer, nullable=False)

    # Relationships
    location = relationship("ParkingLocation", back_populates="availability_records")

    def __repr__(self):
        return f"<Availability(location_id={self.location_id}, available={self.available_spots}/{self.occupied_spots + self.available_spots})>"


class ReservationStatus(enum.Enum):
    """Reservation workflow states."""
    PENDING_APPROVAL = "pending_approval"  # Submitted, waiting for staff review
    CONFIRMED = "confirmed"                # Approved by staff
    CANCELLED = "cancelled"                # Cancelled by user or staff
    COMPLETED = "completed"                # Reservation period ended


class Reservation(Base):
    """
    Parking spot reservations.

    Stores user reservation requests with personal details.
    Status workflow: pending_approval -> confirmed -> completed
    (or cancelled at any stage).
    """
    __tablename__ = "reservation"

    id = Column(Integer, primary_key=True, index=True)

    # User information
    user_name = Column(String(100), nullable=False)
    user_surname = Column(String(100), nullable=False)
    car_number = Column(String(20), nullable=False, index=True)

    # Reservation details
    location_id = Column(Integer, ForeignKey("parking_location.id"), nullable=False, index=True)
    start_datetime = Column(DateTime(timezone=True), nullable=False)
    end_datetime = Column(DateTime(timezone=True), nullable=False)

    # Workflow
    status = Column(Enum(ReservationStatus), nullable=False, default=ReservationStatus.PENDING_APPROVAL, index=True)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    location = relationship("ParkingLocation")

    def __repr__(self):
        return f"<Reservation(id={self.id}, user={self.user_name} {self.user_surname}, status={self.status.value})>"