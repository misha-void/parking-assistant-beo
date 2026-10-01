"""
Seed SQL database with parking locations and simulated dynamic data.

This script:
1. Reads parking locations from data/parking_locations.json
2. Creates price rules for each zone
3. Creates working hours schedules
4. Generates simulated availability snapshots
"""

import sys
import json
import random
from pathlib import Path
from datetime import datetime, time, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parking_assistant.db.database import init_db, SessionLocal
from parking_assistant.db.models import (
    ParkingLocation, ParkingType, ParkingZone,
    PriceRule, WorkingHours, Availability, Reservation
)
from dotenv import load_dotenv

load_dotenv()


def clear_existing(session):
    """Wipe dynamic tables so the seed can be re-run without UNIQUE/FK errors."""
    print("🧹 Clearing existing data...")
    for model in (Availability, Reservation, WorkingHours, PriceRule, ParkingLocation):
        session.query(model).delete()
    session.commit()


def seed_price_rules(session):
    """Create price rules for each zone."""
    print("💰 Seeding price rules...")
    
    price_rules = [
        PriceRule(
            zone=ParkingZone.PURPLE_A,
            hourly_rate_rsd=150,
            max_duration_minutes=30,
            extended_time_minutes=0,  # No extension
            cooldown_minutes=30
        ),
        PriceRule(
            zone=ParkingZone.RED_1,
            hourly_rate_rsd=120,
            max_duration_minutes=60,
            extended_time_minutes=30,
            cooldown_minutes=30
        ),
        PriceRule(
            zone=ParkingZone.WHITE_B,
            hourly_rate_rsd=100,
            max_duration_minutes=120,
            extended_time_minutes=60,
            cooldown_minutes=30
        ),
        PriceRule(
            zone=ParkingZone.YELLOW_2,
            hourly_rate_rsd=80,
            max_duration_minutes=120,
            extended_time_minutes=60,
            cooldown_minutes=30
        ),
        PriceRule(
            zone=ParkingZone.GREEN_3,
            hourly_rate_rsd=60,
            max_duration_minutes=180,
            extended_time_minutes=60,
            cooldown_minutes=30
        ),
    ]
    
    session.add_all(price_rules)
    session.commit()
    print(f"  ✓ Created {len(price_rules)} price rules")


def seed_working_hours(session):
    """Create working hours schedules for each zone."""
    print("🕐 Seeding working hours...")
    
    working_hours = []
    
    # Purple, Red, White zones: Mon-Sat 7am-10pm, Sun 7am-2pm
    for zone in [ParkingZone.PURPLE_A, ParkingZone.RED_1, ParkingZone.WHITE_B]:
        # Monday to Saturday
        for day in range(6):  # 0-5 (Mon-Sat)
            working_hours.append(WorkingHours(
                zone=zone,
                day_of_week=day,
                open_time=time(7, 0),
                close_time=time(22, 0),
                is_charged=True
            ))
        # Sunday
        working_hours.append(WorkingHours(
            zone=zone,
            day_of_week=6,
            open_time=time(7, 0),
            close_time=time(14, 0),
            is_charged=True
        ))
    
    # Yellow, Green zones: Mon-Fri 7am-9pm, Sat 7am-2pm
    for zone in [ParkingZone.YELLOW_2, ParkingZone.GREEN_3]:
        # Monday to Friday
        for day in range(5):  # 0-4 (Mon-Fri)
            working_hours.append(WorkingHours(
                zone=zone,
                day_of_week=day,
                open_time=time(7, 0),
                close_time=time(21, 0),
                is_charged=True
            ))
        # Saturday
        working_hours.append(WorkingHours(
            zone=zone,
            day_of_week=5,
            open_time=time(7, 0),
            close_time=time(14, 0),
            is_charged=True
        ))
    
    session.add_all(working_hours)
    session.commit()
    print(f"  ✓ Created {len(working_hours)} working hour records")


def seed_parking_locations(session):
    """Load parking locations from JSON file."""
    print("🚗 Seeding parking locations...")
    
    json_path = Path("data/parking_locations.json")
    
    if not json_path.exists():
        print("  ⚠️  parking_locations.json not found!")
        return []
    
    with open(json_path, "r", encoding="utf-8") as f:
        locations_data = json.load(f)
    
    locations = []
    for data in locations_data:
        # Skip TODO placeholders
        if "TODO" in data.get("name", ""):
            continue
        
        # Map string type to enum
        type_map = {
            "garage": ParkingType.GARAGE,
            "car_park": ParkingType.CAR_PARK,
            "reserved_garage": ParkingType.RESERVED_GARAGE
        }
        
        # Map string zone to enum (if present)
        zone_map = {
            "purple_a": ParkingZone.PURPLE_A,
            "red_1": ParkingZone.RED_1,
            "white_b": ParkingZone.WHITE_B,
            "yellow_2": ParkingZone.YELLOW_2,
            "green_3": ParkingZone.GREEN_3
        }
        
        location = ParkingLocation(
            name=data["name"],
            type=type_map[data["type"]],
            zone=zone_map.get(data["zone"]) if data.get("zone") else None,
            address=data["address"],
            capacity=data["capacity"],
            latitude=data.get("latitude"),
            longitude=data.get("longitude"),
            description=data.get("description")
        )
        locations.append(location)
    
    session.add_all(locations)
    session.commit()
    print(f"  ✓ Created {len(locations)} parking locations")
    
    return locations


def seed_availability(session, locations):
    """Generate simulated availability snapshots for each location."""
    print("📊 Generating simulated availability data...")
    
    availability_records = []
    now = datetime.now()

    for location in locations:
        # Generate 3 snapshots per location with distinct timestamps (now-2h, now-1h, now)
        for i in range(3):
            snapshot_time = now - timedelta(hours=2 - i)
            # Random occupancy: 30-90% full
            occupancy_rate = random.uniform(0.3, 0.9)
            occupied = int(location.capacity * occupancy_rate)
            available = location.capacity - occupied
            
            availability_records.append(Availability(
                location_id=location.id,
                available_spots=available,
                occupied_spots=occupied,
                timestamp=snapshot_time
            ))
    
    session.add_all(availability_records)
    session.commit()
    print(f"  ✓ Created {len(availability_records)} availability snapshots")


def seed_database():
    """Main seeding function."""
    print("🌱 Seeding database with parking data...\n")
    
    # Initialize database (create tables)
    print("🔧 Initializing database schema...")
    init_db()
    print()
    
    # Create session
    session = SessionLocal()
    
    try:
        # Clear first so re-running the seed is safe
        clear_existing(session)

        # Seed in order (respecting foreign key constraints)
        seed_price_rules(session)
        seed_working_hours(session)
        locations = seed_parking_locations(session)
        seed_availability(session, locations)
        
        print("\n✅ Database seeding complete!")
        print(f"\nCreated:")
        print(f"  • {session.query(ParkingLocation).count()} parking locations")
        print(f"  • {session.query(PriceRule).count()} price rules")
        print(f"  • {session.query(WorkingHours).count()} working hour records")
        print(f"  • {session.query(Availability).count()} availability snapshots")
        
    except Exception as e:
        print(f"\n❌ Error seeding database: {e}")
        session.rollback()
        raise
    
    finally:
        session.close()


if __name__ == "__main__":
    seed_database()
