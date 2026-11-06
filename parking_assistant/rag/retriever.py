"""Retrieval functions for RAG system - queries vector DB and SQL database."""

from typing import Any

from sqlalchemy import and_, func

from parking_assistant.db.database import SessionLocal
from parking_assistant.db.models import (
    Availability,
    ParkingLocation,
    PriceRule,
    max_stay_minutes,
)
from parking_assistant.rag.vector_store import MilvusVectorStorage


def price_label(price) -> str:
    """Human label for an hourly price (None = zone without a published rate)."""
    return f"{price} RSD/hour" if price else "price varies (private garage)"


def zone_label(zone: str | None) -> str:
    """Turn a stored zone value such as 'green_3' into 'Green 3' for display."""
    return zone.replace("_", " ").title() if zone else ""


def format_location_line(loc: dict[str, Any]) -> str:
    """Render one location as '- name — address — zone — price [— N spots]'.

    The zone is included because the pricing context lists rates per zone: without it
    the model had to guess which zone a location belonged to, and invented the link.
    """
    parts = [f"- {loc.get('name')}", loc.get("address", "address n/a")]
    zone = zone_label(loc.get("zone"))
    parts.append(f"{zone} zone" if zone else "no zone")
    parts.append(price_label(loc.get("price")))
    line = " — ".join(parts)
    available = loc.get("available")
    if available is not None:
        capacity = loc.get("capacity")
        line += f" — {available}/{capacity} spots" if capacity else f" — {available} spots"
    return line


def _rate_by_zone(session) -> dict[Any, int]:
    """Map each parking zone to its hourly rate in RSD."""
    return {rule.zone: rule.hourly_rate_rsd for rule in session.query(PriceRule).all()}


def _max_stay_by_zone(session) -> dict[Any, int]:
    """Map each parking zone to its maximum stay in minutes."""
    return {rule.zone: rule.max_duration_minutes for rule in session.query(PriceRule).all()}





def get_bookable_locations() -> list[dict[str, Any]]:
    """All bookable locations with their hourly price (SQL only; None price for garages)."""
    session = SessionLocal()
    try:
        rates = _rate_by_zone(session)
        limits = _max_stay_by_zone(session)
        return [
            {
                "id": location.id,
                "name": location.name,
                "address": location.address,
                "price": rates.get(location.zone) if location.zone else None,
                "max_stay_minutes": max_stay_minutes(location, limits),
            }
            for location in session.query(ParkingLocation).all()
        ]
    finally:
        session.close()


def _latest_availability_rows(session) -> list[Any]:
    """One (ParkingLocation, Availability) row per location, using its latest snapshot."""
    latest = session.query(
        Availability.location_id,
        func.max(Availability.timestamp).label("max_timestamp"),
    ).group_by(Availability.location_id).subquery()

    rows = (
        session.query(ParkingLocation, Availability)
        .join(Availability, ParkingLocation.id == Availability.location_id)
        .join(
            latest,
            and_(
                Availability.location_id == latest.c.location_id,
                Availability.timestamp == latest.c.max_timestamp,
            ),
        )
        .order_by(Availability.timestamp.desc())
        .all()
    )

    # Robust to timestamp ties: keep the first row seen per location
    unique_rows = {}
    for location, availability in rows:
        unique_rows.setdefault(location.id, (location, availability))
    return list(unique_rows.values())


def _location_summary(location, availability, rates) -> dict[str, Any]:
    """Build the location dict returned to callers for one availability row."""
    return {
        "id": location.id,
        "name": location.name,
        "type": location.type.value,
        "zone": location.zone.value if location.zone else None,
        "address": location.address,
        "price": rates.get(location.zone),
        "capacity": location.capacity,
        "available": availability.available_spots,
    }


class Retriever:
    """Handles retrieval from both vector DB and SQL database."""

    def __init__(self):
        self.vector_store = MilvusVectorStorage()

    def close(self):
        """Close vector store connection."""
        self.vector_store.close()

    async def retrieve_general_info(self, query: str, top_k: int = 3) -> dict[str, Any]:
        """
        Retrieve general information from vector DB (zones, rules, FAQs, parking types).

        Returns dict with 'source', 'context' and 'num_results' keys.
        """
        all_results = []
        for collection in self.vector_store.ALL_COLLECTIONS:
            all_results.extend(self.vector_store.search(collection, query, top_k=2))

        all_results.sort(key=lambda result: result["score"], reverse=True)
        top_results = all_results[:top_k]

        context_parts = [
            f"[Source {i}: {result['source_file']}]\n{result['text']}\n"
            for i, result in enumerate(top_results, 1)
        ]

        return {
            "source": "vector_db",
            "context": "\n".join(context_parts),
            "num_results": len(top_results),
        }

    async def retrieve_pricing_hours(self, query: str) -> dict[str, Any]:
        """Retrieve pricing and hours info: vector DB text, zone rates and per-location prices."""
        vector_results = self.vector_store.search(
            self.vector_store.COLLECTION_PRICING, query, top_k=2
        )

        session = SessionLocal()
        try:
            price_rules = session.query(PriceRule).all()
        finally:
            session.close()

        rate_lines = "\n".join(
            f"- {rule.zone.value}: {rule.hourly_rate_rsd} RSD/hour, "
            f"max {rule.max_duration_minutes} min"
            for rule in price_rules
        )
        locations = get_bookable_locations()

        context_parts = []
        if vector_results:
            context_parts.append("[General Pricing Information]\n" + vector_results[0]["text"])
        context_parts.append(f"[Current Rates]\n{rate_lines}")
        context_parts.append(
            "[Prices per Location]\n" + "\n".join(format_location_line(loc) for loc in locations)
        )

        return {
            "source": "hybrid",
            "context": "\n\n".join(context_parts),
            "num_results": len(vector_results) + len(price_rules) + len(locations),
        }

    async def retrieve_availability(self) -> dict[str, Any]:
        """Return the latest availability and price of every parking location."""
        session = SessionLocal()
        try:
            rows = _latest_availability_rows(session)
            rates = _rate_by_zone(session)
        finally:
            session.close()

        if not rows:
            return {
                "source": "sql",
                "context": "No parking locations found.",
                "locations": [],
                "num_results": 0,
            }

        locations = [_location_summary(loc, avail, rates) for loc, avail in rows]
        return {
            "source": "sql",
            "context": "\n".join(format_location_line(loc) for loc in locations),
            "locations": locations,
            "num_results": len(locations),
        }

    async def retrieve_hybrid(self, query: str) -> dict[str, Any]:
        """Retrieve from both sources for mixed/complex queries."""
        vector_context = await self.retrieve_general_info(query, top_k=2)
        availability_context = await self.retrieve_availability()

        combined = (
            f"[Knowledge Base]\n{vector_context['context']}\n\n"
            f"[Current Availability]\n{availability_context['context']}"
        )

        return {
            "source": "hybrid",
            "context": combined,
            "num_results": vector_context["num_results"] + availability_context["num_results"],
        }
