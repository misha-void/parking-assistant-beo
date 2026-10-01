"""Retrieval functions for RAG system - queries vector DB and SQL database."""

from typing import List, Dict, Any, Optional
from parking_assistant.rag.vector_store import MilvusVectorStorage
from parking_assistant.db.database import SessionLocal
from parking_assistant.db.models import ParkingLocation, PriceRule, WorkingHours, Availability, ParkingType, ParkingZone
from sqlalchemy import func, and_


def price_label(price) -> str:
    """Human label for an hourly price (None = private garage without a published rate)."""
    return f"{price} RSD/hour" if price else "price varies (private garage)"


def get_bookable_locations() -> List[Dict[str, Any]]:
    """All bookable locations with their hourly price (SQL only; None price for garages)."""
    session = SessionLocal()
    try:
        rate_by_zone = {pr.zone: pr.hourly_rate_rsd for pr in session.query(PriceRule).all()}
        return [
            {
                "id": l.id,
                "name": l.name,
                "address": l.address,
                "price": rate_by_zone.get(l.zone) if l.zone else None,
            }
            for l in session.query(ParkingLocation).all()
        ]
    finally:
        session.close()


class Retriever:
    """Handles retrieval from both vector DB and SQL database."""
    
    def __init__(self):
        self.vector_store = MilvusVectorStorage()
        
    def close(self):
        """Close vector store connection."""
        self.vector_store.close()
    
    async def retrieve_general_info(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieve general information from vector DB (zones, rules, FAQs, parking types).
        
        Returns dict with 'source' and 'context' keys.
        """
        # Search across all collections, prioritize based on query
        all_results = []
        
        for collection in self.vector_store.ALL_COLLECTIONS:
            results = self.vector_store.search(collection, query, top_k=2)
            all_results.extend(results)
        
        # Sort by score and take top_k
        all_results.sort(key=lambda x: x['score'], reverse=True)
        top_results = all_results[:top_k]
        
        # Format context
        context_parts = []
        for i, result in enumerate(top_results, 1):
            context_parts.append(
                f"[Source {i}: {result['source_file']}]\n{result['text']}\n"
            )
        
        return {
            "source": "vector_db",
            "context": "\n".join(context_parts),
            "num_results": len(top_results)
        }
    
    async def retrieve_pricing_hours(self, query: str) -> Dict[str, Any]:
        """Retrieve pricing and hours info from vector DB + SQL."""
        # Get general pricing context from vector DB
        vector_results = self.vector_store.search(
            self.vector_store.COLLECTION_PRICING,
            query,
            top_k=2
        )
        
        # Get price rules from SQL
        session = SessionLocal()
        try:
            price_rules = session.query(PriceRule).all()
            working_hours = session.query(WorkingHours).limit(10).all()  # Sample
            
            # Format SQL data
            price_info = "\n".join([
                f"- {rule.zone.value}: {rule.hourly_rate_rsd} RSD/hour, "
                f"max {rule.max_duration_minutes} min"
                for rule in price_rules
            ])
            
            context_parts = []
            
            # Add vector context
            if vector_results:
                context_parts.append("[General Pricing Information]\n" + vector_results[0]['text'])
            
            # Add SQL data
            context_parts.append(f"\n[Current Rates]\n{price_info}")
            
            return {
                "source": "hybrid",
                "context": "\n\n".join(context_parts),
                "num_results": len(vector_results) + len(price_rules)
            }
        finally:
            session.close()
    
    async def retrieve_availability(
        self,
        parking_type: Optional[str] = None,
        zone: Optional[str] = None,
        min_available: int = 0
    ) -> Dict[str, Any]:
        """
        Search for available parking locations.
        
        Args:
            parking_type: Filter by type (garage, car_park, reserved_garage)
            zone: Filter by zone (purple_a, red_1, etc.)
            min_available: Minimum available spots
        
        Returns formatted availability data
        """
        session = SessionLocal()
        try:
            # Get latest availability per location
            subquery = session.query(
                Availability.location_id,
                func.max(Availability.timestamp).label('max_timestamp')
            ).group_by(Availability.location_id).subquery()
            
            query = session.query(
                ParkingLocation.name,
                ParkingLocation.type,
                ParkingLocation.zone,
                ParkingLocation.address,
                ParkingLocation.capacity,
                Availability.available_spots,
                Availability.occupied_spots,
                ParkingLocation.id,
                Availability.timestamp
            ).join(
                Availability, ParkingLocation.id == Availability.location_id
            ).join(
                subquery,
                and_(
                    Availability.location_id == subquery.c.location_id,
                    Availability.timestamp == subquery.c.max_timestamp
                )
            ).filter(
                Availability.available_spots >= min_available
            )
            
            # Apply filters
            if parking_type:
                type_map = {
                    "garage": ParkingType.GARAGE,
                    "car_park": ParkingType.CAR_PARK,
                    "reserved_garage": ParkingType.RESERVED_GARAGE
                }
                if parking_type in type_map:
                    query = query.filter(ParkingLocation.type == type_map[parking_type])
            
            if zone:
                zone_map = {
                    "purple_a": ParkingZone.PURPLE_A,
                    "red_1": ParkingZone.RED_1,
                    "white_b": ParkingZone.WHITE_B,
                    "yellow_2": ParkingZone.YELLOW_2,
                    "green_3": ParkingZone.GREEN_3
                }
                if zone in zone_map:
                    query = query.filter(ParkingLocation.zone == zone_map[zone])
            
            rows = query.order_by(Availability.timestamp.desc()).all()

            # Keep exactly one row per location (latest; robust to timestamp ties)
            seen_ids = set()
            results = []
            for row in rows:
                if row[7] in seen_ids:
                    continue
                seen_ids.add(row[7])
                results.append(row)

            rate_by_zone = {pr.zone: pr.hourly_rate_rsd for pr in session.query(PriceRule).all()}

            # Format results
            if not results:
                return {
                    "source": "sql",
                    "context": "No parking locations found matching your criteria.",
                    "locations": []
                }
            
            formatted = []
            for name, ptype, pzone, address, capacity, available, occupied, _loc_id, _ts in results:
                zone_str = f" ({pzone.value})" if pzone else ""
                formatted.append(
                    f"• {name} [{ptype.value}]{zone_str}\n"
                    f"  Address: {address}\n"
                    f"  Price: {price_label(rate_by_zone.get(pzone))}\n"
                    f"  Available: {available}/{capacity} spots\n"
                )

            return {
                "source": "sql",
                "context": "\n".join(formatted),
                "locations": [
                    {
                        "id": r[7],
                        "name": r[0],
                        "type": r[1].value,
                        "zone": r[2].value if r[2] else None,
                        "address": r[3],
                        "price": rate_by_zone.get(r[2]),
                        "available": r[5]
                    }
                    for r in results
                ],
                "num_results": len(results)
            }
        finally:
            session.close()
    
    async def retrieve_hybrid(self, query: str) -> Dict[str, Any]:
        """Retrieve from both sources for mixed/complex queries."""
        # Get general context from vector DB
        vector_context = await self.retrieve_general_info(query, top_k=2)
        
        # Get availability snapshot
        availability_context = await self.retrieve_availability()
        
        combined = (
            f"[Knowledge Base]\n{vector_context['context']}\n\n"
            f"[Current Availability]\n{availability_context['context']}"
        )
        
        return {
            "source": "hybrid",
            "context": combined,
            "num_results": vector_context['num_results'] + availability_context.get('num_results', 0)
        }


async def get_retriever() -> Retriever:
    """Factory function to get retriever instance."""
    return Retriever()
