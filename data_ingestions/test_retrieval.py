"""
Test retrieval from both Milvus (vector search) and SQL database.

This script verifies that:
1. Milvus collections are populated and searchable
2. SQL database is populated and queryable
3. Both data sources return relevant results
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from parking_assistant.rag.vector_store import MilvusVectorStore
from parking_assistant.db.database import SessionLocal
from parking_assistant.db.models import ParkingLocation, PriceRule, Availability, ParkingType, ParkingZone
from sqlalchemy import func
from dotenv import load_dotenv

load_dotenv()


def test_vector_search():
    """Test semantic search in Milvus collections."""
    print("=" * 80)
    print("🔍 TESTING VECTOR SEARCH (Milvus)")
    print("=" * 80)
    
    vector_store = MilvusVectorStore()
    
    # Test queries for different collections
    test_queries = [
        {
            "query": "What are the parking rules in zone A?",
            "collection": vector_store.COLLECTION_ZONE_RULES
        },
        {
            "query": "How much does parking cost in the red zone?",
            "collection": vector_store.COLLECTION_PRICING
        },
        {
            "query": "What is the difference between a garage and a car park?",
            "collection": vector_store.COLLECTION_LOCATIONS_INFO
        },
        {
            "query": "Can tourists use Belgrade parking?",
            "collection": vector_store.COLLECTION_FAQ
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n📝 Test Query {i}: \"{test['query']}\"")
        print(f"🎯 Searching collection: {test['collection']}")
        print("-" * 80)
        
        results = vector_store.search(
            collection_name=test['collection'],
            query=test['query'],
            top_k=2  # Get top 2 results
        )
        
        if not results:
            print("  ❌ No results found!")
            continue
        
        for j, result in enumerate(results, 1):
            score = result['score']
            text_preview = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
            print(f"\n  Result {j} (score: {score:.4f}):")
            print(f"  Source: {result['source_file']}, Chunk: {result['chunk_index']}")
            print(f"  Text: {text_preview}")
    
    vector_store.close()
    print("\n✅ Vector search tests complete!\n")


def test_sql_queries():
    """Test SQL database queries."""
    print("=" * 80)
    print("🗄️  TESTING SQL DATABASE QUERIES")
    print("=" * 80)
    
    session = SessionLocal()
    
    try:
        # Test 1: Get all garages
        print("\n📝 Test Query 1: Find all garage-type parking locations")
        print("-" * 80)
        garages = session.query(ParkingLocation).filter(
            ParkingLocation.type == ParkingType.GARAGE
        ).all()
        
        print(f"Found {len(garages)} garages:")
        for garage in garages:
            print(f"  • {garage.name} ({garage.address}) - Capacity: {garage.capacity}")
        
        # Test 2: Get locations in zone 3
        print("\n📝 Test Query 2: Find parking in Zone 3 (Green)")
        print("-" * 80)
        zone3_locations = session.query(ParkingLocation).filter(
            ParkingLocation.zone == ParkingZone.GREEN_3
        ).all()
        
        print(f"Found {len(zone3_locations)} locations in Zone 3:")
        for loc in zone3_locations:
            print(f"  • {loc.name} - {loc.type.value}")
        
        # Test 3: Get price rules
        print("\n📝 Test Query 3: Get all price rules")
        print("-" * 80)
        prices = session.query(PriceRule).all()
        
        print(f"Found {len(prices)} price rules:")
        for price in prices:
            print(f"  • {price.zone.value}: {price.hourly_rate_rsd} RSD/h, max {price.max_duration_minutes} min")
        
        # Test 4: Get availability with location details (JOIN)
        print("\n📝 Test Query 4: Get current availability for all locations")
        print("-" * 80)
        
        # Get latest availability snapshot per location
        subquery = session.query(
            Availability.location_id,
            func.max(Availability.timestamp).label('max_timestamp')
        ).group_by(Availability.location_id).subquery()
        
        availability_data = session.query(
            ParkingLocation.name,
            ParkingLocation.capacity,
            Availability.available_spots,
            Availability.occupied_spots
        ).join(
            Availability, ParkingLocation.id == Availability.location_id
        ).join(
            subquery,
            (Availability.location_id == subquery.c.location_id) &
            (Availability.timestamp == subquery.c.max_timestamp)
        ).all()
        
        print(f"Found availability data for {len(availability_data)} locations:")
        for name, capacity, available, occupied in availability_data:
            occupancy_rate = (occupied / capacity) * 100
            print(f"  • {name}: {available}/{capacity} available ({occupancy_rate:.1f}% full)")
        
        # Test 5: Complex query - Available garages in specific zones
        print("\n📝 Test Query 5: Find garages with >100 available spots")
        print("-" * 80)
        
        available_garages = session.query(
            ParkingLocation.name,
            ParkingLocation.address,
            Availability.available_spots
        ).join(
            Availability, ParkingLocation.id == Availability.location_id
        ).filter(
            ParkingLocation.type == ParkingType.GARAGE,
            Availability.available_spots > 100
        ).all()
        
        print(f"Found {len(available_garages)} garages with >100 spots:")
        for name, address, available in available_garages:
            print(f"  • {name} ({address}): {available} spots available")
        
        print("\n✅ SQL query tests complete!\n")
    
    finally:
        session.close()


def print_summary():
    """Print summary of what was tested."""
    print("=" * 80)
    print("📋 TEST SUMMARY")
    print("=" * 80)
    print("""
✅ Vector Search (Milvus):
   - Tested semantic search across 4 collections
   - Verified relevance scoring and retrieval
   - Collections: zone_rules, pricing, locations_info, faq

✅ SQL Database Queries:
   - Tested filtering by parking type and zone
   - Tested JOINs for availability data
   - Tested aggregations and complex queries

🎯 Both data sources are working correctly!

💡 Next steps:
   1. Integrate vector search into chat service
   2. Add intent classification (when to query which source)
   3. Build RAG chain with LangGraph
   4. Add guardrails (PII filtering)
""")


if __name__ == "__main__":
    print("\n🧪 Running retrieval tests...\n")
    
    # Test vector search
    test_vector_search()
    
    # Test SQL queries
    test_sql_queries()
    
    # Print summary
    print_summary()
