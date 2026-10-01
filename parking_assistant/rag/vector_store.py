"""Milvus Lite vector store adapter for parking assistant static data."""

from pathlib import Path
from typing import List, Dict, Any, Optional
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from langchain_openai import OpenAIEmbeddings
import os


class MilvusVectorStorage:
    """
    Adapter for Milvus Lite with multiple collections for different content types.
    
    Collections:
    - parking_zone_rules: Zone system, time limits, restrictions
    - parking_pricing: Pricing rules, payment methods
    - parking_locations_info: General info about parking types
    - parking_faq: Frequently asked questions
    """

    # Collection names
    COLLECTION_ZONE_RULES = "parking_zone_rules"
    COLLECTION_PRICING = "parking_pricing"
    COLLECTION_LOCATIONS_INFO = "parking_locations_info"
    COLLECTION_FAQ = "parking_faq"

    ALL_COLLECTIONS = [
        COLLECTION_ZONE_RULES,
        COLLECTION_PRICING,
        COLLECTION_LOCATIONS_INFO,
        COLLECTION_FAQ
    ]

    def __init__(self, embedding_model: Optional[OpenAIEmbeddings] = None):
        """
        Initialize Milvus connection and embedding model.
        
        Args:
            embedding_model: OpenAIEmbeddings instance (if None, creates default)
        """
        # Use Milvus Lite (embedded, file-based).
        # NOTE: the env var is MILVUS_DB_PATH, not MILVUS_URI — pymilvus reserves
        # MILVUS_URI for its own default (http) connection and fails to import if
        # it is set to a local file path.
        self.milvus_uri = os.getenv("MILVUS_DB_PATH", "./milvus_data/parking_assistant.db")

        # Milvus Lite needs the parent directory to exist beforehand
        Path(self.milvus_uri).parent.mkdir(parents=True, exist_ok=True)

        # Connect to Milvus Lite
        connections.connect(
            alias="default",
            uri=self.milvus_uri
        )
        print(f"✅ Connected to Milvus Lite at {self.milvus_uri}")

        # Initialize embedding model
        if embedding_model is None:
            from parking_assistant.llm_factory import get_embeddings
            self.embeddings = get_embeddings()
        else:
            self.embeddings = embedding_model

        # Embedding dimension for text-embedding-3-small
        self.embedding_dim = 1536

    def create_collection(self, collection_name: str, drop_existing: bool = False):
        """
        Create a Milvus collection with schema for text chunks + embeddings.
        
        Args:
            collection_name: Name of the collection
            drop_existing: If True, drop existing collection before creating
        """
        # Drop if exists and requested
        if drop_existing and utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"🗑️  Dropped existing collection: {collection_name}")

        # Skip if already exists
        if utility.has_collection(collection_name):
            print(f"ℹ️  Collection already exists: {collection_name}")
            return

        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
        ]

        schema = CollectionSchema(
            fields=fields,
            description=f"Static parking data: {collection_name}"
        )

        # Create collection
        collection = Collection(name=collection_name, schema=schema)

        # Create index for vector search (IVF_FLAT is simple and works for small datasets)
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",  # Cosine similarity for semantic search
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print(f"✅ Created collection: {collection_name}")

    def create_all_collections(self, drop_existing: bool = False):
        """Create all parking assistant collections."""
        for collection_name in self.ALL_COLLECTIONS:
            self.create_collection(collection_name, drop_existing=drop_existing)

    def insert_documents(
        self,
        collection_name: str,
        texts: List[str],
        source_file: str
    ):
        """
        Embed and insert text chunks into a collection.
        
        Args:
            collection_name: Target collection name
            texts: List of text chunks to insert
            source_file: Name of the source markdown file
        """
        if not texts:
            print(f"⚠️  No texts to insert for {source_file}")
            return

        # Generate embeddings
        print(f"🔄 Embedding {len(texts)} chunks from {source_file}...")
        embeddings = self.embeddings.embed_documents(texts)

        # Prepare data
        data = [
            texts,  # text field
            embeddings,  # embedding field
            [source_file] * len(texts),  # source_file field
            list(range(len(texts)))  # chunk_index field
        ]

        # Insert into collection
        collection = Collection(collection_name)
        collection.insert(data)
        collection.flush()  # Ensure data is persisted
        
        print(f"✅ Inserted {len(texts)} chunks into {collection_name}")

    def search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Semantic search in a collection.
        
        Args:
            collection_name: Collection to search in
            query: User's query text
            top_k: Number of results to return
        
        Returns:
            List of dicts with keys: text, score, source_file, chunk_index
        """
        # Embed query
        query_embedding = self.embeddings.embed_query(query)

        # Load collection
        collection = Collection(collection_name)
        collection.load()

        # Search
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "source_file", "chunk_index"]
        )

        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "text": hit.entity.get("text"),
                    "score": hit.score,  # Cosine similarity score
                    "source_file": hit.entity.get("source_file"),
                    "chunk_index": hit.entity.get("chunk_index")
                })

        return formatted_results

    def close(self):
        """Close Milvus connection."""
        connections.disconnect(alias="default")
        print("✅ Disconnected from Milvus")


def get_vector_store() -> MilvusVectorStorage:
    """Factory function to get a MilvusVectorStorage instance."""
    return MilvusVectorStorage()