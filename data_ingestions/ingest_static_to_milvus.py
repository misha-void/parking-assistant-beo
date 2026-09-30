"""
Ingest static markdown documents into Milvus vector database.

This script:
1. Reads markdown files from data/static/
2. Chunks them into manageable pieces
3. Embeds using OpenAI embeddings
4. Stores in appropriate Milvus collections
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from parking_assistant.rag.vector_store import MilvusVectorStorage
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()


def chunk_markdown(text: str, source_file: str) -> list[str]:
    """
    Chunk markdown text intelligently by headers and paragraphs.
    
    Args:
        text: Markdown content
        source_file: Name of source file (for logging)
    
    Returns:
        List of text chunks
    """
    # First split by headers
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    header_chunks = markdown_splitter.split_text(text)
    
    # Then recursively split large chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Target ~500 chars per chunk
        chunk_overlap=50,  # 50 char overlap for context
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    final_chunks = []
    for doc in header_chunks:
        # Combine header metadata with content
        content = doc.page_content
        if doc.metadata:
            # Prepend headers for context
            header_text = " > ".join([v for v in doc.metadata.values() if v])
            content = f"{header_text}\n\n{content}"
        
        # Split if too large
        sub_chunks = text_splitter.split_text(content)
        final_chunks.extend(sub_chunks)
    
    print(f"  ✓ {source_file}: {len(final_chunks)} chunks created")
    return final_chunks


def ingest_static_data():
    """Main ingestion function."""
    print("🚀 Starting static data ingestion to Milvus...\n")
    
    # Initialize vector store
    vector_store = MilvusVectorStorage()
    
    # Create collections (drop existing to start fresh)
    print("📦 Creating Milvus collections...")
    vector_store.create_all_collections(drop_existing=True)
    print()
    
    # Define file-to-collection mapping
    data_dir = Path("data/static")
    file_collection_map = {
        "zone_system.md": vector_store.COLLECTION_ZONE_RULES,
        "pricing_and_hours.md": vector_store.COLLECTION_PRICING,
        "parking_types.md": vector_store.COLLECTION_LOCATIONS_INFO,
        "faq.md": vector_store.COLLECTION_FAQ,
    }
    
    # Process each file
    print("📄 Processing markdown files...\n")
    for filename, collection_name in file_collection_map.items():
        file_path = data_dir / filename
        
        if not file_path.exists():
            print(f"  ⚠️  Skipping {filename} (not found)")
            continue
        
        print(f"📖 {filename} → {collection_name}")
        
        # Read file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Chunk content
        chunks = chunk_markdown(content, filename)
        
        # Insert into Milvus
        vector_store.insert_documents(
            collection_name=collection_name,
            texts=chunks,
            source_file=filename
        )
        print()
    
    print("✅ Ingestion complete!")
    print(f"\nCreated {len(vector_store.ALL_COLLECTIONS)} collections:")
    for col in vector_store.ALL_COLLECTIONS:
        print(f"  • {col}")
    
    vector_store.close()


if __name__ == "__main__":
    ingest_static_data()
