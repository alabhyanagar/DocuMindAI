"""
vector_db.py - ChromaDB Vector Store & Embedding Manager for DocuMind AI
"""

import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from ingest import extract_text_and_tables_from_pdf, create_parent_child_chunks

# Setup Persistent ChromaDB Client
DB_DIR = Path("vector_db_storage")
DB_DIR.mkdir(exist_ok=True)

client = chromadb.PersistentClient(path=str(DB_DIR))

# Use a lightweight, high-performance open-source embedding model from Hugging Face
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Get or create our document collection
collection = client.get_or_create_collection(
    name="documind_enterprise_docs",
    embedding_function=embedding_fn
)

def ingest_pdf_to_vector_db(pdf_path: str):
    """
    Full Ingestion Pipeline:
    1. Parses PDF text and tables.
    2. Generates Parent-Child chunks.
    3. Embeds and stores child chunks in ChromaDB with parent text mapped in metadata.
    """
    print(f"Starting ingestion for: {pdf_path}")
    
    # 1. Parse
    pages = extract_text_and_tables_from_pdf(pdf_path)
    if not pages:
        print("No text extracted from PDF.")
        return 0
        
    # 2. Chunk
    chunks = create_parent_child_chunks(pages)
    if not chunks:
        print("No valid chunks created.")
        return 0
        
    # 3. Prepare data for ChromaDB batch insertion
    ids = [chunk["child_id"] for chunk in chunks]
    documents = [chunk["child_text"] for chunk in chunks] # Chroma embeds this text
    
    # Store metadata including the larger parent text for context retrieval later
    metadatas = [
        {
            "parent_id": chunk["parent_id"],
            "parent_text": chunk["parent_text"],
            "source_file": chunk["metadata"]["source_file"],
            "page_number": chunk["metadata"]["page_number"]
        }
        for chunk in chunks
    ]
    
    # 4. Upsert into ChromaDB
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"Successfully indexed {len(chunks)} child chunks into ChromaDB!")
    return len(chunks)

def query_vector_store(query_text: str, n_results: int = 5):
    """
    Performs pure semantic vector search against ChromaDB.
    """
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results