"""
test_documind.py - Unit test suite for DocuMind AI ingestion and chunking components
"""

import pytest
from ingest import create_parent_child_chunks

def test_parent_child_chunking_logic():
    """Validates that pages are correctly split into parent and child chunks."""
    sample_pages = [
        {
            "page_number": 1,
            "source_file": "test_manual.pdf",
            "content": "DocuMind AI is an advanced enterprise knowledge assistant designed for hybrid retrieval-augmented generation. It utilizes dense vectors and sparse keyword indexes to deliver precise answers with citations."
        }
    ]
    
    # Run chunker with small limits for testing
    chunks = create_parent_child_chunks(
        pages=sample_pages, 
        parent_chunk_size=100, 
        child_chunk_size=40
    )
    
    # Assertions
    assert len(chunks) > 0, "Chunks list should not be empty."
    
    first_chunk = chunks[0]
    assert "child_id" in first_chunk
    assert "parent_id" in first_chunk
    assert "parent_text" in first_chunk
    assert "child_text" in first_chunk
    assert first_chunk["metadata"]["source_file"] == "test_manual.pdf"
    assert first_chunk["metadata"]["page_number"] == 1
    assert len(first_chunk["child_text"]) <= 50 # Respects child limit roughly