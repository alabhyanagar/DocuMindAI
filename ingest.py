"""
ingest.py - Layout-aware PDF Parser & Parent-Child Chunker for DocuMind AI
"""

import os
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def extract_text_and_tables_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extracts text and markdown tables page-by-page from a complex PDF.
    Returns a list of dictionaries containing page content and metadata.
    """
    extracted_pages = []
    
    # 1. Extract structural text using PyMuPDF
    doc = fitz.open(pdf_path)
    
    # 2. Extract tables using pdfplumber for layout accuracy
    with pdfplumber.open(pdf_path) as plumber_doc:
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            
            # Check for tables on this page using pdfplumber
            plumber_page = plumber_doc.pages[page_num]
            tables = plumber_page.extract_tables()
            
            markdown_tables = []
            if tables:
                for table in tables:
                    # Convert table rows into clean markdown format
                    md_table = "\n".join([" | ".join([str(cell) if cell else "" for cell in row]) for row in table])
                    markdown_tables.append(f"\n--- Table Data ---\n{md_table}\n------------------\n")
            
            combined_content = text + "\n".join(markdown_tables)
            
            extracted_pages.append({
                "page_number": page_num + 1,
                "source_file": Path(pdf_path).name,
                "content": combined_content.strip()
            })
            
    return extracted_pages

def create_parent_child_chunks(pages: list[dict], parent_chunk_size: int = 1000, child_chunk_size: int = 300) -> list[dict]:
    """
    Implements Parent-Child chunking:
    - Parent chunks: Large context windows (e.g., 1000 chars) passed to the LLM.
    - Child chunks: Smaller segments (e.g., 300 chars) indexed in ChromaDB for precise vector search.
    """
    structured_chunks = []
    
    for page in pages:
        full_text = page["content"]
        source = page["source_file"]
        page_no = page["page_number"]
        
        # Simple sliding window splitter for illustration (can be enhanced with LangChain splitters)
        parent_tokens = [full_text[i:i+parent_chunk_size] for i in range(0, len(full_text), parent_chunk_size)]
        
        for p_idx, parent_text in enumerate(parent_tokens):
            parent_id = f"{source}_p{page_no}_par{p_idx}"
            
            # Sub-divide parent into smaller child chunks
            child_tokens = [parent_text[j:j+child_chunk_size] for j in range(0, len(parent_text), child_chunk_size)]
            
            for c_idx, child_text in enumerate(child_tokens):
                if len(child_text.strip()) > 30:  # Ignore tiny empty noise snippets
                    structured_chunks.append({
                        "child_id": f"{parent_id}_c{c_idx}",
                        "parent_id": parent_id,
                        "parent_text": parent_text,
                        "child_text": child_text,
                        "metadata": {
                            "source_file": source,
                            "page_number": page_no
                        }
                    })
                    
    return structured_chunks

if __name__ == "__main__":
    print("DocuMind AI Ingestion Module Initialized.")
    print("Place sample PDFs inside the 'data/' directory to begin parsing.")