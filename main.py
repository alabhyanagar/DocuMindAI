"""
main.py - FastAPI REST Backend for DocuMind AI
Exposes endpoints for document upload/ingestion and hybrid RAG querying.
"""

import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our ingestion and RAG engines
from vector_db import ingest_pdf_to_vector_db
from rag_engine import query_rag

app = FastAPI(
    title="DocuMind AI API",
    description="Enterprise Knowledge Assistant with Hybrid RAG, Cross-Encoder Re-Ranking, and Gemini Generation.",
    version="1.0.0"
)

# Enable CORS for frontend integration (Streamlit / React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("data")
UPLOAD_DIR.mkdir(exist_ok=True)

# Pydantic Schemas for Request/Response validation
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    answer: str

@app.get("/")
def read_root():
    return {"status": "online", "service": "DocuMind AI Backend is running successfully!"}

@app.post("/upload", summary="Upload and Ingest PDF Document")
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a PDF document, saves it to the data directory, 
    and triggers the layout-aware parsing and vector embedding pipeline.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")
        
    file_path = UPLOAD_DIR / file.filename
    
    try:
        # Save file locally
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        # Run ingestion pipeline
        chunks_indexed = ingest_pdf_to_vector_db(str(file_path))
        
        return {
            "success": True,
            "filename": file.filename,
            "chunks_indexed": chunks_indexed,
            "message": f"Successfully parsed and indexed {chunks_indexed} chunks into ChromaDB."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/query", response_model=QueryResponse, summary="Query Enterprise Knowledge Base")
def query_knowledge_base(request: QueryRequest):
    """
    Executes Hybrid Search, Cross-Encoder Re-Ranking, and Gemini Generation 
    to answer user questions with citations.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    try:
        answer = query_rag(request.query)
        return QueryResponse(
            query=request.query,
            answer=answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query execution failed: {str(e)}")