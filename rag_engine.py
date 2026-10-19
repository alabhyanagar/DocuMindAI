"""
rag_engine.py - Cross-Encoder Re-Ranking & Gemini Grounded Generation for DocuMind AI
"""

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from retriever import hybrid_search

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

client = genai.Client(api_key=api_key)

# Initialize Cross-Encoder for high-precision re-ranking
print("Loading Cross-Encoder re-ranker model...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

SYSTEM_INSTRUCTION = """
You are DocuMind AI, an elite enterprise knowledge assistant.
Your task is to answer user queries strictly using the provided context snippets from enterprise documents.

RULES FOR GENERATION:
1. Grounding: Every fact or claim in your answer must be directly supported by the provided context. Do not invent or extrapolate outside the text.
2. Citations: Always include inline citations referencing the source file and page number (e.g.,) for every key assertion.
3. Fallback: If the provided context does not contain the answer, state clearly: "I cannot find the answer in the provided documents."
"""

def query_rag(user_query: str) -> str:
    """
    Complete RAG Pipeline:
    1. Hybrid Search (Vector + BM25 + RRF) -> Gets top candidate chunks.
    2. Cross-Encoder Re-Ranking -> Jointly evaluates query and chunks for precision.
    3. Context Assembly -> Collects unique Parent Chunks for full context.
    4. LLM Generation -> Sends prompt & context to Gemini with citations.
    """
    print(f"Executing Hybrid Search for query: '{user_query}'")
    
    # 1. Get top candidates from Hybrid Search
    candidates = hybrid_search(user_query, n_results=10)
    if not candidates:
        return "No relevant information found in the document repository."
        
    # 2. Re-rank candidates using Cross-Encoder
    pairs = [(user_query, item["document"]) for item in candidates]
    scores = cross_encoder.predict(pairs)
    
    # Attach scores and sort descending
    for i, item in enumerate(candidates):
        item["relevance_score"] = float(scores[i])
        
    reranked_candidates = sorted(candidates, key=lambda x: x["relevance_score"], reverse=True)
    
    # Take top 4 highest-scoring re-ranked chunks
    top_chunks = reranked_candidates[:4]
    
    # 3. Assemble unique parent texts and metadata for context window
    context_blocks = []
    seen_parents = set()
    
    for chunk in top_chunks:
        meta = chunk["metadata"]
        parent_id = meta.get("parent_id")
        
        if parent_id not in seen_parents:
            seen_parents.add(parent_id)
            source = meta.get("source_file", "Unknown")
            page = meta.get("page_number", "N/A")
            parent_text = meta.get("parent_text", chunk["document"])
            
            block = f"--- Document: {source} (Page {page}) ---\n{parent_text}\n"
            context_blocks.append(block)
            
    combined_context = "\n".join(context_blocks)
    
    # 4. Generate answer using Gemini
    prompt = f"""
    Context Information from Enterprise Documents:
    {combined_context}
    
    User Query: {user_query}
    
    Please provide a detailed, accurate, and professional answer with inline citations.
    """
    
    response = client.models.generate_content(
        model='gemini-3.5-flash-lite',
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2, # Low temperature for factual accuracy
        )
    )
    
    return response.text