"""
retriever.py - Hybrid Search Engine (Vector Search + BM25 Keyword Search + RRF) for DocuMind AI
"""

import numpy as np
from rank_bm25 import BM25Okapi
from vector_db import collection, query_vector_store

def get_all_chunks_from_db():
    """Fetches all indexed chunks from ChromaDB for BM25 keyword index building."""
    data = collection.get(include=["documents", "metadatas"])
    ids = data["ids"]
    documents = data["documents"]
    metadatas = data["metadatas"]
    
    chunks = []
    for i in range(len(ids)):
        chunks.append({
            "id": ids[i],
            "document": documents[i],
            "metadata": metadatas[i]
        })
    return chunks

def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    """
    Combines two ranked lists using Reciprocal Rank Fusion (RRF).
    RRF Score = 1 / (k + rank)
    """
    fusion_scores = {}
    
    # Helper to process ranked list items
    def accumulate_scores(results_list):
        for rank, item in enumerate(results_list):
            item_id = item["id"]
            if item_id not in fusion_scores:
                fusion_scores[item_id] = {
                    "score": 0.0,
                    "document": item["document"],
                    "metadata": item["metadata"]
                }
            # RRF formula
            fusion_scores[item_id]["score"] += 1.0 / (k + (rank + 1))

    accumulate_scores(vector_results)
    accumulate_scores(bm25_results)
    
    # Sort by fused score descending
    sorted_fusion = sorted(fusion_scores.values(), key=lambda x: x["score"], reverse=True)
    return sorted_fusion

def hybrid_search(query: str, n_results: int = 5) -> list[dict]:
    """
    Executes Hybrid Search:
    1. Dense Vector Search via ChromaDB.
    2. Sparse Keyword Search via BM25Okapi.
    3. Combines results using Reciprocal Rank Fusion (RRF).
    """
    # 1. Vector Search execution
    vec_raw = query_vector_store(query_text=query, n_results=n_results * 2)
    vector_results = []
    if vec_raw and vec_raw["ids"] and vec_raw["ids"][0]:
        for i in range(len(vec_raw["ids"][0])):
            vector_results.append({
                "id": vec_raw["ids"][0][i],
                "document": vec_raw["documents"][0][i],
                "metadata": vec_raw["metadatas"][0][i]
            })

    # 2. BM25 Keyword Search execution
    all_chunks = get_all_chunks_from_db()
    if not all_chunks:
        return vector_results # Fallback if DB is empty
        
    tokenized_corpus = [chunk["document"].lower().split(" ") for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = query.lower().split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Top indices sorted by BM25 score
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:n_results * 2]
    
    bm25_results = []
    for idx in top_bm25_indices:
        if bm25_scores[idx] > 0: # Only include relevant keyword matches
            chunk = all_chunks[idx]
            bm25_results.append({
                "id": chunk["id"],
                "document": chunk["document"],
                "metadata": chunk["metadata"]
            })

    # 3. Merge using Reciprocal Rank Fusion
    fused_results = reciprocal_rank_fusion(vector_results, bm25_results)
    
    # Return top N results
    return fused_results[:n_results]