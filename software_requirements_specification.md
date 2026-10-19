# Software Requirements Specification (SRS)
## Document Title: DocuMind AI — Enterprise Knowledge Assistant
**Document Version:** 1.0.0  
**Date:** October 19, 2026  
**Project Lead:** AI/ML Engineering Team  
**Status:** Approved for Implementation  

---

## Executive Summary

**DocuMind AI** is an enterprise-grade document intelligence platform engineered to parse, index, search, and synthesize insights from unstructured multi-format business documents (PDFs, Markdown tables, legal filings, technical specifications, and financial reports). Standard Retrieval-Augmented Generation (RAG) systems rely almost exclusively on dense vector embeddings, leading to retrieval failures when handling exact keyword terms, tabular structures, or multi-page context dependencies.

DocuMind AI solves these enterprise limitations by implementing a multi-stage hybrid search engine (Dense Vector + BM25 Sparse Keyword Search), Reciprocal Rank Fusion (RRF), Cross-Encoder Re-Ranking, Parent-Child Chunking, and deterministic Safety Guardrails. The system is exposed as a decoupled, asynchronous FastAPI microservice and containerized using Docker for production deployment.

---

## 1. Introduction

### 1.1 Purpose
This Software Requirements Specification (SRS) defines the complete functional, technical, operational, and non-functional requirements for DocuMind AI. This document serves as the single source of truth for software engineers, ML architects, and project stakeholders during the 4-week development cycle.

### 1.2 Scope
DocuMind AI is designed to automate complex document analysis. The system provides:
1. **Layout-Aware Ingestion:** Parsing complex PDFs while preserving structured Markdown tables and section headers.
2. **Parent-Child Chunking:** Linking small, highly retrievable vector chunks to larger semantic parent text blocks.
3. **Hybrid Search Pipeline:** Parallel evaluation using ChromaDB (vector embeddings) and `rank_bm25` (sparse keyword index).
4. **Cross-Encoder Re-Ranking:** Post-retrieval re-scoring of context chunks using transformer cross-encoders to eliminate irrelevance.
5. **Grounded Generation & Citations:** LLM synthesis with strict inline document/page-number attribution.
6. **Input/Output Guardrails:** Automated mitigation of prompt injections, system prompt leaks, and ungrounded hallucinations.

### 1.3 Definitions, Acronyms, and Abbreviations
* **RAG:** Retrieval-Augmented Generation
* **BM25:** Best Matching 25 (A sparse probabilistic term-frequency ranking algorithm)
* **RRF:** Reciprocal Rank Fusion
* **VDB:** Vector Database (ChromaDB)
* **LLM:** Large Language Model (Google Gemini API / OpenAI API)
* **XAI:** Explainable Artificial Intelligence
* **PEFT/LoRA:** Parameter-Efficient Fine-Tuning / Low-Rank Adaptation

---

## 2. System Architecture & High-Level Design

### 2.1 System Context Diagram

```
+-----------------------------------------------------------------------------------+
|                                  USER INTERFACE                                   |
|                        (Streamlit / React Dashboard)                              |
+------------------------------------------+----------------------------------------+
                                           | HTTP REST API Requests
                                           v
+-----------------------------------------------------------------------------------+
|                                FASTAPI BACKEND SERVER                             |
|                                                                                   |
|  +--------------------+     +---------------------+     +----------------------+  |
|  | Ingestion Engine   |     | Hybrid Search       |     | Re-Ranking & Guard   |  |
|  | (Layout Parser)    | --> | (ChromaDB + BM25)   | --> | (Cross-Encoder / LLM)|  |
|  +--------------------+     +---------------------+     +----------------------+  |
+------------------------------------------+----------------------------------------+
                                           | Persistent Storage
                                           v
+-----------------------------------------------------------------------------------+
|                         VECTOR & DOCUMENT DATABASE (ChromaDB)                     |
+-----------------------------------------------------------------------------------+
```

### 2.2 Data Pipeline Execution Flow
1. **Document Ingestion:** The client uploads a PDF/Doc file via the `/api/v1/documents/upload` endpoint.
2. **Text & Table Extraction:** `pdfplumber`/`PyMuPDF` parses the document layout, converting tables to Markdown format.
3. **Dual Chunking Strategy:**
   * **Parent Chunks:** ~1,000 tokens (provides broader background context to the LLM).
   * **Child Chunks:** ~200 tokens (stored in ChromaDB for high-precision vector matches).
4. **Hybrid Search Execution:**
   * Dense vector search returns top $K$ items from ChromaDB.
   * BM25 sparse search returns top $K$ items from the keyword index.
5. **Reciprocal Rank Fusion (RRF):** Merges vector and sparse candidate lists using the formula:
   $$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
   where $k = 60$ and $r_m(d)$ represents the document rank in model $m$.
6. **Cross-Encoder Re-Ranking:** A cross-encoder model (`ms-marco-MiniLM-L-6-v2`) re-evaluates top candidates against the query and filters down to the top 3–5 golden chunks.
7. **Prompt Construction & Generation:** Golden chunks are fed into Gemini 2.5 Flash along with guardrail parameters to produce a grounded response with source citations.

---

## 3. Technology Stack Specification

| Component Layer | Selected Technology | Version / Model | Justification |
| :--- | :--- | :--- | :--- |
| **Programming Language** | Python | 3.10+ | Industry standard for AI/ML pipelines and NLP frameworks. |
| **API Framework** | FastAPI | >= 0.100.0 | High-performance asynchronous REST API with auto Pydantic schema validation. |
| **Document Parsing** | PyMuPDF / `pdfplumber` | Latest stable | Superior layout retention and table-to-markdown extraction capability. |
| **Dense Vector Database** | ChromaDB | >= 0.4.0 | Embedded, low-latency persistent vector database with metadata filtering. |
| **Sparse Search Engine** | `rank_bm25` | Latest stable | Fast Python implementation of BM25 algorithm for exact-match retrieval. |
| **Embeddings Model** | HuggingFace Transformers | `all-MiniLM-L6-v2` | Fast, high-density 384-dimensional vector embeddings. |
| **Re-Ranking Engine** | Cross-Encoder | `ms-marco-MiniLM-L-6-v2` | High-precision joint query-document relevance scoring. |
| **LLM Provider** | Google Gemini API | `gemini-2.5-flash` | State-of-the-art long context window, fast inference, and robust system instruction support. |
| **Frontend Framework** | Streamlit | Latest | Rapid prototyping for interactive knowledge exploration and document viewing. |
| **Deployment Engine** | Docker & Docker Compose | Engine v24+ | Portable multi-container microservice packaging. |

---

## 4. System Features & Detailed Requirements

### 4.1 Ingestion & Indexing Engine (Module 1)
* **REQ-ING-1:** The system shall accept PDF, TXT, and Markdown file formats.
* **REQ-ING-2:** The document parser must convert embedded tables into clean Markdown representations (`| header1 | header2 |`) to retain structural layout during vectorization.
* **REQ-ING-3:** The chunking pipeline shall implement Parent-Child linking: child chunks (200 tokens) must store a `parent_id` reference pointing to their parent chunk (1,000 tokens) in metadata.
* **REQ-ING-4:** Metadata stored alongside vector embeddings must include `filename`, `page_number`, `chunk_id`, `parent_id`, and `created_timestamp`.

### 4.2 Hybrid Search & Re-Ranking Engine (Module 2)
* **REQ-RET-1:** The system shall execute vector similarity search and sparse BM25 keyword search concurrently upon receiving a query.
* **REQ-RET-2:** Search candidate lists must be combined using Reciprocal Rank Fusion (RRF) with a default smoothing parameter $k = 60$.
* **REQ-RET-3:** The merged RRF candidate set (top 20-30 chunks) must pass through a Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`) to re-score candidate relevance.
* **REQ-RET-4:** Only the top 3 to 5 re-ranked golden chunks shall be included in the context payload sent to the LLM.

### 4.3 REST API Interface (Module 3)
* **REQ-API-1:** `POST /api/v1/documents/upload` — Accepts multipart file uploads, processes documents, updates ChromaDB, and builds the BM25 index.
* **REQ-API-2:** `POST /api/v1/query` — Accepts JSON request payload containing `query`, `domain_filter`, and `top_k`; returns streaming or block response with answer and citations.
* **REQ-API-3:** `GET /api/v1/documents` — Lists all indexed documents, total chunks, and indexing status.
* **REQ-API-4:** `DELETE /api/v1/documents/{document_id}` — Deletes specified document chunks from both ChromaDB and BM25 index.

### 4.4 Guardrails & Safety Layer (Module 4)
* **REQ-SEC-1:** **Input Sanitation:** The system shall inspect user queries for prompt injection attacks or instructions attempting to override system behavior.
* **REQ-SEC-2:** **Hallucination Checker:** The final generated output must undergo a strict grounding verification step ensuring facts in the answer map directly back to retrieved context chunks.
* **REQ-SEC-3:** **Out-of-Scope Handling:** If no relevant document context passes the cross-encoder score threshold ($\text{Score} < 0.2$), the system must respond with a standard decline message rather than generating unverified information.

---

## 5. Non-Functional Requirements

### 5.1 Performance & Latency
* **NFR-PERF-1 (Query Latency):** Total end-to-end response generation time (excluding stream delivery) shall not exceed **3.5 seconds** for standard queries.
* **NFR-PERF-2 (Parsing Speed):** The ingestion pipeline shall process standard PDF documents at a minimum speed of **3 pages per second**.

### 5.2 Reliability & Fault Tolerance
* **NFR-REL-1:** If the BM25 keyword index yields zero matches, the system must gracefully fall back to dense vector search without raising an uncaught exception.
* **NFR-REL-2:** Vector database persistence must be maintained on local disk volumes to survive system restarts.

### 5.3 Security
* **NFR-SEC-1:** Environment secrets (API Keys) must never be hardcoded in application source code and must be loaded strictly via `.env` or system environment variables.
* **NFR-SEC-2:** API input parameters must be validated using Pydantic schemas to prevent injection attacks.

---

## 6. Development Lifecycle & Milestone Roadmap

```
+-------------------------------------------------------------------------+
| WEEK 1: Ingestion Engine & Parent-Child Chunking                        |
|  - Layout-aware PDF & Markdown table parsing                            |
|  - ChromaDB vector store initialization & metadata schema setup         |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| WEEK 2: Hybrid Search Engine & Cross-Encoder Re-Ranking                 |
|  - BM25 index integration & Reciprocal Rank Fusion (RRF) logic          |
|  - Cross-Encoder re-scoring pipeline implementation                     |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| WEEK 3: FastAPI Microservice & Guardrails Layer                         |
|  - REST API Endpoints (/upload, /query, /documents)                     |
|  - Pydantic schema validation & Prompt Injection Guardrails             |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| WEEK 4: Streamlit UI Dashboard, Unit Testing & Containerization         |
|  - Interactive frontend with real-time source citation viewer           |
|  - Pytest unit testing suite & Docker Compose deployment packaging      |
+-------------------------------------------------------------------------+
```

---

## 7. Approval & Sign-off

| Role | Name | Signature / Status | Date |
| :--- | :--- | :--- | :--- |
| **Lead AI Engineer** | Alabhya Nagar | **APPROVED** | October 19, 2026 |
| **Solutions Architect** | AI Development Team | **APPROVED** | October 19, 2026 |