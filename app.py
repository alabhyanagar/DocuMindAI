"""
app.py - Streamlit Frontend Dashboard for DocuMind AI
Provides drag-and-drop document upload and an interactive RAG chat interface.
"""

import streamlit as st
import requests

# FastAPI Backend URL
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #4F46E5; }
    .sub-title { font-size: 1rem; color: #6B7280; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! 👋 I am **DocuMind AI**, your enterprise knowledge assistant. Upload a PDF document in the sidebar, and ask me anything about its contents!"
        }
    ]

# ---------------------------------------------------------------------------
# SIDEBAR - DOCUMENT UPLOAD & MANAGEMENT
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📂 Document Repository")
    st.caption("Upload enterprise PDFs to ingest into ChromaDB.")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("🚀 Ingest Document", use_container_width=True):
            with st.spinner(f"Parsing & indexing '{uploaded_file.name}'..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{API_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ Success! Indexed {data.get('chunks_indexed')} chunks.")
                    else:
                        st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"⚠️ Connection error: Could not connect to FastAPI backend. Make sure uvicorn is running.")

    st.divider()
    st.markdown("### 🛠️ System Architecture")
    st.markdown("""
    - **Layout Parsing:** PyMuPDF & pdfplumber
    - **Chunking:** Parent-Child Chunking
    - **Retrieval:** Hybrid Search (ChromaDB + BM25 + RRF)
    - **Re-Ranking:** Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
    - **Generation:** Google Gemini 2.5 Flash
    """)
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat history cleared! How else can I help you?"
            }
        ]
        st.rerun()

# ---------------------------------------------------------------------------
# MAIN CHAT INTERFACE
# ---------------------------------------------------------------------------
st.markdown("<div class='main-title'>🧠 DocuMind AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Advanced Hybrid RAG Enterprise Knowledge Assistant with Source Citations</div>", unsafe_allow_html=True)

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input Box
user_query = st.chat_input("Ask a question about your uploaded documents...")

if user_query:
    # 1. Append and display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Query FastAPI Backend
    with st.chat_message("assistant"):
        with st.spinner("Searching documents, re-ranking chunks, and generating answer..."):
            try:
                response = requests.post(f"{API_URL}/query", json={"query": user_query})
                
                if response.status_code == 200:
                    answer_text = response.json().get("answer", "No answer returned.")
                else:
                    answer_text = f"⚠️ Server Error: {response.json().get('detail', 'Unknown error')}"
            except Exception as e:
                answer_text = f"⚠️ Connection Error: Unable to reach FastAPI backend at `{API_URL}`. Ensure `uvicorn main:app --reload` is running."
            
            st.markdown(answer_text)
            st.session_state.messages.append({"role": "assistant", "content": answer_text})