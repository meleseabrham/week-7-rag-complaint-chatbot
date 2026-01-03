import streamlit as st
import sys
import os

# Add src to path if needed (though current dir is project root)
sys.path.append(os.path.join(os.getcwd(), 'src'))

from rag_pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="CrediTrust - Consumer Complaint Analyst",
    page_icon="🤖",
    layout="centered"
)

# Header
st.title("🤖 CrediTrust Complaint Assistant")
st.markdown("""
Welcome to the internal CrediTrust Financial complaint analysis portal. 
Ask questions about consumer complaints to get grounded, data-driven answers.
""")

# Initialize RAG Pipeline (Cached as a resource)
@st.cache_resource
def load_rag():
    try:
        return RAGPipeline()
    except Exception as e:
        st.error(f"Failed to load RAG Pipeline: {e}")
        return None

rag = load_rag()

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar - Information
with st.sidebar:
    st.header("Project Info")
    st.info("""
    This RAG system uses:
    - **Vector Store**: FAISS
    - **Embeddings**: all-MiniLM-L6-v2
    - **LLM**: flan-t5-small (Local)
    """)
    
    # Custom CSS for the red clear button
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #FF4B4B;
        color: white;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #FF2B2B;
        color: white;
        font-weight: bold;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Sources"):
                for src in message["sources"]:
                    st.write(f"**Complaint ID**: {src['complaint_id']}")
                    st.write(f"**Snippet**: {src['content']}")
                    st.divider()

# Input area
if rag:
    if prompt := st.chat_input("Ask a question about customer complaints..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analyzing complaints..."):
                response = rag.answer_question(prompt)
                full_response = response["result"]
                sources = response["source_documents"]
            
            st.markdown(full_response)
            
            # Format sources for metadata storage
            formatted_sources = []
            for doc in sources:
                formatted_sources.append({
                    "complaint_id": doc.metadata.get("complaint_id", "N/A"),
                    "content": doc.page_content
                })
            
            # Display sources in UI immediately
            with st.expander("View Sources"):
                for src in formatted_sources:
                    st.write(f"**Complaint ID**: {src['complaint_id']}")
                    st.write(f"**Snippet**: {src['content']}")
                    st.divider()
            
            # Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "sources": formatted_sources
            })
else:
    st.warning("⚠️ RAG Pipeline is not initialized. Please ensure the vector store exists.")
