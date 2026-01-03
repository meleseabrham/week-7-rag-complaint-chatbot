import streamlit as st
import sys
import os
import time

# Add src to path if needed
sys.path.append(os.path.join(os.getcwd(), 'src'))

from rag_pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="CrediTrust Advanced Analyst",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for polished UI
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .stChatFloatingInputContainer {
        padding-bottom: 20px;
    }
    .source-card {
        padding: 10px;
        border-radius: 5px;
        background-color: #f9f9f9;
        border-left: 5px solid #FF4B4B;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize RAG Pipeline
@st.cache_resource
def load_rag():
    return RAGPipeline()

rag = load_rag()

# Sidebar: Advanced Controls and Dashboard
with st.sidebar:
    st.title("🛡️ Dashboard")
    st.markdown("---")
    
    st.subheader("Retrieval Settings")
    top_k = st.slider("Retrieval K (Context Chunks)", min_value=1, max_value=10, value=5)
    
    st.subheader("Model Status")
    if rag:
        st.success("Internal LLM: flan-t5-small (Online)")
        st.success("Vector Store: FAISS (Loaded)")
    else:
        st.error("RAG Pipeline Offline")

    st.markdown("---")
    st.subheader("System Actions")
    # Custom CSS for the red clear button (Bold on hover as requested)
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #FF4B4B;
        color: white;
        border: none;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #FF2B2B;
        color: white;
        font-weight: bold;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Main UI
st.title("🛡️ CrediTrust Advanced Analyst")
st.caption("AI-Powered Financial Complaint Intelligence")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Inspection: Retrieval Sources"):
                for i, src in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1} (ID: {src['id']})**")
                    st.info(src['content'])

# Input logic
if rag:
    if prompt := st.chat_input("Query the complaint database..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Generate history string for context
        history_text = ""
        for m in st.session_state.messages[-4:-1]: # Last 3 turns
            role = "Human" if m["role"] == "user" else "Assistant"
            history_text += f"{role}: {m['content']}\n"

        # Assistant response with STREAMING
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Retrieving and generating..."):
                streamer, docs = rag.stream_answer(prompt, history=history_text, k=top_k)
            
            # Streaming loop
            for new_text in streamer:
                full_response += new_text
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
            
            # Format sources
            formatted_sources = [{"id": d.metadata.get("complaint_id", "N/A"), "content": d.page_content} for d in docs]
            
            # Display sources
            with st.expander("Inspection: Retrieval Sources"):
                for i, src in enumerate(formatted_sources):
                    st.markdown(f"**Source {i+1} (ID: {src['id']})**")
                    st.info(src['content'])
            
            # Store history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": formatted_sources
            })
else:
    st.warning("Please wait for the RAG system to initialize...")
