import streamlit as st
import sys
import os
import time
import pandas as pd
import plotly.express as px
from PIL import Image

# Robust path handling for src
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from rag_pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="CrediTrust - Advanced RAG Intelligence",
    page_icon="🛡️",
    layout="wide"
)

# Constants & Paths
LOGO_PATH = r"C:\Users\dell\.gemini\antigravity\brain\8cb5df7c-9a97-407d-af85-f88c33c3445d\creditrust_logo_1767447836333.png"
DATA_PATH = "data/filtered_complaints.csv"

# Custom CSS for polished UI
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stChatFloatingInputContainer {
        padding-bottom: 20px;
    }
    .source-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #ffffff;
        border-left: 6px solid #1E3A8A;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton>button {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize RAG Pipeline
@st.cache_resource
def load_rag():
    return RAGPipeline()

@st.cache_data
def load_stats():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        return df['Product'].value_counts()
    return None

rag = load_rag()
stats = load_stats()

# Sidebar: Branding and Controls
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=150)
    else:
        st.title("🛡️ CrediTrust")
    
    st.markdown("---")
    
    # Tabs in sidebar for organization
    sb_tab1, sb_tab2 = st.tabs(["⚙️ Controls", "📊 Insights"])
    
    with sb_tab1:
        st.subheader("Retrieval Intelligence")
        top_k = st.slider("Discovery Pool (K)", min_value=5, max_value=50, value=5, help="Number of complaints to initially retrieve before re-ranking.")
        st.caption("Lower K is faster; Higher K + Re-ranking is more accurate.")
        
        st.subheader("Model Status")
        if rag:
            st.success("🤖 Generative AI: Online")
            st.success("🗄️ Vector Index: Active")
            st.success("⚖️ Re-ranker: Active")
        else:
            st.error("RAG Pipeline Offline")

        st.markdown("---")
        st.subheader("Conversation Management")
        
        if st.button("📊 Summarize Current Session", use_container_width=True):
            if st.session_state.messages:
                with st.spinner("Synthesizing session insights..."):
                    history_summary = ""
                    for m in st.session_state.messages:
                        history_summary += f"{m['role']}: {m['content']}\n"
                    
                    summary_prompt = f"Summarize the key financial issues and consumer concerns discussed in this conversation so far. Focus on patterns and risks.\n\nConversation:\n{history_summary}\n\nSummary:"
                    summary_res = rag.pipe(summary_prompt, max_new_tokens=150)
                    st.info(summary_res[0]['generated_text'])
            else:
                st.warning("Start a conversation first!")

        # Red button with bold hover
        st.markdown("""
        <style>
        div.stButton > button.clear-btn {
            background-color: #FF4B4B;
            color: white;
            border: none;
            width: 100%;
        }
        div.stButton > button.clear-btn:hover {
            background-color: #FF2B2B;
            color: white;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if st.button("Reset Conversation", key="clear_chat", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.rerun()

    with sb_tab2:
        st.subheader("Complaint Distribution")
        if stats is not None:
            fig = px.pie(values=stats.values, names=stats.index, hole=.4, 
                         color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Distribution of 12,000 sampled complaints across core products.")
        else:
            st.warning("Data stats unavailable.")

# Main UI
st.title("🛡️ CrediTrust Advanced Analyst")
st.caption("Grounded Financial Intelligence | CFPB Complaint Dataset")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("🔍 Inspection: Root Context"):
                for i, src in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1} | Complaint ID: {src['id']}**")
                    st.info(src['content'])

# Input logic
if rag:
    if prompt := st.chat_input("Ask about complaint trends, specific products, or issues..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Contextual Memory (Last 4 turns)
        history_text = ""
        for m in st.session_state.messages[-5:-1]:
            role = "Human" if m["role"] == "user" else "Assistant"
            history_text += f"{role}: {m['content']}\n"

        # Assistant response with STREAMING
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Decoding records..."):
                streamer, docs = rag.stream_answer(prompt, history=history_text, k=top_k)
            
            for new_text in streamer:
                full_response += new_text
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
            
            formatted_sources = [{"id": d.metadata.get("complaint_id", "N/A"), "content": d.page_content} for d in docs]
            
            with st.expander("🔍 Inspection: Root Context"):
                for i, src in enumerate(formatted_sources):
                    st.markdown(f"**Source {i+1} | Complaint ID: {src['id']}**")
                    st.info(src['content'])
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": formatted_sources
            })

    # Export Button at the bottom of the chat
    if st.session_state.messages:
        chat_text = "CrediTrust Chat Export\n" + "="*30 + "\n\n"
        for m in st.session_state.messages:
            chat_text += f"[{m['role'].upper()}]: {m['content']}\n\n"
        
        st.sidebar.download_button(
            label="📄 Export Conversation",
            data=chat_text,
            file_name=f"creditrust_chat_{int(time.time())}.txt",
            mime="text/plain",
            use_container_width=True
        )
else:
    st.warning("Credentializing RAG system... please wait.")
