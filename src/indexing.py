import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os

def load_and_sample(file_path, sample_size=12000):
    """Load filtered complaints and perform stratified sampling."""
    print("Loading filtered data...")
    df = pd.read_csv(file_path)
    
    print(f"Total available records: {len(df)}")
    
    # Stratified sampling
    # We want sample_size total. We use train_test_split as a convenient way to stratify.
    _, df_sample = train_test_split(
        df, 
        test_size=sample_size / len(df), 
        stratify=df['Product'], 
        random_state=42
    )
    
    print(f"Sampled {len(df_sample)} records across products:")
    print(df_sample['Product'].value_counts())
    
    return df_sample

def create_chunks(df_sample, chunk_size=500, chunk_overlap=50):
    """Split narratives into chunks using LangChain."""
    print(f"Creating chunks (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    documents = []
    for _, row in df_sample.iterrows():
        # metadata includes original ID and product
        metadata = {
            "complaint_id": str(row.get('Complaint ID', 'N/A')),
            "product": row['Product'],
            "sub_product": str(row.get('Sub-product', 'N/A'))
        }
        
        # Use the cleaned_narrative column from Task 1
        narrative = row['cleaned_narrative']
        if pd.isna(narrative) or narrative == "":
            continue
            
        chunks = text_splitter.split_text(narrative)
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata=metadata))
            
    print(f"Generated {len(documents)} document chunks.")
    return documents

def build_vector_store(documents, store_path="vector_store/faiss_index"):
    """Generate embeddings and build FAISS index."""
    print("Initializing embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Building FAISS index (this may take a few minutes)...")
    vector_store = FAISS.from_documents(documents, embeddings)
    
    print(f"Saving vector store to {store_path}...")
    # Create directory if not exists
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    vector_store.save_local(store_path)
    print("Vector store saved successfully.")
    return vector_store

if __name__ == "__main__":
    input_file = "data/filtered_complaints.csv"
    output_store = "vector_store/faiss_index"
    
    if os.path.exists(input_file):
        df_sample = load_and_sample(input_file)
        docs = create_chunks(df_sample)
        build_vector_store(docs, output_store)
    else:
        print(f"Error: {input_file} not found. Run Task 1 first.")
