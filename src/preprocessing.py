import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load the complaint dataset."""
    return pd.read_csv(file_path, low_memory=False)

def filter_data(df):
    """Filter dataset by products and non-empty narratives."""
    target_products = [
        'Credit card', 
        'Personal loan', 
        'Buy Now, Pay Later (BNPL)', 
        'Savings account', 
        'Money transfers'
    ]
    # Filter products
    df_filtered = df[df['Product'].isin(target_products)].copy()
    
    # Remove records with empty Consumer complaint narrative
    df_filtered = df_filtered.dropna(subset=['Consumer complaint narrative'])
    df_filtered = df_filtered[df_filtered['Consumer complaint narrative'].str.strip() != '']
    
    return df_filtered

def clean_text(text):
    """Clean the text narrative."""
    if not isinstance(text, str):
        return ""
    
    # Lowercasing
    text = text.lower()
    
    # Remove boilerplate snippets (example)
    boilerplate = [
        r"i am writing to file a complaint",
        r"to whom it may concern",
        r"dear cfpb",
        r"thank you for your time"
    ]
    for pattern in boilerplate:
        text = re.sub(pattern, "", text)
    
    # Remove special characters but keep some punctuation for context if needed?
    # For RAG, keeping some structure is often good, but let's remove excessive symbols.
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_pipeline(input_path, output_path):
    """Full pipeline for Task 1."""
    print("Loading data...")
    df = load_data(input_path)
    
    print("Initial Analysis...")
    total_complaints = len(df)
    complaints_with_narrative = df['Consumer complaint narrative'].notnull().sum()
    print(f"Total complaints: {total_complaints}")
    print(f"Complaints with narrative: {complaints_with_narrative}")
    
    print("Filtering data...")
    df_filtered = filter_data(df)
    print(f"Filtered complaints (Target products + non-empty narrative): {len(df_filtered)}")
    
    print("Cleaning text...")
    df_filtered['cleaned_narrative'] = df_filtered['Consumer complaint narrative'].apply(clean_text)
    
    print(f"Saving to {output_path}...")
    df_filtered.to_csv(output_path, index=False)
    print("Done!")
    return df_filtered

if __name__ == "__main__":
    # For testing or direct script run
    preprocess_pipeline('data/complaints.csv', 'data/filtered_complaints.csv')
