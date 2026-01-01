# 🤖 Consumer Complaint RAG Chatbot

An intelligent, Retrieval-Augmented Generation (RAG) powered chatbot designed to navigate and answer questions about the CFPB (Consumer Financial Protection Bureau) database. This project leverages natural language processing to extract insights from thousands of consumer complaints spanning credit cards, personal loans, savings accounts, and more.

---

## 🚀 Features

-   **Intelligent Data Preprocessing**: Automated cleaning, filtering, and stratified sampling of a massive complaint database (~9M+ records).
-   **Recursive Chunking**: High-context splitting of long consumer narratives to ensure semantic integrity during retrieval.
-   **Semantic Search**: Utilizes `all-MiniLM-L6-v2` embeddings for fast and accurate similarity matching.
-   **FAISS Vector Store**: Locally persisted vector database for lightning-fast retrieval of relevant complaint metadata and narratives.
-   **CI/CD Integrated**: Robust testing pipeline using GitHub Actions to ensure code quality and dependency reliability.

---

## 🛠️ Tech Stack

-   **Core Framework**: [LangChain](https://www.langchain.com/)
-   **Embeddings**: [Hugging Face Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
-   **Vector Search**: [FAISS](https://github.com/facebookresearch/faiss)
-   **Data Wrangling**: Pandas, Scikit-learn
-   **Environment**: Python 3.10+
-   **Frontend**: Gradio / Streamlit (In Development)

---

## 📂 Project Structure

```bash
rag-complaint-chatbot/
├── .github/workflows/    # CI/CD pipelines (Unit Tests)
├── data/                 # Raw and filtered CSV datasets (Git ignored)
├── notebooks/            # EDA and experimental research
├── src/                  # Core logic
│   ├── preprocessing.py  # Data cleaning and filtering
│   └── indexing.py       # Chunking and Vector Store building
├── tests/                # Pytest unit tests
├── vector_store/         # Persisted FAISS index files
├── app.py                # Main chatbot application
└── requirements.txt      # Project dependencies
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/meleseabrham/week-7-rag-complaint-chatbot.git
   cd week-7-rag-complaint-chatbot
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Usage

### 1. Data Preprocessing
Clean and filter the raw CFPB dataset to focus on relevant products and non-empty narratives:
```bash
python src/preprocessing.py
```

### 2. Building the Vector Index
Perform stratified sampling (12,000 records), chunk narratives, and build the FAISS vector index:
```bash
python src/indexing.py
```

### 3. Running the Chatbot
Launch the RAG-enabled interface:
```bash
python app.py
```

---

## 🧪 Testing
Run unit tests to verify data cleaning and indexing logic:
```bash
pytest
```

---

## 🤝 Contribution
This project follows a feature-branch workflow. Please ensure all code passes CI tests before submitting a PR.
- **`task-1`**: EDA and Preprocessing
- **`task-2`**: Indexing and Embeddings
