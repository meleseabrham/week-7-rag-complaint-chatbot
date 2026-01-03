# Interim Report: Consumer Complaint RAG Chatbot

## 1. Understanding and Defining the Business Objective

### The Core Problem
At **CrediTrust Financial**, internal teams—including Product Managers, Customer Support, and Compliance officers—are currently overwhelmed by a vast influx of unstructured consumer complaints. Manually extracting actionable insights from thousands of records is a slow, error-prone process that often leads to reactive rather than proactive problem-solving.

### Strategic Objective
The objective of this project is to develop a **Retrieval-Augmented Generation (RAG)** powered chatbot. This tool will serve as a strategic asset, allowing non-technical stakeholders to "talk to the data" and extract semantic insights without requiring complex SQL queries or manual transcript reviews.

### Key Performance Indicators (KPIs)
*   **Efficiency**: Reduce trend identification time from days of manual analysis to seconds/minutes.
*   **Accessibility**: Empower non-technical teams with self-service data exploration.
*   **Proactivity**: Enable rapid identification of emerging compliance risks and product friction points to shift from reactive mitigation to proactive strategy.

---

## 2. Discussion of Completed Work

### Task 1: Exploratory Data Analysis (EDA) and Preprocessing
The foundation of the RAG pipeline was established by analyzing the CFPB complaint dataset.

#### Key EDA Findings:
*   **Scale**: Initial analysis covered over **9.6M complaints**.
*   **Product Distribution**: The focus was narrowed to four primary categories: *Credit Card, Personal Loan, Savings Account,* and *Money Transfers*.
*   **Narrative Presence**: We identified that only ~31% of complaints contain narratives. Filtering for narratives and target products resulted in a high-quality subset of **82,164 records**.
*   **Narrative Length**: Word counts vary significantly, highlighting the need for a robust chunking strategy to handle long-form consumer stories.

#### Data Cleaning Steps:
*   Standardized product categories using a comprehensive mapping dictionary.
*   Normalized text (lowercasing, removal of boilerplate headers like "To whom it may concern").
*   Removed special characters while preserving punctuation essential for semantic context.

| Metric | Value |
| :--- | :--- |
| **Total Raw Records** | 9,609,797 |
| **Target Product Subset** | 82,164 |
| **Cleaning Pipeline** | Lowercasing, Regex, Boilerplate removal |

---

### Task 2: Text Chunking, Embedding, and Vector Store Indexing
To prepare the data for the RAG retriever, we implemented a sophisticated indexing pipeline.

#### Stratified Sampling:
Due to the volume of data, we created a **stratified sample of 12,000 records**. This ensures that even smaller categories (like "Money transfers") are proportionally represented compared to "Credit card" complaints, maintaining model fairness and accuracy.

#### Recursive Chunking Strategy:
We utilized LangChain's `RecursiveCharacterTextSplitter` with:
*   **Chunk Size**: 500 characters.
*   **Chunk Overlap**: 50 characters.
**Justification**: This size provides enough context for a single consumer issue while the overlap ensures that semantic meaning isn't lost at the split boundaries.

#### Embedding Model & Vector Store:
*   **Model**: `all-MiniLM-L6-v2` (Sentence Transformers). High performance for semantic similarity at low latency.
*   **Vector Store**: **FAISS** (Facebook AI Similarity Search). Chosen for its efficiency in high-dimensional similarity searches and local persistence capabilities.
*   **Metadata Storage**: Every vector is tagged with `complaint_id`, `product`, and `sub_product`, ensuring full traceability back to the original database for reference.

---

## 3. Next Steps and Key Areas of Focus

### Task 3: Retriever and Generator Implementation
The next phase focus on bridging the gap between retrieval and response.
*   **Retriever**: Implement the similarity search function with adjustable "k" parameters.
*   **Prompt Engineering**: Design system prompts that instruct the LLM to answer *only* based on the provided complaints to eliminate hallucinations.
*   **Generator**: Integrate a LangChain-based generator (e.g., GPT-4 or Hugging Face local models) to synthesize retrieved chunks into coherent summaries.
*   **Evaluation**: Qualitative testing using 5-10 specific "stress test" questions across different product categories.

### Task 4: User Interface (UI) Development
Building the user-facing portal using **Gradio** or **Streamlit**.
*   **Features**: Text input, interactive "Submit" button, and a citation display showing the specific Complaint IDs used to generate the answer.
*   **Transparency**: Ensuring the user can see *why* the AI is giving a specific answer by linking back to raw narratives.

### Challenges & Considerations:
*   **Hallucinations**: Strict prompt constraint is critical.
*   **Performance**: Balancing chunk size with retrieval speed as the index grows.
