import os
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

class RAGPipeline:
    def __init__(self, vector_store_path="vector_store/faiss_index", model_name="all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}...")
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        print(f"Loading vector store from: {vector_store_path}...")
        self.vector_store = FAISS.load_local(
            vector_store_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Generator Setup
        model_id = "google/flan-t5-small"
        print(f"Loading local LLM: {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        
        self.pipe = pipeline(
            "text2text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer,
            max_new_tokens=256, 
            device=-1 # CPU
        )
        
        self.template = """You are a financial analyst assistant for CrediTrust. Your task is to provide accurate answers about customer complaints using ONLY the provided context.

Instructions:
1. If the context contains the answer, summarize it clearly.
2. If the context does NOT explicitly mention the topic or contain the answer, state: "Based on the retrieved snippets, I don't have enough information to confirm that."
3. Do NOT make up facts or say 'No' unless the context explicitly provides a negative.

Chat History:
{history}

Context: {context}

Question: {question}

Answer:"""

    def answer_question(self, question, history="", k=5):
        """Advanced RAG implementation with history and parameterized retrieval."""
        # 1. Retrieval
        docs = self.vector_store.similarity_search(question, k=k)
        context = "\n\n".join([f"[Source {i+1}]: {d.page_content}" for i, d in enumerate(docs)])
        
        # 2. Generation
        prompt = self.template.format(history=history, context=context, question=question)
        response = self.pipe(prompt)
        
        return {
            "result": response[0]["generated_text"],
            "source_documents": docs,
            "prompt_used": prompt
        }

    def stream_answer(self, question, history="", k=5):
        """Yield tokens using TextIteratorStreamer for Streamlit."""
        # 1. Retrieval
        docs = self.vector_store.similarity_search(question, k=k)
        context = "\n\n".join([f"[Source {i+1}]: {d.page_content}" for i, d in enumerate(docs)])
        
        # 2. Generation Setup
        prompt = self.template.format(history=history, context=context, question=question)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # 3. Threaded Generation
        generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=256)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        return streamer, docs

def run_evaluation(rag, report_path="reports/task3_evaluation.md"):
    eval_questions = [
        "What are the common issues reported for Credit cards?",
        "How do customers describe problems with money transfers?",
        "Are there complaints about savings account interest rates?",
        "What are the main sub-products in the Personal Loan category?",
        "Does the data contain any Buy Now Pay Later (BNPL) complaints?"
    ]
    
    print("\n--- RESULTS ---")
    results = []
    for q in eval_questions:
        try:
            print(f"Querying: {q}")
            res = rag.answer_question(q)
            results.append({
                "question": q,
                "answer": res["result"],
                "source_1": res["source_documents"][0].metadata.get("complaint_id", "N/A") if res["source_documents"] else "N/A",
                "source_2": res["source_documents"][1].metadata.get("complaint_id", "N/A") if len(res["source_documents"]) > 1 else "N/A"
            })
        except Exception as e:
            print(f"Error: {e}")
            
    # Save to Markdown
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Task 3: RAG Qualitative Evaluation\n\n")
        f.write("| Question | Generated Answer | Retrieved Sources | Quality (1-5) | Comments |\n")
        f.write("| :--- | :--- | :--- | :---: | :--- |\n")
        for r in results:
            sources = f"{r['source_1']}, {r['source_2']}"
            f.write(f"| {r['question']} | {r['answer']} | {sources} | | |\n")
    
    print(f"\nEvaluation complete. Results saved to {report_path}")

if __name__ == "__main__":
    try:
        pipeline_inst = RAGPipeline()
        run_evaluation(pipeline_inst)
    except Exception as e:
        print(f"Critical Failure: {e}")
