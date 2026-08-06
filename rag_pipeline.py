"""
Qlib RAG System
Authors: Mann Patel, Gurparkash Randhawa
Date: August 5, 2026
Student ID: 210852760, 190406260
"""

import os
import re
import numpy as np
from rank_bm25 import BM250kapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import ollama
import random

np.random.seed(42)
random.seed(42)

QLIB_DOCUMENT_URLS = [
    "https://raw.githubusercontent.com/microsoft/qlib/main/README.md",
    "https://raw.githubusercontent.com/microsoft/qlib/main/docs/introduction/introduction.rst",
    "https://raw.githubusercontent.com/microsoft/qlib/main/docs/getting_started/installation.rst",
    "https://raw.githubusercontent.com/microsoft/qlib/main/docs/getting_started/quick_start.rst",
    "https://raw.githubusercontent.com/microsoft/qlib/main/docs/component/data.rst",
    "https://raw.githubusercontent.com/microsoft/qlib/main/docs/component/model.rst",
    "https://raw.githubusercontent.com/microsoft/qlib/main/docs/component/strategy.rst",
    "https://raw.githubusercontent.com/microsoft/qlib/main/docs/component/backtest.rst",
    "https://raw.githubusercontent.com/microsoft/qlib/main/docs/component/workflow.rst",
    "https://raw.githubusercontent.com/microsoft/qlib/main/docs/developer/architecture.rst",
    "https://raw.githubusercontent.com/microsoft/qlib/main/docs/advanced/rl.rst",
    "https://raw.githubusercontent.com/microsoft/qlib/main/docs/advanced/meta.rst",
]

def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +", " ", text)
    return text.strip()

def chunk_document(text: str, source_url: str, chunk_size: int = 150, overlap: int = 30):
    words = text.split()
    chunks = []

    if len(words) == 0:
        return chunks

    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_size)

        if len(chunk_text.strip()) > 30:
            chunks.append({"text": chunk_text, 
                           "metadata": {
                               "source": source_url,
                               "length_words": len(chunk_words)
                           }
                           })

    return chunks

def build_corpus():
    print("Fetching and processing corpus documents...")
    all_chunks = []
    chunk_counter = 0

    for url in QLIB_DOCUMENT_URLS:
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                cleaned = clean_text(res.text)
                doc_chunks = chunk_document(cleaned, source_url=url)
                for c in doc_chunks:
                    c["doc_id"] = f"doc_{chunk_counter}"
                    all_chunks.append(c)
                    chunk_counter += 1
        except Exception as e:
            print(f"Warning: Failed to fetch {url}: {e}")

    print(f"Total Chunks Processed: {len(all_chunks)}")

    return all_chunks


class BM25Retriever:
    def __init__(self, corpus):
        self.corpus = corpus
        self.tokenized_corpus = [doc['text'].lower().split() for doc in corpus]
        self.bm25 = BM250kapi(self.tokenized_corpus)

    def retrieve(self, query: str, top: int = 3):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top]

        return [self.corpus[i] for i in top_indices]

class DenseRetriever:
    def __init__(self, corpus, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.corpus = corpus
        print(f" Initializing Dense Retriever using {model_name}...")
        self.model = SentenceTransformer(model_name)
        texts = [doc['text'] for doc in corpus]
        self.embeddings = self.model.encode(texts, show_progress_bar = True, batch_size = 32)

    def retrieve(self, query: str, top: int = 3):
        query_vec = self.model.encode([query])
        similiraties = cosine_similarity(query_vec, self.embeddings)[0]
        top_indices = np.argsort(similiraties)[::-1][:top]

        return [self.corpus[i] for i in top_indices]


SYSTEM_PROMPT = """You are a precise technical AI assistant answering questions about Microsoft Qlib.
Instructions: 
1. Answer the question relying ONLY on the provided Context Chunks below.
2. Include inline chunk citations for every key fact mentioned (e.g., [doc_14]).
3. If the provided context is insufficient or irrelevant to answer the question, output EXACTLY: "I don't know."
"""

def generate_answer(query: str, retrieved_chunks: list, model_name: str = "llama3.2") -> str:
    context_str = "\n\n".join([f"--- Chunk ID: {c['doc_id']} ---\n{c['text']}" for c in retrieved_chunks])

    user_msg = f"Context Chunks: \n{context_str}\n\nQuestion: {query}"

    try:
        response = ollama.chat(
            model = model_name,
            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_msg}
            ], 
            options = {'temperature': 0.0}
        )

        return response['message']['content'].strip()

    except Exception as e:
        return f"[LLM Call Failed: Ensure Ollama is running Llama 3.2 locally. Error: {e}]"

EVALUATION_SET = [
    {
        "id": 1,
        "question": "What is Microsoft Qlib designed for?",
        "type": "factoid",
        "ground_truth_keywords": ["quantitative", "investment", "AI", "platform"],
        "expected_unanswerable": False
    },
    {
        "id": 2,
        "question": "How is high-frequency financial data stored and handled in Qlib?",
        "type": "factoid",
        "ground_truth_keywords": ["data", "format", "bin", "storage"],
        "expected_unanswerable": False
    },
    {
        "id": 3,
        "question": "What python command initializes data in Qlib?",
        "type": "factoid",
        "ground_truth_keywords": ["qlib.init", "provider_uri"],
        "expected_unanswerable": False
    },
    {
        "id": 4,
        "question": "Which machine learning frameworks does Qlib integrate for model training?",
        "type": "factoid",
        "ground_truth_keywords": ["PyTorch", "LightGBM", "scikit-learn"],
        "expected_unanswerable": False
    },
    {
        "id": 5,
        "question": "What is the purpose of the Strategy module in Qlib?",
        "type": "factoid",
        "ground_truth_keywords": ["portfolio", "generator", "signal", "trading"],
        "expected_unanswerable": False
    },
    {
        "id": 6,
        "question": "How does Qlib perform backtesting for quantitative portfolios?",
        "type": "factoid",
        "ground_truth_keywords": ["executor", "backtest", "position", "trade"],
        "expected_unanswerable": False
    },
    {
        "id": 7,
        "question": "How do data preprocessing modules connect to model training and backtesting in Qlib?",
        "type": "multi-hop",
        "ground_truth_keywords": ["dataset", "handler", "model", "executor"],
        "expected_unanswerable": False
    },
    {
        "id": 8,
        "question": "What are the structural steps involved when converting an offline model experiment to a live workflow execution in Qlib?",
        "type": "multi-hop",
        "ground_truth_keywords": ["workflow", "online", "manager", "record"],
        "expected_unanswerable": False
    },
    {
        "id": 9,
        "question": "What is the real-time stock price of Apple (AAPL) on NASDAQ?",
        "type": "unanswerable",
        "ground_truth_keywords": [],
        "expected_unanswerable": True
    },
    {
        "id": 10,
        "question": "What is the maximum allowed GPU core temperature for TabNet training in Qlib?",
        "type": "unanswerable",
        "ground_truth_keywords": [],
        "expected_unanswerable": True
    }
]

