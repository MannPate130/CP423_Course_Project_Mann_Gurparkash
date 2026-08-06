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




