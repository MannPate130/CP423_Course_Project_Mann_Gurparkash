"""
Qlib RAG System
Authors: Mann Patel, Gurparkash Randhawa
Date: August 5, 2026
Student ID: 210852760, 190406260
"""

import numpy as np
from rank_bm25 import BM250kapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import ollama
import random

