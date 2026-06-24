import os
import sys
import math
import numpy as np
from dotenv import load_dotenv
load_dotenv() 
# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import database_chroma_new as database

def run_diagnostic():
    print("=== SEMANTIC SEARCH DIAGNOSTIC TEST ===")
    
    # 1. Initialize models
    print("Loading embedding model...")
    embed_model = database.get_transformer_model()
    print("Loading reranker model...")
    rerank_model = database.get_reranker_model()
    
    if embed_model is None or rerank_model is None:
        print("Error: Models could not be loaded!")
        return
        
    # Test documents and queries matching the user's scenario
    documents = [
        "What is your primary fitness goal? Weight loss",
        "What is your diet preference? Vegan",
        "How often do you currently work out? Daily",
        "Do you have any injuries or health conditions? No injuries (I'm good to go!)",
        "Where do you prefer to train? At Home",
        "What is your current weight (in kg)? 80"
    ]
    
    queries = [
        "food",
        "frequently",
        "cherry blossom",
        "trees are green",
        "i love it"
    ]
    
    # Pre-encode documents
    doc_embs = []
    for doc in documents:
        # Nomic required prefix for documents
        doc_emb = embed_model.encode(f"search_document: {doc}")
        # L2 Normalize
        norm = np.linalg.norm(doc_emb)
        if norm > 0:
            doc_emb = doc_emb / norm
        doc_embs.append(doc_emb)
        
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)
        
        # Nomic query encoding
        q_emb = embed_model.encode(f"search_query: {query}")
        norm = np.linalg.norm(q_emb)
        if norm > 0:
            q_emb = q_emb / norm
            
        matches = []
        for idx, doc in enumerate(documents):
            # Cosine similarity (since they are L2 normalized, dot product = cosine sim)
            cos_sim = float(np.dot(q_emb, doc_embs[idx]))
            
            # Cross Encoder score
            pair = (query, doc)
            ce_logit = float(rerank_model.predict(pair))
            ce_sim = 1.0 / (1.0 + math.exp(-ce_logit))
            
            matches.append({
                "doc": doc,
                "cos_sim": cos_sim,
                "ce_logit": ce_logit,
                "ce_sim": ce_sim
            })
            
        # Sort by Cosine Similarity
        print("  Sorted by Cosine Similarity (ChromaDB):")
        matches.sort(key=lambda x: x["cos_sim"], reverse=True)
        for idx, m in enumerate(matches):
            print(f"   [{idx+1}] Cos: {m['cos_sim']:.4f} | Document: '{m['doc']}'")
            
        # Sort by Cross-Encoder Similarity
        print("  Sorted by Cross-Encoder (Reranker):")
        matches.sort(key=lambda x: x["ce_sim"], reverse=True)
        for idx, m in enumerate(matches):
            print(f"   [{idx+1}] CE: {m['ce_sim']:.4f} (Logit: {m['ce_logit']:.2f}) | Document: '{m['doc']}'")

if __name__ == "__main__":
    run_diagnostic()
