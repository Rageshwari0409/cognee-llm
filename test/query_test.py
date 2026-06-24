import os
import sys
import math
import numpy as np
from datetime import datetime

# Resolve parent directory for database import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import database_chroma_new as database

def debug_query_memories(username, tag, query_text):
    """
    Performs retrieval analysis by reproducing the search pipeline of database_chroma_new
    and returning all candidates with their intermediate scores.
    """
    database.init_db()
    username = username.strip().lower()
    tag = tag.strip().lower()
    query_text = query_text.strip()
    
    collection = database._get_chroma_collection()
    if collection is None:
        return [], []
        
    try:
        results = collection.get(
            where={"$and": [{"username": username}, {"tag": tag}]}
        )
    except Exception as e:
        print("Error fetching from ChromaDB:", e)
        return [], []
        
    if not results or not results["ids"]:
        return [], []
        
    memories = []
    for idx, str_id in enumerate(results["ids"]):
        meta = results["metadatas"][idx]
        memories.append({
            "id": str_id,
            "query": meta.get("query", ""),
            "response": meta.get("response", ""),
            "timestamp": meta.get("timestamp", ""),
            "subtag": meta.get("subtag", "implicit")
        })
        
    semantic_scores = {m["id"]: 0.0 for m in memories}
    keyword_scores = {m["id"]: 0.0 for m in memories}
    
    # 1. Fetch semantic cosine similarity from ChromaDB
    try:
        embedding_fn = database.get_embedding_function()
        q_emb = embedding_fn([f"search_query: {query_text}"])[0]
        search_results = collection.query(
            query_embeddings=[q_emb],
            n_results=len(memories),
            where={"$and": [{"username": username}, {"tag": tag}]}
        )
        if search_results and "ids" in search_results and search_results["ids"]:
            ids_list = search_results["ids"][0]
            distances_list = search_results["distances"][0] if "distances" in search_results and search_results["distances"] else []
            for idx, str_id in enumerate(ids_list):
                mem_id = str_id
                dist = distances_list[idx] if idx < len(distances_list) else 0.0
                sim = 1.0 - dist
                semantic_scores[mem_id] = max(0.0, min(1.0, sim))
        except Exception as e:
            print("Error during Chroma search:", e)
            
    # 2. Calculate Lexical Similarity (ChromaBm25EmbeddingFunction)
    try:
        from chromadb.utils.embedding_functions import ChromaBm25EmbeddingFunction
        bm25_ef = ChromaBm25EmbeddingFunction(k=1.5, b=0.75)
        documents = [f"{m['query']} {m['response']}" for m in memories]
        all_texts = documents + [query_text]
        all_embs = bm25_ef(all_texts)
        doc_embs = all_embs[:-1]
        query_emb = all_embs[-1]
        
        def get_sparse_dict(emb):
            if isinstance(emb, dict):
                return dict(zip(emb.get("indices", []), emb.get("values", [])))
            indices = getattr(emb, "indices", [])
            values = getattr(emb, "values", [])
            return dict(zip(indices, values))
            
        q_dict = get_sparse_dict(query_emb)
        
        for idx, doc_emb in enumerate(doc_embs):
            d_dict = get_sparse_dict(doc_emb)
            score = sum(d_dict.get(term_idx, 0.0) for term_idx in q_dict.keys())
            mem_id = memories[idx]["id"]
            keyword_scores[mem_id] = float(score)
            
        for mem_id in keyword_scores:
            keyword_scores[mem_id] = math.tanh(keyword_scores[mem_id] / 3.0)
    except Exception as e:
        print("Error during BM25 search:", e)
        
    # 3. Fuse Scores
    all_candidates = []
    for m in memories:
        mem_id = m["id"]
        sem_s = semantic_scores.get(mem_id, 0.0)
        key_s = keyword_scores.get(mem_id, 0.0)
        hybrid_score = (0.7 * sem_s) + (0.3 * key_s)
        all_candidates.append({
            "memory": m,
            "semantic_score": sem_s,
            "keyword_score": key_s,
            "hybrid_score": hybrid_score
        })
        
    # 3. Interleave Candidates (no pre-filtering threshold)
    by_semantic = sorted(all_candidates, key=lambda x: x["semantic_score"], reverse=True)
    by_keyword = sorted(all_candidates, key=lambda x: x["keyword_score"], reverse=True)
    
    candidate_ids = set()
    top_candidates = []
    
    for i in range(max(len(by_semantic), len(by_keyword))):
        if i < len(by_semantic):
            c = by_semantic[i]
            m_id = c["memory"]["id"]
            if m_id not in candidate_ids:
                candidate_ids.add(m_id)
                top_candidates.append(c)
        if i < len(by_keyword):
            c = by_keyword[i]
            m_id = c["memory"]["id"]
            if m_id not in candidate_ids:
                candidate_ids.add(m_id)
                top_candidates.append(c)
        if len(top_candidates) >= 15:
            break
            
    # Remaining candidates that didn't make the top 15 cut
    distractors = [c for c in all_candidates if c["memory"]["id"] not in candidate_ids]
    
    # 4. Rerank top candidates using Cross-Encoder
    reranker = database.get_reranker_model()
    now = datetime.now()
    
    passed_gate = []
    failed_gate = []
    
    if reranker is not None and len(top_candidates) > 0:
        try:
            pairs = [(query_text, f"{c['memory']['query']} {c['memory']['response']}") for c in top_candidates]
            rerank_scores = reranker.predict(pairs)
            
            for idx, c in enumerate(top_candidates):
                raw_score = float(rerank_scores[idx])
                rerank_score = 1.0 / (1.0 + math.exp(-raw_score))
                c["rerank_score"] = rerank_score
                
                try:
                    dt = datetime.fromisoformat(c["memory"]["timestamp"])
                    age_hours = (now - dt).total_seconds() / 3600.0
                    if age_hours < 0:
                        age_hours = 0.0
                    recency_score = 1.0 / (1.0 + age_hours / 168.0)
                except Exception:
                    recency_score = 0.0
                    
                is_explicit = 1.0 if c["memory"]["subtag"] == "explicit" else 0.0
                metadata_boost = (0.6 * is_explicit) + (0.4 * recency_score)
                final_score = (0.7 * rerank_score) + (0.3 * metadata_boost)
                
                c["recency_score"] = recency_score
                c["is_explicit"] = is_explicit
                c["metadata_boost"] = metadata_boost
                c["final_score"] = final_score
                
                # Apply Relevance Gate (Cross-Encoder score must be >= 0.002)
                if rerank_score >= 0.002:
                    passed_gate.append(c)
                else:
                    failed_gate.append(c)
        except Exception as e:
            print("Error during reranking:", e)
            reranker = None
            
    if reranker is None:
        for c in top_candidates:
            c["rerank_score"] = 0.0
            try:
                dt = datetime.fromisoformat(c["memory"]["timestamp"])
                age_hours = (now - dt).total_seconds() / 3600.0
                if age_hours < 0:
                    age_hours = 0.0
                recency_score = 1.0 / (1.0 + age_hours / 168.0)
            except Exception:
                recency_score = 0.0
                
            is_explicit = 1.0 if c["memory"]["subtag"] == "explicit" else 0.0
            metadata_boost = (0.6 * is_explicit) + (0.4 * recency_score)
            final_score = (0.7 * c["hybrid_score"]) + (0.3 * metadata_boost)
            
            c["recency_score"] = recency_score
            c["is_explicit"] = is_explicit
            c["metadata_boost"] = metadata_boost
            c["final_score"] = final_score
            
            # Apply Relevance Gate for fallback (Hybrid score >= 0.25 or Semantic Cosine Similarity >= 0.55)
            if c["hybrid_score"] >= 0.25 or c["semantic_score"] >= 0.55:
                passed_gate.append(c)
            else:
                failed_gate.append(c)
                
    for c in distractors:
        c["rerank_score"] = 0.0
        c["recency_score"] = 0.0
        c["is_explicit"] = 0.0
        c["metadata_boost"] = 0.0
        c["final_score"] = c["hybrid_score"]
        
    passed_gate.sort(key=lambda x: x["final_score"], reverse=True)
    
    final_passed = passed_gate
    final_failed = failed_gate + distractors
    final_failed.sort(key=lambda x: x["final_score"], reverse=True)
    return final_passed, final_failed

def run_debug_search(username, tag, query):
    passed, failed = debug_query_memories(username, tag, query)
    
    print(f"\n======================================================================")
    print(f"🔎 DIAGNOSTIC SEARCH REPORT FOR: \"{query}\"")
    print(f"👤 User: {username} | Tag: {tag.upper()}")
    print(f"======================================================================")
    
    print(f"\n🟢 PASSED RELEVANCE GATE (CE >= 0.002; fallback hybrid >= 0.25 or semantic >= 0.55) - {len(passed)} matches found:")
    if not passed:
        print("  (No memories matched your query above the relevance threshold)")
    for idx, c in enumerate(passed):
        m = c["memory"]
        print(f"\nRank #{idx+1}: [Memory ID: {m['id']}] (Subtag: {m['subtag']})")
        print(f"  Query:    \"{m['query']}\"")
        print(f"  Response: \"{m['response']}\"")
        print(f"  Score Breakdown:")
        print(f"  ├─ Semantic Score (Cosine):   {c['semantic_score']:.4f}  (Weight: 70%)")
        print(f"  ├─ Lexical Score (BM25):      {c['keyword_score']:.4f}  (Weight: 30%)")
        print(f"  ├─ Hybrid Score (Combined):   {c['hybrid_score']:.4f}  (Threshold: >= 0.25 or Semantic >= 0.55) -> PASSED")
        if "final_score" in c:
            print(f"  ├─ Cross-Encoder Rerank:      {c.get('rerank_score', 0.0):.4f}  (Weight: 70%)")
            print(f"  ├─ Metadata Boost (Combined): {c.get('metadata_boost', 0.0):.4f}  (Weight: 30%)")
            print(f"  │  ├─ Explicit Subtag Boost: {c.get('is_explicit', 0.0):.4f}  (Weight: 60%)")
            print(f"  │  └─ Recency Boost (7d half):{c.get('recency_score', 0.0):.4f}  (Weight: 40%)")
            print(f"  └─ Final Reranked Score:      {c['final_score']:.4f}")
        else:
            print(f"  └─ Final Score (Hybrid):      {c['hybrid_score']:.4f}")
            
    print(f"\n----------------------------------------------------------------------")
    print(f"🔴 FILTERED OUT / IRRELEVANT (CE < 0.002; fallback hybrid < 0.25 and semantic < 0.55) - showing top 5 closest distractors:")
    if not failed:
        print("  (No memories failed the relevance threshold)")
    for idx, c in enumerate(failed[:5]):
        m = c["memory"]
        print(f"\nDistractor #{idx+1}: [Memory ID: {m['id']}] (Subtag: {m['subtag']})")
        print(f"  Query:    \"{m['query']}\"")
        print(f"  Response: \"{m['response']}\"")
        print(f"  Score Breakdown:")
        print(f"  ├─ Semantic Score (Cosine):   {c['semantic_score']:.4f}")
        print(f"  ├─ Lexical Score (BM25):      {c['keyword_score']:.4f}")
        print(f"  ├─ Hybrid Score (Combined):   {c['hybrid_score']:.4f}")
        if "final_score" in c:
            print(f"  ├─ Cross-Encoder Rerank:      {c.get('rerank_score', 0.0):.4f} -> FAILED")
            print(f"  └─ Final Reranked Score:      {c['final_score']:.4f}")
        else:
            print(f"  └─ Hybrid Score (Combined):   {c['hybrid_score']:.4f} -> FAILED")

def interactive_search():
    print("==================================================")
    print("🧠 AI Workout Coach - Diagnostic Vector Search")
    print("==================================================")
    
    status = database.get_db_status()
    print(f"Database Status: {status['engine_mode']}")
    print(f"Total records in DB: {status['total_records']}")
    print(f"Active users: {status['active_users']}")
    
    collection = database._get_chroma_collection()
    users = []
    if collection is not None:
        try:
            results = collection.get()
            if results and "metadatas" in results and results["metadatas"]:
                users = list(set(meta.get("username") for meta in results["metadatas"] if meta.get("username")))
        except Exception as e:
            print("Error listing users from ChromaDB:", e)

    
    if not users:
        print("\n❌ No memories found in database! Please populate the database or run evaluations first.")
        return
        
    print("\nSelect Username to Query:")
    for idx, u in enumerate(users):
        print(f" [{idx + 1}] {u}")
        
    choice = input("\nEnter number (default 1): ").strip()
    username = users[0]
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(users):
            username = users[idx]
            
    print(f"-> Querying memories for user: '{username}'")
    
    tags = ["semantic", "episodic", "procedural"]
    print("\nSelect Memory Tag:")
    print(" [1] semantic (General facts, rules, schedules)")
    print(" [2] episodic (Logs, diary, specific events)")
    print(" [3] procedural (Workout sequences, technique)")
    print(" [4] all (Query all tags individually)")
    
    tag_choice = input("Enter choice (1-4, default 4): ").strip()
    query_tag = "all"
    if tag_choice == "1":
        query_tag = "semantic"
    elif tag_choice == "2":
        query_tag = "episodic"
    elif tag_choice == "3":
        query_tag = "procedural"
        
    print(f"-> Querying tag: '{query_tag}'")
    print("\n==================================================")
    print("Enter custom queries to examine raw score details.")
    print("Type 'exit' or 'q' to quit.")
    print("==================================================")
    
    while True:
        query = input("\nSearch Query > ").strip()
        if not query or query.lower() in ["exit", "quit", "q"]:
            break
            
        print("-" * 70)
        
        if query_tag != "all":
            run_debug_search(username, query_tag, query)
        else:
            for tag in tags:
                run_debug_search(username, tag, query)
        print("-" * 70)

if __name__ == "__main__":
    interactive_search()
