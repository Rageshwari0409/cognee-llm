import os
import sys
import sqlite3
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

import database_chromadb as database

def debug_query_memories(username, tag, query_text):
    """
    Performs retrieval analysis by reproducing the search pipeline of database_chromadb
    and returning all candidates with their intermediate scores.
    """
    database.init_db()
    username = username.strip().lower()
    tag = tag.strip().lower()
    query_text = query_text.strip()
    
    conn = sqlite3.connect(database.DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, query, response, timestamp, embedding, subtag
        FROM memories
        WHERE username = ? AND tag = ?
    """, (username, tag))
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return [], []
        
    memories = []
    for r in rows:
        memories.append({
            "id": r[0],
            "query": r[1],
            "response": r[2],
            "timestamp": r[3],
            "embedding": r[4],
            "subtag": r[5]
        })
        
    semantic_scores = {m["id"]: 0.0 for m in memories}
    keyword_scores = {m["id"]: 0.0 for m in memories}
    
    # 1. Fetch semantic cosine similarity from ChromaDB
    collection = database._get_chroma_collection()
    if collection is not None:
        try:
            database.sync_chroma_with_sqlite(username, tag, collection)
            embedding_fn = database.get_embedding_function()
            q_emb = embedding_fn([f"search_query: {query_text}"])[0]
            results = collection.query(
                query_embeddings=[q_emb],
                n_results=len(memories),
                where={"$and": [{"username": username}, {"tag": tag}]}
            )
            if results and "ids" in results and results["ids"]:
                ids_list = results["ids"][0]
                distances_list = results["distances"][0] if "distances" in results and results["distances"] else []
                for idx, str_id in enumerate(ids_list):
                    mem_id = int(str_id)
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
            
        max_score = max(keyword_scores.values()) if keyword_scores else 0.0
        if max_score > 0:
            for mem_id in keyword_scores:
                keyword_scores[mem_id] /= max_score
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
        
    # Sort hybrid matches descending
    all_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
    
    # Split by hybrid relevance threshold
    passed_hybrid = [c for c in all_candidates if c["hybrid_score"] >= database.HYBRID_THRESHOLD or c["semantic_score"] >= 0.60]
    failed_hybrid = [c for c in all_candidates if not (c["hybrid_score"] >= database.HYBRID_THRESHOLD or c["semantic_score"] >= 0.60)]
    
    # 4. Rerank top 15 hybrid candidates using Cross-Encoder
    reranker = database.get_reranker_model()
    now = datetime.now()
    top_passed = passed_hybrid[:15]
    
    if reranker is not None and len(top_passed) > 0:
        try:
            pairs = [(query_text, f"{c['memory']['query']} {c['memory']['response']}") for c in top_passed]
            rerank_scores = reranker.predict(pairs)
            
            for idx, c in enumerate(top_passed):
                raw_score = float(rerank_scores[idx])
                rerank_score = 1.0 / (1.0 + math.exp(-raw_score))
                
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
                
                c["rerank_score"] = rerank_score
                c["recency_score"] = recency_score
                c["is_explicit"] = is_explicit
                c["metadata_boost"] = metadata_boost
                c["final_score"] = final_score
        except Exception as e:
            print("Error during reranking:", e)
            reranker = None
            
    if reranker is None:
        for c in top_passed:
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
            
            c["rerank_score"] = 0.0
            c["recency_score"] = recency_score
            c["is_explicit"] = is_explicit
            c["metadata_boost"] = metadata_boost
            c["final_score"] = final_score
            
    # Filter by Cross-Encoder threshold
    top_passed_ce = []
    failed_ce = []
    
    if reranker is not None:
        for c in top_passed:
            if c.get("rerank_score", 0.0) >= 0.30 or c.get("keyword_score", 0.0) >= 0.30:
                top_passed_ce.append(c)
            else:
                failed_ce.append(c)
    else:
        top_passed_ce = top_passed
        
    # Sort top_passed_ce by final_score descending
    top_passed_ce.sort(key=lambda x: x["final_score"], reverse=True)
    
    # Keep remaining elements that didn't make top 15 cut
    other_passed = passed_hybrid[15:]
    for c in other_passed:
        c["rerank_score"] = 0.0
        c["recency_score"] = 0.0
        c["is_explicit"] = 0.0
        c["metadata_boost"] = 0.0
        c["final_score"] = c["hybrid_score"]
        
    final_passed = top_passed_ce + other_passed
    final_failed = failed_hybrid + failed_ce
    return final_passed, final_failed

def run_debug_search(username, tag, query):
    passed, failed = debug_query_memories(username, tag, query)
    
    print(f"\n======================================================================")
    print(f"🔎 DIAGNOSTIC SEARCH REPORT FOR: \"{query}\"")
    print(f"👤 User: {username} | Tag: {tag.upper()}")
    print(f"======================================================================")
    
    print(f"\n🟢 PASSED RELEVANCE THRESHOLD (hybrid >= {database.HYBRID_THRESHOLD} or semantic >= 0.60) - {len(passed)} matches found:")
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
        print(f"  ├─ Hybrid Score (Combined):   {c['hybrid_score']:.4f}  (Threshold: >= {database.HYBRID_THRESHOLD} or Semantic >= 0.60) -> PASSED")
        if "final_score" in c:
            print(f"  ├─ Cross-Encoder Rerank:      {c.get('rerank_score', 0.0):.4f}  (Weight: 70%)")
            print(f"  ├─ Metadata Boost (Combined): {c.get('metadata_boost', 0.0):.4f}  (Weight: 30%)")
            print(f"  │  ├─ Explicit Subtag Boost: {c.get('is_explicit', 0.0):.4f}  (Weight: 60%)")
            print(f"  │  └─ Recency Boost (7d half):{c.get('recency_score', 0.0):.4f}  (Weight: 40%)")
            print(f"  └─ Final Reranked Score:      {c['final_score']:.4f}")
        else:
            print(f"  └─ Final Score (Hybrid):      {c['hybrid_score']:.4f}")
            
    print(f"\n----------------------------------------------------------------------")
    print(f"🔴 FILTERED OUT / IRRELEVANT (hybrid < {database.HYBRID_THRESHOLD} and semantic < 0.60, or CE/keyword < 0.30) - showing top 5 closest distractors:")
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
            print(f"  ├─ Cross-Encoder Rerank:      {c.get('rerank_score', 0.0):.4f}  (RERANK & Keyword < 0.30) -> FAILED")
            print(f"  └─ Final Reranked Score:      {c['final_score']:.4f}")
        else:
            print(f"  └─ Hybrid Score (Combined):   {c['hybrid_score']:.4f}  (Threshold: >= {database.HYBRID_THRESHOLD} or Semantic >= 0.60) -> FAILED")

def interactive_search():
    print("==================================================")
    print("🧠 AI Workout Coach - Diagnostic Vector Search")
    print("==================================================")
    
    status = database.get_db_status()
    print(f"Database Status: {status['engine_mode']}")
    print(f"Total records in DB: {status['total_records']}")
    print(f"Active users: {status['active_users']}")
    
    conn = sqlite3.connect(database.DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT username FROM memories")
    users = [row[0] for row in cursor.fetchall()]
    conn.close()
    
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
