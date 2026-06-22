import os
from dotenv import load_dotenv
load_dotenv()

import sys
import sqlite3
import traceback
import math
import collections
import re
from datetime import datetime, timedelta
import numpy as np

# Path definitions matching database.py
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "memories.db"))
HYBRID_THRESHOLD = 0.48  # Baseline hybrid search threshold


# Models cache to match database.py caching pattern
_model_cache = None
_reranker_cache = None
_chroma_client = None
_chroma_collection = None
_synced_user_tags = set()

# ChromaDB import
try:
    import chromadb
    from chromadb.utils.embedding_functions import ChromaBm25EmbeddingFunction
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

def get_transformer_model():
    """Lazy loader for Nomic Embedding model (matching database.py)."""
    global _model_cache
    if _model_cache is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading sentence-transformers model: nomic-ai/nomic-embed-text-v1.5...")
            _model_cache = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5", 
                trust_remote_code=True
            )
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            _model_cache = None
    return _model_cache

import threading

_local_model_ef_cache = None
_embedding_lock = threading.Lock()

def get_embedding_function():
    """
    Returns a local SentenceTransformer using nomic-ai/nomic-embed-text-v1.5.
    (Google Gemini Embedding is commented out as requested).
    """
    global _local_model_ef_cache
    import chromadb.utils.embedding_functions as embedding_functions
    
    if _local_model_ef_cache is not None:
        return _local_model_ef_cache
        
    with _embedding_lock:
        if _local_model_ef_cache is None:
            print("Initializing local ChromaDB embedding function using nomic-ai/nomic-embed-text-v1.5...")
            from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
            class LocalNomicEmbeddingFunction(EmbeddingFunction):
                def __init__(self):
                    # Reuse the globally cached model to avoid loading 547 MB twice
                    self.model = get_transformer_model()
                    self.model_name = "nomic-ai/nomic-embed-text-v1.5"
                    
                def __call__(self, input: Documents) -> Embeddings:
                    processed_input = []
                    for x in input:
                        if x.startswith("search_query:") or x.startswith("search_document:"):
                            processed_input.append(x)
                        else:
                            processed_input.append(f"search_document: {x}")
                    return self.model.encode(processed_input).astype(np.float32).tolist()
                    
            _local_model_ef_cache = LocalNomicEmbeddingFunction()
    return _local_model_ef_cache


def get_reranker_model():
    """Lazy loader for Cross-Encoder model (matching database.py)."""
    global _reranker_cache
    if _reranker_cache is None:
        try:
            from sentence_transformers import CrossEncoder
            print("Loading CrossEncoder: cross-encoder/ms-marco-MiniLM-L-6-v2...")
            _reranker_cache = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("CrossEncoder loaded successfully.")
        except Exception as e:
            print(f"Error loading reranker: {e}")
            _reranker_cache = None
    return _reranker_cache

def init_db():
    """Initializes the SQLite database and ChromaDB collection."""
    # 1. Initialize SQLite (matches database.py exactly)
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
            if cursor.fetchone():
                cursor.execute("SELECT embedding FROM memories WHERE embedding IS NOT NULL LIMIT 1")
                row = cursor.fetchone()
                if row and row[0]:
                    blob_len = len(row[0])
                    if blob_len != 3072:
                        print(f"Embedding size mismatch ({blob_len} bytes != 3072 bytes). Resetting database memories table.")
                        cursor.execute("DROP TABLE IF EXISTS memories")
                        conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error checking SQLite dimensions: {e}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            tag TEXT NOT NULL,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            embedding BLOB
        )
    """)
    
    cursor.execute("PRAGMA table_info(memories)")
    columns = [info[1] for info in cursor.fetchall()]
    if "subtag" not in columns:
        cursor.execute("ALTER TABLE memories ADD COLUMN subtag TEXT DEFAULT 'implicit'")
        
    conn.commit()
    conn.close()

    # 2. Initialize ChromaDB
    # Ephemeral in-memory client will be used on-the-fly, no persistent setup needed.
    pass

def _get_chroma_collection():
    """Returns an ephemeral in-memory ChromaDB collection or None if library is missing."""
    global _chroma_client, _chroma_collection
    if not HAS_CHROMA:
        return None
    try:
        if _chroma_client is None:
            from chromadb.config import Settings
            _chroma_client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))

        if _chroma_collection is None:
            embedding_fn = get_embedding_function()
            _chroma_collection = _chroma_client.get_or_create_collection(
                name="memories",
                metadata={"hnsw:space": "cosine"},
                embedding_function=embedding_fn
            )
        return _chroma_collection
    except Exception as e:
        print(f"Error fetching ChromaDB collection: {e}")
        return None

def sync_chroma_with_sqlite(username: str, tag: str, collection):
    """
    Self-healing sync logic. Loads all SQLite memories for this user and tag,
    converting SQLite embedding BLOBs back to lists to populate the ephemeral collection.
    If any legacy memories are missing embeddings, computes them on the fly and saves back to SQLite.
    """
    global _synced_user_tags
    if (username, tag) in _synced_user_tags:
        return
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, query, response, timestamp, embedding, subtag
        FROM memories
        WHERE username = ? AND tag = ?
    """, (username, tag))
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return
        
    ids_to_add = []
    documents_to_add = []
    metadatas_to_add = []
    embeddings_to_add = []
    
    embedding_fn = get_embedding_function()
    needs_update_ids = {}
    
    for r in rows:
        mem_id = r[0]
        q_text = r[1]
        r_text = r[2]
        ts = r[3]
        emb_blob = r[4]
        sub = r[5]
        
        doc_text = f"{q_text} {r_text}"
        
        if emb_blob:
            # Parse from stored SQLite bytes
            emb = np.frombuffer(emb_blob, dtype=np.float32).tolist()
        else:
            # Re-compute if missing (self-healing)
            try:
                emb = embedding_fn([doc_text])[0]
                new_blob = np.array(emb, dtype=np.float32).tobytes()
                needs_update_ids[mem_id] = new_blob
            except Exception as e:
                print(f"Error computing missing embedding during sync: {e}")
                continue
                
        ids_to_add.append(str(mem_id))
        embeddings_to_add.append(emb)
        documents_to_add.append(doc_text)
        metadatas_to_add.append({
            "id": mem_id,
            "username": username,
            "tag": tag,
            "timestamp": ts,
            "subtag": sub,
            "query": q_text,
            "response": r_text
        })
        
    # Write self-healed embeddings back to SQLite
    if needs_update_ids:
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            for m_id, blob in needs_update_ids.items():
                cursor.execute("UPDATE memories SET embedding = ? WHERE id = ?", (blob, m_id))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error updating SQLite with healed embeddings: {e}")
            
    if ids_to_add:
        collection.upsert(
            ids=ids_to_add,
            embeddings=embeddings_to_add,
            documents=documents_to_add,
            metadatas=metadatas_to_add
        )
    _synced_user_tags.add((username, tag))



def save_memory(username: str, tag: str, query: str, response: str, subtag: str = "implicit", timestamp: str = None) -> bool:
    """
    Saves a memory in SQLite, computing Gemini embeddings and storing them as a BLOB in SQLite.
    No local vector database cache is stored on disk.
    """
    init_db()
    username = username.strip().lower()
    tag = tag.strip().lower()
    subtag = subtag.strip().lower()
    
    if timestamp is None:
        timestamp = datetime.now().isoformat()
        
    embedding_blob = None
    
    # 1. Compute Gemini Embedding and convert to BLOB bytes
    if HAS_CHROMA:
        try:
            embedding_fn = get_embedding_function()
            combined_text = f"{query} {response}"
            embedding = embedding_fn([combined_text])[0]  # This is a list of floats
            
            # Convert to numpy float32 bytes for storing as SQLite BLOB
            embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
        except Exception as e:
            print(f"Error generating Gemini embedding during save: {e}")
            
    # 2. Save to SQLite (store the computed embedding blob)
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO memories (username, tag, query, response, timestamp, embedding, subtag)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (username, tag, query, response, timestamp, embedding_blob, subtag))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving to SQLite: {e}")
        return False

    global _synced_user_tags
    _synced_user_tags.discard((username, tag))
    return True

def save_or_update_explicit_memory(username: str, query: str, response: str) -> bool:
    """
    Saves or updates an explicit semantic memory (profile onboarding choice).
    If it already exists in SQLite, we update the response, timestamp, and embedding.
    We also clear or update the sync state for ChromaDB so it stays in sync.
    """
    init_db()
    username = username.strip().lower()
    tag = "semantic"
    subtag = "explicit"
    timestamp = datetime.now().isoformat()
    
    # 1. Compute Embedding
    embedding_blob = None
    if HAS_CHROMA:
        try:
            embedding_fn = get_embedding_function()
            combined_text = f"{query} {response}"
            embedding = embedding_fn([combined_text])[0]
            embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
        except Exception as e:
            print(f"Error generating embedding during save_or_update_explicit_memory: {e}")
            
    # 2. Save or Update in SQLite
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id FROM memories
            WHERE username = ? AND tag = ? AND subtag = ? AND query = ?
        """, (username, tag, subtag, query))
        row = cursor.fetchone()
        
        if row:
            mem_id = row[0]
            cursor.execute("""
                UPDATE memories
                SET response = ?, timestamp = ?, embedding = ?
                WHERE id = ?
            """, (response, timestamp, embedding_blob, mem_id))
        else:
            cursor.execute("""
                INSERT INTO memories (username, tag, query, response, timestamp, embedding, subtag)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (username, tag, query, response, timestamp, embedding_blob, subtag))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving/updating explicit memory in SQLite: {e}")
        return False
        
    # 3. Discard synced flag to trigger rebuild
    global _synced_user_tags
    _synced_user_tags.discard((username, tag))
    return True

def get_memories_by_tag(username: str, tag: str) -> list[dict]:
    """Retrieves memories from SQLite matching the tag (same as database.py)."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, query, response, timestamp, subtag
        FROM memories
        WHERE username = ? AND tag = ?
    """, (username, tag))
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for r in rows:
        result.append({
            "id": r[0],
            "query": r[1],
            "response": r[2],
            "timestamp": r[3],
            "subtag": r[4]
        })
    return result

def vector_query_memories(username: str, tag: str, query_text: str, top_k: int = 5) -> list[dict]:
    """
    Query memories using ChromaDB vector database.
    Integrates Hybrid Search (70% Chroma Dense Cosine + 30% Lexical TF-IDF), Relevance Filtering (>= 0.30),
    and Reranking (Cross-Encoder relevance 70% + Metadata boosts 30% with 60% Explicit tag & 40% Recency).
    """
    init_db()
    username = username.strip().lower()
    tag = tag.strip().lower()
    query_text = query_text.strip()
    
    if not query_text:
        return get_memories_by_tag(username, tag)

    # 1. Fetch collection
    collection = _get_chroma_collection()
    if collection is None:
        print("ChromaDB is not active. Falling back to default database implementation.")
        import database
        return database.vector_query_memories(username, tag, query_text, top_k)

    # 2. Heal / Sync ChromaDB against SQLite modifications (e.g. deletions)
    try:
        sync_chroma_with_sqlite(username, tag, collection)
    except Exception as e:
        print(f"Sync warning: {e}")

    # 3. Retrieve memories matching query filters from SQLite to construct response objects
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, query, response, timestamp, subtag
        FROM memories
        WHERE username = ? AND tag = ?
    """, (username, tag))
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return []

    memories_map = {}
    for r in rows:
        memories_map[r[0]] = {
            "id": r[0],
            "query": r[1],
            "response": r[2],
            "timestamp": r[3],
            "subtag": r[4]
        }

    # 4. Calculate Semantic Similarity using ChromaDB query
    semantic_scores = {m_id: 0.0 for m_id in memories_map}
    try:
        where_filter = {"$and": [{"username": username}, {"tag": tag}]}
        
        # Load embedding function to manually pre-embed query with correct Nomic query prefix
        embedding_fn = get_embedding_function()
        q_emb = embedding_fn([f"search_query: {query_text}"])[0]
            
        # Fetch up to 15 candidates using query_embeddings
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=min(15, len(memories_map)),
            where=where_filter
        )
        
        if results and "ids" in results and results["ids"]:
            ids_list = results["ids"][0]
            distances_list = results["distances"][0] if "distances" in results and results["distances"] else []
            
            for idx, str_id in enumerate(ids_list):
                mem_id = int(str_id)
                if mem_id in semantic_scores:
                    dist = distances_list[idx] if idx < len(distances_list) else 0.0
                    # Cosine similarity = 1.0 - cosine_distance in cosine index
                    sim = 1.0 - dist
                    semantic_scores[mem_id] = max(0.0, min(1.0, sim))
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")

    # 5. Calculate Lexical Similarity (ChromaBm25EmbeddingFunction)
    keyword_scores = {m_id: 0.0 for m_id in memories_map}
    try:
        memories_list = list(memories_map.values())
        documents = [f"{m['query']} {m['response']}" for m in memories_list]
        
        # Initialize BM25 embedding function using Chroma's utility
        bm25_ef = ChromaBm25EmbeddingFunction(
            k=1.5,
            b=0.75
        )
        
        # Batch all document texts and query text to calculate relative document frequencies
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
            # Match query term indices and sum their BM25 weights
            score = sum(d_dict.get(term_idx, 0.0) for term_idx in q_dict.keys())
            mem_id = memories_list[idx]["id"]
            keyword_scores[mem_id] = float(score)
            
        # Normalize keyword scores to [0.0, 1.0] range
        max_score = max(keyword_scores.values()) if keyword_scores else 0.0
        if max_score > 0:
            for mem_id in keyword_scores:
                keyword_scores[mem_id] /= max_score
    except Exception as e:
        print(f"ChromaBm25EmbeddingFunction error: {e}")

    # 6. Fuse Scores (70% Semantic, 30% Keyword)
    scored_memories = []
    for m_id, m in memories_map.items():
        sem_s = semantic_scores.get(m_id, 0.0)
        key_s = keyword_scores.get(m_id, 0.0)
        
        hybrid_score = (0.7 * sem_s) + (0.3 * key_s)
        
        # Apply combined relevance threshold (0.30) or allow high-confidence semantic matches (>= 0.60)
        if hybrid_score >= HYBRID_THRESHOLD or sem_s >= 0.60:
            scored_memories.append((hybrid_score, m))

    # Sort candidates
    scored_memories.sort(key=lambda x: x[0], reverse=True)
    candidates = scored_memories[:15]

    # 7. Apply Reranking with Cross-Encoder and Preference Weights
    reranker = get_reranker_model()
    final_scored = []
    now = datetime.now()
    
    if reranker is not None and len(candidates) > 0:
        try:
            pairs = []
            for _, m in candidates:
                doc_text = f"{m['query']} {m['response']}"
                pairs.append((query_text, doc_text))
                
            rerank_scores = reranker.predict(pairs)
            
            for idx, (_, m) in enumerate(candidates):
                raw_score = float(rerank_scores[idx])
                # Sigmoid normalization
                rerank_score = 1.0 / (1.0 + math.exp(-raw_score))
                
                # Recency score
                try:
                    dt = datetime.fromisoformat(m["timestamp"])
                    age_hours = (now - dt).total_seconds() / 3600.0
                    if age_hours < 0:
                        age_hours = 0.0
                    # 7-day half-life decay (168 hours)
                    recency_score = 1.0 / (1.0 + age_hours / 168.0)
                except Exception:
                    recency_score = 0.0
                    
                # Metadata boost (60% explicit preference, 40% recency)
                is_explicit = 1.0 if m["subtag"] == "explicit" else 0.0
                metadata_boost = (0.6 * is_explicit) + (0.4 * recency_score)
                
                # Combined score: 70% CE relevance + 30% metadata boost
                final_score = (0.7 * rerank_score) + (0.3 * metadata_boost)
                
                # Filter out weak candidate if it doesn't meet the rerank or lexical relevance threshold
                key_s = keyword_scores.get(m["id"], 0.0)
                if rerank_score >= 0.30 or key_s >= 0.30:
                    final_scored.append((final_score, m))
        except Exception as e:
            print(f"Reranking error: {e}. Falling back to standard hybrid.")
            reranker = None
            
    if reranker is None:
        for hybrid_score, m in candidates:
            try:
                dt = datetime.fromisoformat(m["timestamp"])
                age_hours = (now - dt).total_seconds() / 3600.0
                if age_hours < 0:
                    age_hours = 0.0
                recency_score = 1.0 / (1.0 + age_hours / 168.0)
            except Exception:
                recency_score = 0.0
                
            is_explicit = 1.0 if m["subtag"] == "explicit" else 0.0
            metadata_boost = (0.6 * is_explicit) + (0.4 * recency_score)
            
            final_score = (0.7 * hybrid_score) + (0.3 * metadata_boost)
            final_scored.append((final_score, m))

    # 8. Sort by highest final score and return top_k
    final_scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in final_scored[:top_k]]

def get_db_status() -> dict:
    """Returns the current status of the vector database engine."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM memories")
    total_memories = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL")
    memories_with_embeddings = cursor.fetchone()[0]
    
    cursor.execute("SELECT DISTINCT tag FROM memories")
    tags = [row[0] for row in cursor.fetchall()]
    
    cursor.execute("SELECT DISTINCT username FROM memories")
    users = [row[0] for row in cursor.fetchall()]
    conn.close()

    gemini_key_configured = os.environ.get("GEMINI_API_KEY") is not None

    return {
        "chroma_available": HAS_CHROMA,
        "chroma_library": "Available" if HAS_CHROMA else "Missing",
        "sentence_transformers_library": "Available",
        "model_loaded": "Yes (Google Gemini API)" if gemini_key_configured else "No (Missing API Key)",
        "total_records": total_memories,
        "records_with_vectors": memories_with_embeddings,
        "active_users": len(users),
        "memory_tags": tags,
        "engine_mode": "ChromaDB (Semantic + BM25 Hybrid)" if HAS_CHROMA else "NumPy Fallback (TF-IDF)"
    }

def warm_up_cache(username: str):
    """Pre-syncs all tags for the given user into ChromaDB cache in a background thread."""
    import threading
    username = username.strip().lower()
    def _warm_up():
        try:
            collection = _get_chroma_collection()
            if collection is not None:
                for tag in ["semantic", "episodic", "procedural"]:
                    sync_chroma_with_sqlite(username, tag, collection)
        except Exception as e:
            print(f"Warm up cache failed: {e}")
            
    threading.Thread(target=_warm_up, daemon=True).start()

