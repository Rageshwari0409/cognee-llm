import sqlite3
import os
import json
import re
import collections
import math
import traceback
import threading
from datetime import datetime

# Check if numpy is available
import numpy as np

# Lazy load FAISS and SentenceTransformer to keep imports quick
HAS_FAISS = False
_model_lock = threading.Lock()
_model_cache = None

def get_transformer_model():
    global _model_cache, HAS_FAISS
    with _model_lock:
        if _model_cache is not None:
            return _model_cache
        try:
            from sentence_transformers import SentenceTransformer
            # Correct Hugging Face repo name is "nomic-ai/nomic-embed-text-v1.5"
            _model_cache = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
            HAS_FAISS = True
            return _model_cache
        except Exception as e:
            print("--- MODEL LOAD ERROR ---")
            traceback.print_exc()
            print("------------------------")
            HAS_FAISS = False
            return None

_reranker_lock = threading.Lock()
_reranker_cache = None

def get_reranker_model():
    global _reranker_cache
    with _reranker_lock:
        if _reranker_cache is not None:
            return _reranker_cache
        try:
            from sentence_transformers import CrossEncoder
            # Use a lightweight, high-performance cross-encoder
            _reranker_cache = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            return _reranker_cache
        except Exception as e:
            print(f"Reranker load failed: {e}. Running without cross-encoder reranking.")
            return None

# Check if faiss is importable
try:
    import faiss
except ImportError:
    HAS_FAISS = False

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memories.db")
HYBRID_THRESHOLD = 0.48  # Standard relevance threshold to filter out unrelated distractors (set to 0.48 to block vague queries)

def init_db():
    """Initializes the SQLite database for storing memories and performs auto-migration."""
    # Auto-detect embedding size mismatch to prevent FAISS shape mismatch crashes
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            # Check if memories table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
            if cursor.fetchone():
                cursor.execute("SELECT embedding FROM memories WHERE embedding IS NOT NULL LIMIT 1")
                row = cursor.fetchone()
                if row and row[0]:
                    blob_len = len(row[0])
                    # nomic-embed-text-v1.5 has 768 dimensions (768 * 4 = 3072 bytes)
                    if blob_len != 3072:
                        print(f"Embedding size mismatch ({blob_len} bytes != 3072 bytes). Resetting database memories table.")
                        cursor.execute("DROP TABLE IF EXISTS memories")
                        conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error checking database dimensions: {e}")

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
            embedding BLOB,
            subtag TEXT DEFAULT 'implicit'
        )
    """)
    
    # Auto-migration: check if 'subtag' column exists
    cursor.execute("PRAGMA table_info(memories)")
    columns = [info[1] for info in cursor.fetchall()]
    if "subtag" not in columns:
        cursor.execute("ALTER TABLE memories ADD COLUMN subtag TEXT DEFAULT 'implicit'")
        
    conn.commit()
    conn.close()


class SimpleTFIDF:
    """A lightweight pure-python/numpy TF-IDF vectorizer fallback."""
    def __init__(self):
        self.vocabulary = {}
        self.idf = {}
        
    def _tokenize(self, text):
        text = text.lower()
        return re.findall(r'\b[a-z0-9]+\b', text)
        
    def fit(self, documents):
        doc_tokens = [self._tokenize(doc) for doc in documents]
        vocab = set()
        for tokens in doc_tokens:
            vocab.update(tokens)
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(vocab))}
        
        num_docs = len(documents)
        doc_counts = collections.defaultdict(int)
        for tokens in doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                if token in self.vocabulary:
                    doc_counts[token] += 1
                    
        self.idf = {}
        for word, count in doc_counts.items():
            self.idf[word] = math.log((1 + num_docs) / (1 + count)) + 1
            
    def transform(self, documents):
        if not self.vocabulary:
            return np.zeros((len(documents), 1))
        
        vectors = np.zeros((len(documents), len(self.vocabulary)))
        for i, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            counter = collections.Counter(tokens)
            for token, count in counter.items():
                if token in self.vocabulary:
                    idx = self.vocabulary[token]
                    tf = count / len(tokens) if tokens else 0
                    vectors[i, idx] = tf * self.idf[token]
        return vectors
        
    def cosine_similarity(self, q_vector, doc_vectors):
        q_norm = np.linalg.norm(q_vector)
        if q_norm == 0:
            return np.zeros(doc_vectors.shape[0])
        doc_norms = np.linalg.norm(doc_vectors, axis=1)
        doc_norms[doc_norms == 0] = 1.0
        
        scores = np.dot(doc_vectors, q_vector) / (doc_norms * q_norm)
        return scores

def save_memory(username: str, tag: str, query: str, response: str, subtag: str = "implicit") -> bool:
    """
    Saves a memory in SQLite, computing and normalizing embeddings if FAISS/SentenceTransformer
    are available.
    """
    init_db()
    username = username.strip().lower()
    tag = tag.strip().lower()
    subtag = subtag.strip().lower()
    timestamp = datetime.now().isoformat()
    
    embedding_blob = None
    
    # Try generating embedding
    model = get_transformer_model()
    if model is not None:
        try:
            # Prepend Nomic's required document prefix
            combined_text = f"search_document: {query} {response}"
            embedding = model.encode(combined_text).astype(np.float32)
            
            # Explicitly normalize vector to unit length (L2 norm = 1.0)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            embedding_blob = embedding.tobytes()
        except Exception:
            print("Error generating embedding during save:")
            traceback.print_exc()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO memories (username, tag, query, response, timestamp, embedding, subtag)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (username, tag, query, response, timestamp, embedding_blob, subtag))
    conn.commit()
    conn.close()
    return True

def save_or_update_explicit_memory(username: str, query: str, response: str) -> bool:
    """
    Saves or updates an explicit semantic memory in SQLite (FAISS/NumPy fallback engine).
    """
    init_db()
    username = username.strip().lower()
    tag = "semantic"
    subtag = "explicit"
    timestamp = datetime.now().isoformat()
    
    embedding_blob = None
    model = get_transformer_model()
    if model is not None:
        try:
            combined_text = f"search_document: {query} {response}"
            embedding = model.encode(combined_text).astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            embedding_blob = embedding.tobytes()
        except Exception as e:
            print(f"Error generating embedding in save_or_update_explicit_memory: {e}")
            
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
        print(f"Error saving/updating explicit memory in SQLite fallback: {e}")
        return False
    return True

def get_memories_by_tag(username: str, tag: str) -> list[dict]:
    """
    Fetches all memories for a user with a specific tag, sorted chronologically.
    """
    init_db()
    username = username.strip().lower()
    tag = tag.strip().lower()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, query, response, timestamp, subtag
        FROM memories
        WHERE username = ? AND tag = ?
        ORDER BY timestamp DESC
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
    Searches memories using Hybrid Search (70% Semantic FAISS + 30% Lexical TF-IDF).
    Filters results using a combined relevance threshold (>= 0.35) and returns
    matches sorted chronologically.
    """
    init_db()
    username = username.strip().lower()
    tag = tag.strip().lower()
    query_text = query_text.strip()
    
    if not query_text:
        return get_memories_by_tag(username, tag)

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
        return []

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

    # Initialize scores for all entries
    semantic_scores = {m["id"]: 0.0 for m in memories}
    keyword_scores = {m["id"]: 0.0 for m in memories}

    # 1. Calculate Semantic Similarity (FAISS FlatIP)
    model = get_transformer_model()
    if model is not None and any(m["embedding"] is not None for m in memories):
        try:
            valid_memories = [m for m in memories if m["embedding"] is not None]
            embeddings = [np.frombuffer(m["embedding"], dtype=np.float32) for m in valid_memories]
            
            if len(embeddings) > 0:
                embeddings_matrix = np.stack(embeddings)
                dimension = embeddings_matrix.shape[1]
                
                # Build Flat IP Index (unit normalized vectors -> Inner Product = Cosine Similarity)
                index = faiss.IndexFlatIP(dimension)
                index.add(embeddings_matrix)
                
                # Encode and normalize query
                q_emb = model.encode(f"search_query: {query_text}").astype(np.float32)
                q_norm = np.linalg.norm(q_emb)
                if q_norm > 0:
                    q_emb = q_emb / q_norm
                q_emb = q_emb.reshape(1, -1)
                
                distances, indices = index.search(q_emb, len(valid_memories))
                
                for i, idx in enumerate(indices[0]):
                    if idx != -1 and idx < len(valid_memories):
                        mem_id = valid_memories[idx]["id"]
                        semantic_scores[mem_id] = float(distances[0][i])
        except Exception:
            print("Error during FAISS search:")
            traceback.print_exc()

    # 2. Calculate Lexical Similarity (NumPy TF-IDF)
    try:
        documents = [f"{m['query']} {m['response']}" for m in memories]
        tfidf = SimpleTFIDF()
        tfidf.fit(documents)
        
        doc_vectors = tfidf.transform(documents)
        q_vector = tfidf.transform([query_text])[0]
        
        scores = tfidf.cosine_similarity(q_vector, doc_vectors)
        for idx, score in enumerate(scores):
            mem_id = memories[idx]["id"]
            keyword_scores[mem_id] = float(score)
    except Exception:
        pass

    # 3. Fuse Scores (70% Semantic, 30% Keyword)
    scored_memories = []
    for m in memories:
        mem_id = m["id"]
        sem_s = semantic_scores.get(mem_id, 0.0)
        key_s = keyword_scores.get(mem_id, 0.0)
        
        # Weighted combination score
        hybrid_score = (0.7 * sem_s) + (0.3 * key_s)
        
        # Apply combined threshold on standard hybrid score, or allow high-confidence semantic matches (>= 0.60)
        if hybrid_score >= HYBRID_THRESHOLD or sem_s >= 0.60:
            scored_memories.append((hybrid_score, m))

    # Sort hybrid matches descending
    scored_memories.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 15 candidates for reranking
    candidates = scored_memories[:15]
    
    # 4. Rerank using Cross-Encoder if available
    reranker = get_reranker_model()
    final_scored = []
    now = datetime.now()
    
    if reranker is not None and len(candidates) > 0:
        try:
            # Prepare pairs: (query_text, document_text)
            pairs = []
            for _, m in candidates:
                doc_text = f"{m['query']} {m['response']}"
                pairs.append((query_text, doc_text))
                
            # Predict scores (higher is better)
            rerank_scores = reranker.predict(pairs)
            
            # Normalize cross-encoder scores (sigmoid: 1 / (1 + exp(-x)))
            for idx, (_, m) in enumerate(candidates):
                raw_score = float(rerank_scores[idx])
                rerank_score = 1.0 / (1.0 + math.exp(-raw_score))
                
                # Calculate recency score
                try:
                    dt = datetime.fromisoformat(m["timestamp"])
                    age_hours = (now - dt).total_seconds() / 3600.0
                    if age_hours < 0:
                        age_hours = 0.0
                    recency_score = 1.0 / (1.0 + age_hours / 168.0)
                except Exception:
                    recency_score = 0.0
                    
                # Calculate metadata boost (60% explicit subtag preference, 40% recency boost)
                is_explicit = 1.0 if m["subtag"] == "explicit" else 0.0
                metadata_boost = (0.6 * is_explicit) + (0.4 * recency_score)
                
                # Final score combines rerank relevance (70%) and metadata boost (30%)
                final_score = (0.7 * rerank_score) + (0.3 * metadata_boost)
                
                # Filter out candidate if it has low semantic relevance (CE < 0.30) AND low keyword overlap (TF-IDF < 0.30)
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
                
            # Calculate metadata boost (60% explicit subtag preference, 40% recency boost)
            is_explicit = 1.0 if m["subtag"] == "explicit" else 0.0
            metadata_boost = (0.6 * is_explicit) + (0.4 * recency_score)
            
            # Final score combines hybrid relevance (70%) and metadata boost (30%)
            final_score = (0.7 * hybrid_score) + (0.3 * metadata_boost)
            final_scored.append((final_score, m))
            
    # 5. Sort by highest final score and return top_k
    final_scored.sort(key=lambda x: x[0], reverse=True)
    top_matches = [item[1] for item in final_scored[:top_k]]
    
    return top_matches

def get_db_status() -> dict:
    """Returns the current status of the vector database engine."""
    try:
        import faiss
        has_faiss_lib = True
    except ImportError:
        has_faiss_lib = False

    try:
        from sentence_transformers import SentenceTransformer
        has_st_lib = True
    except ImportError:
        has_st_lib = False

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

    model_loaded = _model_cache is not None

    return {
        "faiss_available": has_faiss_lib and has_st_lib,
        "faiss_library": "Available" if has_faiss_lib else "Missing",
        "sentence_transformers_library": "Available" if has_st_lib else "Missing",
        "model_loaded": "Yes" if model_loaded else "No",
        "total_records": total_memories,
        "records_with_vectors": memories_with_embeddings,
        "active_users": len(users),
        "memory_tags": tags,
        "engine_mode": "FAISS (Semantic)" if (has_faiss_lib and has_st_lib) else "NumPy Fallback (TF-IDF)"
    }

def warm_up_cache(username: str):
    """No-op fallback for default FAISS database engine."""
    pass

