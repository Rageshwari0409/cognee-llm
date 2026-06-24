import sqlite3
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

USERS_DB = os.path.join(BASE_DIR, "users.db")

def verify():
    print("==================================================")
    print("AI WORKOUT COACH - DATABASE VERIFICATION REPORT")
    print("==================================================")
    
    # 1. Inspect Users Database
    if os.path.exists(USERS_DB):
        conn = sqlite3.connect(USERS_DB)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT username FROM users")
            users = cursor.fetchall()
            print(f"\n[1] USERS DATABASE (users.db):")
            print(f"    Total registered athletes: {len(users)}")
            for idx, u in enumerate(users):
                print(f"     - User {idx+1}: {u[0]}")
        except Exception as e:
            print(f"Error reading users.db: {e}")
        finally:
            conn.close()
    else:
        print("\n[1] USERS DATABASE: File 'users.db' does not exist yet (no registrations).")
        
    # 2. Inspect Memories Database in ChromaDB
    try:
        import database_chroma_new as database
        collection = database._get_chroma_collection()
        if collection is not None:
            print(f"\n[2] MEMORIES DATABASE (ChromaDB - {database.CHROMA_PATH}):")
            results = collection.get()
            if results and "ids" in results and results["ids"]:
                explicit_rows = []
                implicit_rows = []
                for idx, str_id in enumerate(results["ids"]):
                    meta = results["metadatas"][idx]
                    row_data = (meta.get("username"), meta.get("query"), meta.get("response"), meta.get("timestamp"), meta.get("tag"))
                    if meta.get("subtag") == "explicit":
                        explicit_rows.append(row_data)
                    else:
                        implicit_rows.append(row_data)
                
                print(f"    - Onboarding/Procedural (Explicit) Memories: {len(explicit_rows)} records found.")
                for r in explicit_rows:
                    print(f"       * User: {r[0]} | Tag: {r[4]} | Question: '{r[1]}' -> Answer: '{r[2]}' ({r[3]})")
                    
                print(f"\n    - Chat Context (Implicit) Memories: {len(implicit_rows)} records found.")
                for r in implicit_rows:
                    print(f"       * User: {r[0]} | Tag: {r[4]} | Prompt: '{r[1]}' -> Memory: '{r[2]}' ({r[3]})")
            else:
                print("\n[2] MEMORIES DATABASE: No memories found in ChromaDB.")
        else:
            print("\n[2] MEMORIES DATABASE: ChromaDB collection is not active.")
    except Exception as e:
        print(f"\n[2] MEMORIES DATABASE: Error reading ChromaDB: {e}")
    
    print("\n==================================================")

if __name__ == "__main__":
    verify()
