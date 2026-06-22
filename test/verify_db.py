import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USERS_DB = os.path.join(BASE_DIR, "users.db")
MEMORIES_DB = os.path.join(BASE_DIR, "memories.db")


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
        
    # 2. Inspect Memories Database
    if os.path.exists(MEMORIES_DB):
        conn = sqlite3.connect(MEMORIES_DB)
        cursor = conn.cursor()
        try:
            print("\n[2] MEMORIES DATABASE (memories.db):")
            
            # Explicit memories (Onboarding answers)
            cursor.execute("SELECT username, query, response, timestamp FROM memories WHERE subtag = 'explicit'")
            explicit_rows = cursor.fetchall()
            print(f"    - Onboarding (Explicit) Memories: {len(explicit_rows)} records found.")
            for r in explicit_rows:
                print(f"       * User: {r[0]} | Question: '{r[1]}' -> Answer: '{r[2]}' ({r[3]})")
                
            # Implicit memories (Chat logs)
            cursor.execute("SELECT username, tag, query, response, timestamp FROM memories WHERE subtag = 'implicit'")
            implicit_rows = cursor.fetchall()
            print(f"\n    - Chat Context (Implicit) Memories: {len(implicit_rows)} records found.")
            for r in implicit_rows:
                print(f"       * User: {r[0]} | Tag: {r[1]} | Prompt: '{r[2]}' -> Memory: '{r[3]}' ({r[4]})")
                
        except Exception as e:
            print(f"Error reading memories.db: {e}")
        finally:
            conn.close()
    else:
        print("\n[2] MEMORIES DATABASE: File 'memories.db' does not exist yet.")
    
    print("\n==================================================")

if __name__ == "__main__":
    verify()
