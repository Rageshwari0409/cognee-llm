import os
import sys

# Add current path to sys.path
sys.path.append(os.path.abspath('.'))

import database_chroma_new as database

def verify_queries():
    database.init_db()
    
    # 1. Get all active users in the DB
    status = database.get_db_status()
    print("Database Status:")
    print(f"Total Records: {status['total_records']}")
    print(f"Active Users: {status['active_users']}")
    print(f"Memory Tags: {status['memory_tags']}")
    
    collection = database._get_chroma_collection()
    if collection is None:
        print("Chroma collection is not initialized.")
        return
        
    # Find active usernames
    try:
        results = collection.get()
        if results and "metadatas" in results and results["metadatas"]:
            usernames = list(set(meta.get("username") for meta in results["metadatas"] if meta.get("username")))
            print(f"Found Usernames: {usernames}")
        else:
            usernames = []
    except Exception as e:
        print(f"Error querying usernames: {e}")
        usernames = []
        
    if not usernames:
        print("No users found in database. Exiting verification.")
        return

    # Use the first username or 'eval_athlete_notebook' if available
    test_user = usernames[0]
    for u in usernames:
        if "notebook" in u or "eval" in u or u == "rishabh":
            test_user = u
            break
            
    print(f"\nRunning test queries for user: '{test_user}' on tag: 'semantic'")
    
    queries = ["food", "frequently", "cherry blossom", "trees are green", "i love it"]
    for q in queries:
        print(f"\n--------------------------------------------------")
        print(f"🔍 QUERY: '{q}'")
        print(f"--------------------------------------------------")
        results = database.vector_query_memories(test_user, "semantic", q)
        if not results:
            print("❌ No records found in semantic memory matching these criteria.")
        else:
            print(f"✅ Found {len(results)} matching records:")
            for idx, r in enumerate(results):
                print(f"  [{idx+1}] Query: '{r['query']}'")
                print(f"      Response: '{r['response']}'")
                print(f"      Subtag: '{r['subtag']}'")

if __name__ == "__main__":
    verify_queries()
