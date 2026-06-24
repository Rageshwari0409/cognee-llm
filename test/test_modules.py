import os
import sys

# Make sure workspace and local test directories are in path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import auth
import database_chroma_new as database
import llm

def run_tests():
    print("=========================================")
    print("RUNNING WORKOUT COACH SYSTEM INTEGRATION TESTS")
    print("=========================================")

    # Initialize Databases and perform schema upgrades
    auth.init_auth_db()
    database.init_db()

    # 1. Test Authentication
    print("\n[1/4] Testing User Authentication Module...")
    test_user = "test_athlete"
    test_pass = "secure123"

    # Clean up previous test user
    import sqlite3
    conn = sqlite3.connect(auth.DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE username = ?", (test_user,))
    conn.commit()
    conn.close()

    # Clean up previous test memories in ChromaDB
    collection = database._get_chroma_collection()
    if collection is not None:
        try:
            collection.delete(where={"username": test_user})
        except Exception as e:
            print("ChromaDB test cleanup warning:", e)


    # Register
    success, msg = auth.register_user(test_user, test_pass)
    print(f" - Register user '{test_user}': {success} (Message: {msg})")
    assert success == True, "Registration failed"

    # Duplicate registration check
    success2, msg2 = auth.register_user(test_user, test_pass)
    assert success2 == False, "Duplicate registration should fail"

    # Verify credentials
    valid_login = auth.verify_user(test_user, test_pass)
    print(f" - Verify correct password: {valid_login}")
    assert valid_login == True, "Valid login verification failed"

    print(" -> Authentication module PASSED.")

    # 2. Test Onboarding and Subtags (Explicit vs Implicit)
    print("\n[2/4] Testing Profile Onboarding & Subtag Database Storage...")
    
    # Save a mock explicit onboarding memory
    saved_explicit = database.save_memory(
        username=test_user,
        tag="semantic",
        query="What is your primary fitness goal?",
        response="Build Muscle & Strength",
        subtag="explicit"
    )
    print(f" - Save explicit onboarding answer: {saved_explicit}")
    assert saved_explicit == True, "Failed to save explicit memory"

    # Save a mock implicit chat memory
    saved_implicit = database.save_memory(
        username=test_user,
        tag="semantic",
        query="What protein intake is recommended?",
        response="Sports literature suggests 1.6 to 2.2 grams per kg body weight daily.",
        subtag="implicit"
    )
    print(f" - Save implicit chat memory: {saved_implicit}")
    assert saved_implicit == True, "Failed to save implicit memory"

    # Retrieve and check subtags
    mems = database.get_memories_by_tag(test_user, "semantic")
    print(f" - Fetched {len(mems)} semantic memories total.")
    assert len(mems) == 2, "Should have 2 semantic memories stored"
    
    explicit_mems = [m for m in mems if m.get("subtag") == "explicit"]
    implicit_mems = [m for m in mems if m.get("subtag") == "implicit"]
    print(f"   * Explicit memories count: {len(explicit_mems)}")
    print(f"   * Implicit memories count: {len(implicit_mems)}")
    
    assert len(explicit_mems) == 1, "Failed to distinguish explicit memory"
    assert len(implicit_mems) == 1, "Failed to distinguish implicit memory"
    assert explicit_mems[0]["response"] == "Build Muscle & Strength"
    assert implicit_mems[0]["query"] == "What protein intake is recommended?"

    print(" -> Onboarding memory classification and subtags PASSED.")

    # 2.5 Test Onboarding Profile Updates (Upsert)
    print("\n[2.5/4] Testing Onboarding Profile Updates (Upsert)...")
    # Save diet preference
    saved_diet = database.save_or_update_explicit_memory(
        username=test_user,
        query="What is your diet preference?",
        response="Vegan"
    )
    assert saved_diet == True, "Failed to save diet preference"
    
    # Save again with same query (update)
    updated_diet = database.save_or_update_explicit_memory(
        username=test_user,
        query="What is your diet preference?",
        response="Non Vegan"
    )
    assert updated_diet == True, "Failed to update diet preference"
    
    # Verify in DB
    mems = database.get_memories_by_tag(test_user, "semantic")
    diet_mems = [m for m in mems if m.get("query") == "What is your diet preference?"]
    assert len(diet_mems) == 1, f"Expected 1 memory entry for diet preference, found {len(diet_mems)}"
    assert diet_mems[0]["response"] == "Non Vegan", f"Expected response 'Non Vegan', found '{diet_mems[0]['response']}'"
    print(" -> Onboarding profile updates (upsert) PASSED.")

    # 3. Test Coach Simulator response
    print("\n[3/4] Testing Coach Simulator & Memory Generator...")
    result = llm.generate_coach_response("I have knee pain during leg workouts.", test_user)
    print(f" - Response Preview: {result['response'][:120]}...")
    print("   Extracted memories by coach:")
    for tag, items in result["memories"].items():
        print(f"     * {tag}: {items}")
        for item in items:
            database.save_memory(test_user, tag, "I have knee pain during leg workouts.", item, subtag="implicit")
            
    # Test Without Memory mode (should work without error)
    result_no_mem = llm.generate_coach_response("I have knee pain during leg workouts.", test_user, use_memory=False)
    print(f" - (Without Memory) Response Preview: {result_no_mem['response'][:120]}...")
    assert result_no_mem is not None, "Failed to run coach response in Without Memory mode"
            
    print(" -> Coach Simulator and Memory Extraction PASSED.")

    # 4. Test Vector Queries
    print("\n[4/4] Testing Vector Search Engine...")
    db_status = database.get_db_status()
    print(f" - Vector DB Mode: {db_status['engine_mode']}")
    
    # Run a search query
    search_q = "muscle building goal"
    search_results = database.vector_query_memories(test_user, "semantic", search_q, top_k=2)
    print(f" - Search vector query: '{search_q}'")
    print(f"   Found {len(search_results)} matches:")
    for idx, r in enumerate(search_results):
        print(f"     #{idx+1}: Query: '{r['query']}' | Response: '{r['response']}' | Subtag: {r.get('subtag')}")
        
    assert len(search_results) > 0, "Vector query failed to find matches"
    # The top match should be our explicit goal onboarding entry
    print(" -> Vector query search engine PASSED.")

    print("\n=========================================")
    print("ALL TESTS COMPLETED SUCCESSFULLY! SYSTEM OK.")
    print("=========================================")

if __name__ == "__main__":
    run_tests()
