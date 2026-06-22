import os
import sys
import time
import json
import sqlite3
from datetime import datetime, timedelta

# Add parent workspace and local test directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import database_chromadb as database
import llm

# Check if required libraries are installed
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevance,
        context_recall,
        context_precision,
    )
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    HAS_RAGAS = True
except ImportError:
    HAS_RAGAS = False

def install_instructions():
    print("\n" + "="*80)
    print("❌ REQUIRED EVALUATION DEPENDENCIES NOT FOUND!")
    print("="*80)
    print("To run Ragas evaluation, you need to install the following packages:")
    print("   pip install ragas datasets langchain-google-genai pandas")
    print("\nEnsure you have configured your Gemini API key in your environment:")
    print("   $env:GEMINI_API_KEY=\"your_key_here\"  (PowerShell)")
    print("   set GEMINI_API_KEY=your_key_here      (CMD)")
    print("="*80 + "\n")

def run_ragas_evaluation():
    if not HAS_RAGAS:
        install_instructions()
        return

    print("="*80)
    print("🌟 STARTING RAGAS EVALUATION FOR AI WORKOUT COACH")
    print("="*80)

    # 1. Retrieve the Gemini API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable is not set.")
        print("Please set it using: $env:GEMINI_API_KEY=\"your-key\" before running.")
        return

    # Load test data from the main evaluation suite
    from test_semantic_search import TEST_MEMORIES, EVAL_QUERIES

    test_user = "ragas_eval_athlete"
    
    # 2. Seed Database
    print("\n1. Seeding evaluation database with 50 memories...")
    conn = sqlite3.connect(database.DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM memories WHERE username = ?", (test_user,))
    conn.commit()
    conn.close()

    now = datetime.now()
    for idx, mem in enumerate(TEST_MEMORIES):
        hours_ago = (idx * 15) % 720
        timestamp = (now - timedelta(hours=hours_ago)).isoformat()
        database.save_memory(
            username=test_user,
            tag="semantic",
            query=mem["q"],
            response=mem["r"],
            subtag=mem["subtag"],
            timestamp=timestamp
        )

    # 3. Gather Evaluation Dataset
    # We will pick 1 paraphrased query per memory (50 queries total) to keep LLM token usage reasonable
    queries_to_eval = []
    seen_targets = set()
    for q_case in EVAL_QUERIES:
        t_id = q_case["target_id"]
        if t_id not in seen_targets:
            queries_to_eval.append(q_case)
            seen_targets.add(t_id)
            
    print(f"\n2. Gathering context and answers from the LLM for {len(queries_to_eval)} test cases...")
    print("   (This will perform database vector query searches and call the Gemini API)")

    eval_data = {
        "question": [],
        "contexts": [],
        "answer": [],
        "ground_truth": []
    }

    # Fetch ground-truth dict
    gt_map = {mem["id"]: mem["r"] for mem in TEST_MEMORIES}

    for idx, q_case in enumerate(queries_to_eval):
        query_text = q_case["query"]
        target_id = q_case["target_id"]
        
        # A. Retrieve contexts
        retrieved = database.vector_query_memories(test_user, tag="semantic", query_text=query_text, top_k=3)
        # Context list should be formatted as list of strings
        context_list = [f"Memory: {m['response']}" for m in retrieved]
        
        # B. Call LLM to generate the answer
        try:
            start_time = time.time()
            llm_response = llm.generate_coach_response(query_text, test_user)
            answer_text = llm_response["response"]
            elapsed = time.time() - start_time
            print(f"   [{idx+1}/{len(queries_to_eval)}] Query: '{query_text[:30]}...' -> Generated in {elapsed:.2f}s")
        except Exception as e:
            print(f"   [{idx+1}/{len(queries_to_eval)}] ⚠️ LLM Call failed for query: '{query_text[:30]}...'. Error: {e}")
            continue

        eval_data["question"].append(query_text)
        eval_data["contexts"].append(context_list if context_list else ["No relevant memories found in database."])
        eval_data["answer"].append(answer_text)
        eval_data["ground_truth"].append(gt_map[target_id])

    # Convert gathered data to a Hugging Face Dataset
    dataset = Dataset.from_dict(eval_data)
    print(f"\nSuccessfully collected {len(dataset)} evaluation samples.")

    # 4. Configure Ragas with Gemini Model
    print("\n3. Configuring Ragas with ChatGoogleGenerativeAI (model: gemini-2.5-flash)...")
    evaluator_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key,
        temperature=0.0
    )
    evaluator_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        api_key=api_key
    )

    ragas_llm = LangchainLLMWrapper(evaluator_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(evaluator_embeddings)

    # 5. Run Evaluation
    print("\n4. Running Ragas Evaluation (computing metrics: Faithfulness, Answer Relevance, Context Precision, Context Recall)...")
    print("   (This runs LLM judge evaluations on Ragas)")
    
    start_eval = time.time()
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevance,
            context_precision,
            context_recall
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )
    eval_duration = time.time() - start_eval
    print(f"\n📊 RAGAS Evaluation Completed in {eval_duration/60:.2f} minutes!")

    # 6. Print Results Report
    print("\n" + "="*80)
    print("RAGAS EVALUATION METRICS REPORT")
    print("="*80)
    
    # Extract scores
    scores = result.scores
    for metric, score in scores.items():
        print(f"📈 {metric.capitalize():<25} : {score:.4f}")
        
    print("-"*80)
    print("Ragas Metric Definitions:")
    print(" - Faithfulness       : How factually correct the generated answer is based on retrieved context (no hallucinations).")
    print(" - Answer Relevance   : How well the generated response directly addresses the user's question.")
    print(" - Context Precision  : Whether the retrieved memories rank relevant facts higher in the search output.")
    print(" - Context Recall     : Whether the retrieved memories contain all the necessary details to answer the question.")
    print("="*80)

    # Save to JSON file
    results_path = os.path.join(current_dir, "ragas_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    print(f"\nSaved Ragas evaluation scores to '{results_path}'")

if __name__ == "__main__":
    run_ragas_evaluation()
