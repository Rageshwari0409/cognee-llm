import re
import urllib.request
import json
import os
from datetime import datetime

def call_gemini_api(api_key: str, user_query: str, history_context: str = "") -> dict:
    # Directly define API version and Model name variables as requested
    api_version = os.environ.get("GEMINI_API_VERSION", "v1")
    model_name = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")
    
    prompt = f"""You are a professional, encouraging, and intelligent AI Workout Coach.
Analyze the user's query and generate:
1. A highly tailored, high-quality, professional coaching response.
2. Extracted memory logs from the interaction grouped into:
   - "semantic": General fitness concepts, physiological rules, or facts.
   - "episodic": User's personal experiences, status, logs, or events.
   - "procedural": Actionable guide steps, workout splitting guidelines, or execution manuals.

Rules:
- Topic Guardrail: If the User Query is not related to fitness, workouts, exercises, gym training, sports science, nutrition, diet, muscle building, joint injury recovery, or physical health coaching, DO NOT answer the question. Instead, return a polite refusal stating that you are an AI Workout Coach and can only answer questions related to training, fitness, nutrition, and health. In this case, "memories" should contain empty lists for all tags.
- DO NOT use any emojis in the coaching response text itself.
- Return the output strictly as a JSON object matching this schema:
{{
  "response": "Your detailed coaching response here",
  "memories": {{
    "semantic": ["fact 1", "fact 2"],
    "episodic": ["personal detail 1"],
    "procedural": ["action guide step 1"]
  }}
}}

Relevant Retrieved Memories from database:
{history_context}

User Query: {user_query}
JSON Output:"""

    # Simplified payload without responseMimeType for maximum compatibility
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model_name}:generateContent?key={api_key}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            res_data = response.read().decode("utf-8")
            res_json = json.loads(res_data)
            
            # Parse the JSON response text
            text_content = res_json["candidates"][0]["content"]["parts"][0]["text"]
            
            # Extract JSON using regex (robust to markdown wrapper blocks)
            json_match = re.search(r"\{.*\}", text_content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0), strict=False)
            else:
                result = json.loads(text_content, strict=False)
            
            # Validate structure
            if "response" in result and "memories" in result:
                mems = result["memories"]
                if isinstance(mems, dict):
                    return {
                        "response": result["response"],
                        "memories": {
                            "semantic": mems.get("semantic", []),
                            "episodic": mems.get("episodic", []),
                            "procedural": mems.get("procedural", [])
                        }
                    }
            raise ValueError("Invalid structure returned by Gemini")
    except urllib.error.HTTPError as http_err:
        try:
            error_body = http_err.read().decode("utf-8")
            err_json = json.loads(error_body)
            err_msg = err_json.get("error", {}).get("message", http_err.reason)
            print(f"Gemini API Call ({api_version}/{model_name}) failed: {err_msg}")
        except Exception:
            print(f"Gemini API Call ({api_version}/{model_name}) failed with code {http_err.code}")
        raise http_err
    except Exception as other_err:
        print(f"Gemini API Call ({api_version}/{model_name}) error: {other_err}")
        raise other_err
            


def generate_coach_response(user_query: str, username: str) -> dict:
    """
    Generates a response from the coach. Calls the real Gemini API if an API key is provided
    (integrating user memories fetched from the DB), otherwise falls back to simulated rule-based generation.
    """
    # Try retrieving API key in order of precedence:
    # 1. User manual input from Streamlit Session State (UI override)
    api_key = None
    try:
        import streamlit as st
        api_key = st.session_state.get("gemini_api_key")
    except Exception:
        pass

    # 2. Streamlit Cloud Secrets (standard for deployed Streamlit Cloud apps)
    if not api_key:
        try:
            import streamlit as st
            if "GEMINI_API_KEY" in st.secrets:
                api_key = st.secrets["GEMINI_API_KEY"]
            elif "gemini_api_key" in st.secrets:
                api_key = st.secrets["gemini_api_key"]
        except Exception:
            pass

    # 3. Environment Variables (e.g. system environment)
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")

    # 4. Local .env file fallback (for local development)
    if not api_key:
        try:
            env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            if k.strip() == "GEMINI_API_KEY":
                                os.environ["GEMINI_API_KEY"] = v.strip().strip('"').strip("'")
                                api_key = os.environ["GEMINI_API_KEY"]
                                break
        except Exception:
            pass
            
    if api_key:
        try:
            # 1. Fetch relevant memories for retrieval context
            history_context = ""
            try:
                import database_chromadb as database
                related_semantic = database.vector_query_memories(username, "semantic", user_query, top_k=3)
                related_episodic = database.vector_query_memories(username, "episodic", user_query, top_k=3)
                related_procedural = database.vector_query_memories(username, "procedural", user_query, top_k=3)
                
                context_parts = []
                if related_semantic:
                    context_parts.append("Semantic Knowledge (Facts/Goals):")
                    for m in related_semantic:
                        context_parts.append(f"- Query: {m['query']} | Answer: {m['response']}")
                if related_episodic:
                    context_parts.append("Episodic Memories (Workout details/Injuries/History):")
                    for m in related_episodic:
                        context_parts.append(f"- {m['response']}")
                if related_procedural:
                    context_parts.append("Procedural Guidance (Form protocols/Guides):")
                    for m in related_procedural:
                        context_parts.append(f"- {m['response']}")
                        
                if context_parts:
                    history_context = "\n".join(context_parts)
            except Exception as db_err:
                print(f"Failed to fetch memories for LLM context: {db_err}")
                
            # 2. Call Gemini
            return call_gemini_api(api_key, user_query, history_context)
        except Exception as e:
            print(f"Error calling live Gemini API: {e}. Falling back to simulator.")

    # Fallback to simulated coach
    query_lower = user_query.lower()
    
    # Check if the query is related to the topic of fitness/workouts/health/nutrition/diet
    topic_keywords = [
        "pain", "hurt", "sore", "injury", "knee", "shoulder", "back", "joint", "muscle", "stretch",
        "squat", "leg", "quad", "glute", "deadlift", "lat", "pullup", "row", "bench", "press", "lift",
        "chest", "arm", "bicep", "tricep", "cardio", "run", "walk", "jog", "sprint", "swim", "cycle",
        "protein", "eat", "diet", "nutrition", "meal", "calorie", "bulk", "cut", "fat", "weight", "carb",
        "food", "water", "hydrate", "supplement", "creatine", "vitamin", "sleep", "rest", "recovery",
        "routine", "split", "plan", "program", "schedule", "week", "workout", "gym", "fitness", "exercise",
        "training", "coach", "athlete", "stretch", "warmup", "cooldown", "injuries", "goal", "stage",
        "active", "level"
    ]
    
    is_related = any(k in query_lower for k in topic_keywords)
    
    # Simple greetings are allowed but redirect to fitness
    if query_lower.strip() in ["hello", "hi", "hey", "greetings", "yo"]:
        return {
            "response": "Hello! I am your AI Workout Coach. How can I help you with your fitness, training, or nutrition goals today?",
            "memories": {"semantic": [], "episodic": [], "procedural": []}
        }
        
    if not is_related:
        return {
            "response": "I am your AI Workout Coach, specialized in training guidance, nutrition planning, and injury prevention. I can only assist you with fitness, workout, diet, or health-related questions. Please let me know how I can help you with your physical training!",
            "memories": {"semantic": [], "episodic": [], "procedural": []}
        }
        
    # Initialize default response structure
    response_text = ""
    memories = {
        "semantic": [],
        "episodic": [],
        "procedural": []
    }

    
    # Case 1: Injury / Pain (e.g. Knee pain, shoulder soreness)
    if any(k in query_lower for k in ["pain", "hurt", "sore", "injury", "knee", "shoulder", "back"]):
        # Identify specific body part
        body_part = "joints"
        if "knee" in query_lower:
            body_part = "knees"
        elif "shoulder" in query_lower:
            body_part = "shoulders"
        elif "back" in query_lower:
            body_part = "lower back"
            
        response_text = (
            f"I hear you, and it is crucial to address any discomfort in your {body_part} immediately. "
            "Pain or joint soreness is a signal that your body is reaching its limit, which is often due to "
            "form breakdown, poor mobility, or lack of recovery. "
            "For joint health, remember that ligaments and tendons have a much lower blood supply than muscle, "
            "meaning they take significantly longer to adapt to weight loads and recover from strain. "
            "I highly recommend reducing the load or switching to a low-impact variation until the pain subsides. "
            "Let's make sure we log this session note to closely monitor your recovery in the coming workouts."
        )
        
        # Extract memories
        memories["episodic"].append(
            f"User experienced discomfort or pain in the {body_part} during training and was advised to rest."
        )
        memories["semantic"].append(
            "Ligaments and tendons have lower blood supply than muscles, causing them to recover and adapt slower to load."
        )
        memories["procedural"].append(
            f"Joint Pain Recovery Protocol: 1. Reduce workout load immediately. 2. Switch to low-impact exercise variations. 3. Monitor pain levels for 48 hours."
        )

    # Case 2: Squat / Leg exercises
    elif any(k in query_lower for k in ["squat", "leg", "quad", "glute"]):
        response_text = (
            "Squats are the undisputed king of lower body exercises, primarily targeting the quadriceps, glutes, "
            "and core. Proper execution is critical for both safety and effectiveness. "
            "To perform a squat safely, keep your chest upright, sit back into your hips, and ensure your knees "
            "track in line with your toes rather than caving inward. "
            "Maintaining proper foot stability and pushing through your mid-foot will optimize muscle recruitment. "
            "Since you are focusing on lower body training, let's keep track of your movement patterns and progress."
        )
        
        memories["episodic"].append(
            "User queried about lower body training (squats/legs) and is focused on quad/glute development."
        )
        memories["semantic"].append(
            "Squats target the quadriceps, glutes, and core. Keeping knees in line with toes prevents patellar shear stress."
        )
        memories["procedural"].append(
            "Barbell Squat Form Guide: 1. Place bar on upper traps. 2. Set feet shoulder-width apart. 3. Hinge at hips and sit back. 4. Keep knees tracking over toes. 5. Push through mid-foot to stand."
        )

    # Case 3: Deadlift / Back exercises
    elif any(k in query_lower for k in ["deadlift", "back", "lats", "pullup", "row"]):
        response_text = (
            "A strong back is the foundation of structural strength and posture. Exercises like deadlifts, rows, "
            "and pull-ups recruit major muscle groups including the latissimus dorsi, rhomboids, and erector spinae. "
            "When deadlifting, your spine must remain neutral; do not round your lower back. "
            "Keep the bar close to your shins, drive through your legs, and hinge at the hips. "
            "This ensures the load is safely distributed across the posterior chain rather than the lumbar spine. "
            "Let's document this so we prioritize spinal alignment in your back routines."
        )
        
        memories["episodic"].append(
            "User asked about back exercises or deadlift technique, highlighting posterior chain training."
        )
        memories["semantic"].append(
            "Deadlifts engage the posterior chain (hamstrings, glutes, lower back). A rounded spine under load causes lumbar compression."
        )
        memories["procedural"].append(
            "Deadlift Form Protocol: 1. Stand with mid-foot under the bar. 2. Bend and grip the bar. 3. Drop hips slightly and flatten back. 4. Pull slack out of the bar. 5. Drive legs into the floor and stand."
        )

    # Case 4: Nutrition / Protein / Diet
    elif any(k in query_lower for k in ["protein", "eat", "diet", "nutrition", "meal", "calorie", "bulk", "cut"]):
        response_text = (
            "Nutrition represents the fuel for your workouts and the building blocks for muscle repair. "
            "To build muscle, a daily protein intake of approximately 1.6 to 2.2 grams per kilogram of body weight "
            "is recommended by sports science literature. "
            "Furthermore, muscle synthesis is optimized when protein intake is distributed evenly across 3 to 5 meals "
            "throughout the day. "
            "Whether you are in a caloric deficit for fat loss or a caloric surplus for muscle building, prioritizing "
            "whole foods and adequate protein is essential. I will log this nutrition reference for your coach dashboard."
        )
        
        memories["episodic"].append(
            "User checked nutrition and dietary recommendations, focusing on protein intake or weight goals."
        )
        memories["semantic"].append(
            "Daily protein intake for muscle building should be 1.6 to 2.2 grams per kilogram of body weight, spread across meals."
        )
        memories["procedural"].append(
            "Daily Nutrition Setup: 1. Calculate target daily caloric intake. 2. Set protein target (1.8g/kg). 3. Divide protein intake into 4 equal meals. 4. Track hydration (3-4 liters daily)."
        )

    # Case 5: Routine / Split (e.g. Push Pull Legs, Workout plan)
    elif any(k in query_lower for k in ["routine", "split", "plan", "program", "schedule", "week"]):
        response_text = (
            "Designing a balanced workout split is critical to allow adequate muscle recovery. A standard "
            "Push-Pull-Legs (PPL) split is highly effective. It groups muscles that work together: "
            "chest/shoulders/triceps on Push day, back/biceps on Pull day, and legs/abs on Legs day. "
            "This split ensures each muscle group is trained with sufficient intensity while getting "
            "at least 48 to 72 hours of rest before being worked again. "
            "I recommend a 3 to 6 day weekly training frequency depending on your experience. Let's record this "
            "split structure to guide your scheduling."
        )
        
        memories["episodic"].append(
            "User is designing or adjusting their weekly workout schedule and training split."
        )
        memories["semantic"].append(
            "Muscle groups require 48 to 72 hours of rest between intense training sessions to optimize recovery and growth."
        )
        memories["procedural"].append(
            "Weekly PPL Split Setup: 1. Day 1: Push (Chest, Shoulders, Triceps). 2. Day 2: Pull (Back, Biceps). 3. Day 3: Legs (Quads, Hamstrings, Calves). 4. Day 4: Rest. 5. Repeat or Rest."
        )

    # Default / General Workout Advice
    else:
        response_text = (
            "Consistency and progressive overload are the two core tenets of any successful physical transformation. "
            "Progressive overload means gradually increasing the stress placed on your muscles over time, "
            "which can be achieved by adding weight, increasing repetitions, or shortening rest periods. "
            "Always prioritize correct exercise execution before increasing training intensity. "
            "I'm here to help you refine your workouts, stay consistent, and avoid injury. "
            "Tell me more about your fitness goals so we can customize this journey."
        )
        
        memories["episodic"].append(
            "User initiated conversation about general fitness goals and workout consistency."
        )
        memories["semantic"].append(
            "Progressive overload (increasing weight, reps, or reducing rest) is required to trigger muscle hypertrophy."
        )
        memories["procedural"].append(
            "Progressive Overload Application: 1. Keep a workout log. 2. Aim to add 1 rep or small weight increment each session. 3. Maintain strict form."
        )

    return {
        "response": response_text,
        "memories": memories
    }
