import re
import urllib.request
import json
import os
from datetime import datetime

def call_gemini_api(api_key: str, user_query: str, history_context: str = "") -> dict:
    from google import genai
    
    model_name = os.environ.get("GEMINI_MODEL_NAME", "gemini-flash-lite-latest")
    if model_name == "gemini-flash-lite-latest":
        model_name = "gemini-flash-lite-latest"
    
    prompt = f"""You are a professional, encouraging, and intelligent AI Workout Coach.
Analyze the user's query and generate:
1. A highly tailored, high-quality, professional coaching response.
2. Extracted memory logs from the interaction grouped into:
   - "semantic": Universal sports science laws, nutritional facts, and biomechanical principles—such as mechanical tension driving hypertrophy or the necessity of a caloric deficit for fat loss—that serve as the objective, unchanging textbook for structuring any safe and effective training program. General fitness concepts, physiological rules, or facts.
   - "episodic": Your personalized athlete dashboard tracking time-stamped history, containing exact workout logs, personal records, specific injury history, and subjective biofeedback like sleep quality, daily stress levels, and localized muscle soreness from past sessions. User's personal experiences, status, logs, or events.
   - "procedural": The tactical playbook containing step-by-step technical execution manuals, setup cues for movements like the Romanian deadlift, progression protocols, and specific scheduling rules for building structured workout splits like Push/Pull/Legs. Actionable guide steps, workout splitting guidelines, or execution manuals.

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

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.4,
            }
        )
        
        text_content = getattr(response, "text", None) or ""
        if not text_content.strip():
            raise RuntimeError("Empty response from Gemini")
            
        json_match = re.search(r"\{.*\}", text_content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0), strict=False)
        else:
            result = json.loads(text_content, strict=False)
            
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
    except Exception as e:
        print(f"Gemini API Call ({model_name}) error: {e}")
        raise e
            


def get_user_profile_dict(username: str) -> dict:
    """
    Query explicit semantic memories from ChromaDB and compile a profile dictionary
    matching DietCoach's schema.
    """
    import database_chroma_new as database
    memories = database.get_memories_by_tag(username, "semantic")
    explicit_map = {}
    for m in memories:
        if m.get("subtag") == "explicit":
            explicit_map[m["query"]] = m["response"]

    print(f"DEBUG: explicit_map for user '{username}' = {explicit_map}")

    def find_val(keywords, default=None):
        for q, r in explicit_map.items():
            if all(kw in q.lower() for kw in keywords):
                return r
        return default

    # Diet preference mapping
    diet_pref = find_val(["diet preference"], "Non Vegan")
    diet_types = ["vegan"] if diet_pref and diet_pref.lower() == "vegan" else ["non-vegan"]

    # Fitness goal mapping
    primary_goal_raw = find_val(["fitness goal"], "Improve General Fitness & Health")
    fitness_goal = "general_fitness_maintenance"
    if "muscle" in primary_goal_raw.lower() or "strength" in primary_goal_raw.lower():
        fitness_goal = "muscle_gain_hypertrophy"
    elif "loss" in primary_goal_raw.lower() or "weight" in primary_goal_raw.lower():
        fitness_goal = "fat_loss_body_recomposition"

    # Workout frequency mapping
    freq_raw = find_val(["how often", "work out"], "2-3 times a week")
    workout_days = 3
    if "daily" in freq_raw.lower():
        workout_days = 7
    elif "once" in freq_raw.lower():
        workout_days = 1
    elif "rarely" in freq_raw.lower():
        workout_days = 0

    # Focus area & workout types mapping
    focus_raw = find_val(["focus on"], "Full Body Core & Flexibility")
    workout_types = ["strength_hypertrophy"]
    if "cardio" in focus_raw.lower():
        workout_types = ["cardio_endurance"]
    elif "full" in focus_raw.lower() or "core" in focus_raw.lower():
        workout_types = ["strength_hypertrophy", "cardio_endurance"]

    # Weight parsing
    weight_raw = find_val(["weight"], None)
    weight_kg = None
    if weight_raw:
        try:
            nums = re.findall(r"\d+\.?\d*", weight_raw)
            if nums:
                weight_kg = float(nums[0])
        except Exception:
            pass

    # Height parsing
    height_raw = find_val(["height"], None)
    height_cm = None
    if height_raw:
        try:
            nums = re.findall(r"\d+\.?\d*", height_raw)
            if nums:
                height_cm = float(nums[0])
        except Exception:
            pass

    # Country mapping
    country_raw = find_val(["country"], "India")
    country_code = "IN"
    if "united states" in country_raw.lower() or "us" in country_raw.lower():
        country_code = "US"
    elif "united kingdom" in country_raw.lower() or "uk" in country_raw.lower() or "gb" in country_raw.lower():
        country_code = "GB"
    elif "canada" in country_raw.lower() or "ca" in country_raw.lower():
        country_code = "CA"
    elif "australia" in country_raw.lower() or "au" in country_raw.lower():
        country_code = "AU"

    # Allergies parsing (filter out 'none' values)
    allergies_raw = find_val(["allergies"], "None")
    allergies = [a.strip().lower() for a in allergies_raw.split(",") if a.strip() and a.strip().lower() != "none"]

    # Intolerances parsing
    intolerances_raw = find_val(["intolerances"], "None")
    intolerances = [i.strip().lower() for i in intolerances_raw.split(",") if i.strip() and i.strip().lower() != "none"]

    # Meal prep time limit parsing
    max_prep_raw = find_val(["meal prep"], "20 mins")
    max_prep = 20
    try:
        nums = re.findall(r"\d+", max_prep_raw)
        if nums:
            max_prep = int(nums[0])
    except Exception:
        pass

    # Preferred cuisines parsing
    cuisines_raw = find_val(["cuisines"], "Indian, Mediterranean")
    cuisines = [c.strip() for c in cuisines_raw.split(",") if c.strip() and c.strip().lower() != "none"]

    # Typical workout time
    typical_time = find_val(["typical workout time"], "Evening")

    # Extra onboarding questions
    life_stage = find_val(["life stage"], "Working Professional")
    injuries = find_val(["injuries"], "None")
    workout_level = find_val(["workout level"], "Beginner")
    training_location = find_val(["prefer to train"], "At Home")
    workout_duration_limit = find_val(["time can you dedicate"], "30-60 mins")

    # Build the profile dict containing standard and raw fields
    profile = {
        "country": country_code,
        "onboarding_complete": True,
        "life_stage": life_stage,
        "injuries": injuries,
        "workout_level": workout_level,
        "training_location": training_location,
        "workout_duration_limit": workout_duration_limit,
        "diet_constraints": {
            "diet_type": diet_types,
            "allergies": allergies,
            "intolerances": intolerances
        },
        "preferences": {
            "max_prep_minutes": max_prep,
            "cuisines_liked": cuisines
        },
        "goals": {
            "primary": primary_goal_raw.lower()
        },
        "fitness_profile": {
            "workout_types": workout_types,
            "workout_days_per_week": workout_days,
            "typical_workout_time": typical_time.lower() if typical_time else "evening",
            "fitness_goal": fitness_goal,
            "body_weight_kg": weight_kg,
            "height_cm": height_cm
        },
        "raw_onboarding_answers": explicit_map
    }
    return profile


def get_user_memories_string(username: str) -> str:
    """
    Fetch memories from ChromaDB and format them as a single string.
    Each memory is prefixed with its subtag, e.g. '-[explicit]' or '-[implicit]'.
    """
    import database_chroma_new as database
    
    # Get all semantic, episodic, and procedural memories
    sem = database.get_memories_by_tag(username, "semantic")
    epi = database.get_memories_by_tag(username, "episodic")
    pro = database.get_memories_by_tag(username, "procedural")
    
    # Combine them
    all_mems = []
    for m in sem:
        all_mems.append((m.get("timestamp", ""), m.get("subtag", "implicit"), m.get("response", "")))
    for m in epi:
        all_mems.append((m.get("timestamp", ""), m.get("subtag", "implicit"), m.get("response", "")))
    for m in pro:
        all_mems.append((m.get("timestamp", ""), m.get("subtag", "implicit"), m.get("response", "")))
        
    # Sort by timestamp ascending so older context is first and newer is last
    all_mems.sort(key=lambda x: x[0])
    
    # Keep the latest 30 memories to prevent prompt bloating
    all_mems = all_mems[-30:]
    
    lines = []
    for ts, subtag, response in all_mems:
        cleaned_subtag = subtag.strip(" -[]")
        lines.append(f"-[{cleaned_subtag}] {response}")
        
    if not lines:
        return "-[explicit] User has initialized a new profile."
        
    return "\n".join(lines)


def extract_memories(api_key: str, user_query: str, response_text: str) -> dict:
    """
    Call Gemini to extract new memories (semantic, episodic, procedural)
    from the current conversation turn.
    """
    from google import genai
    
    model_name = os.environ.get("GEMINI_MODEL_NAME", "gemini-flash-lite-latest")
    if model_name == "gemini-flash-lite-latest":
        model_name = "gemini-flash-lite-latest"
    
    prompt = f"""You are an advanced cognitive memory compiler for an AI Workout Coach.
Your job is to analyze a single conversation turn between a User and their Coach, and extract key facts/guidelines to save in the Coach's long-term memory database.

Input:
User Query: {user_query}
Coach Response: {response_text}

Rules for Extraction:
1. "semantic": Extract general fitness concepts, physiological rules, nutrition/sports science facts, or hard rules mentioned.
   - Do NOT write personal details here.
   - Every entry MUST be a complete, third-person declarative sentence summarizing the fact (e.g. "The user loves to have yogurt as their main preference" or "Creatine baseline levels are lower for vegetarians"). Do NOT store as a Q&A or conversation fragment.
2. "episodic": Extract the user's personal experiences, workout logs, symptoms, pain/injuries, preferences, or specific physical status mentioned in this turn.
   - Every entry MUST be a complete, third-person declarative sentence summarizing the user's details, choice, or log (e.g. "The user has knee pain when performing squats" or "The user wants to incorporate eggs into their diet"). Do NOT store as a Q&A or conversation fragment.
3. "procedural": Extract actionable step-by-step guides, exercise instructions, split setup rules, or execution manuals mentioned in the coach's response.

Format your output strictly as a JSON object matching this schema (with no markdown formatting or extra text outside the JSON):
{{
  "semantic": ["fact 1", "fact 2"],
  "episodic": ["personal log 1"],
  "procedural": ["action guide step 1"]
}}
"""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.2,
            }
        )
        
        # Safely extract text parts to avoid thought_signature warnings
        text_parts = []
        try:
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)
        except Exception:
            pass
            
        text_content = "".join(text_parts) if text_parts else (getattr(response, "text", None) or "")
        if not text_content.strip():
            raise RuntimeError("Empty response from Gemini")
            
        json_match = re.search(r"\{.*\}", text_content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0), strict=False)
        else:
            result = json.loads(text_content, strict=False)
            
        proc_list = result.get("procedural", []) or result.get("procedics", []) or result.get("procedurals", [])
        return {
            "semantic": result.get("semantic", []) or [],
            "episodic": result.get("episodic", []) or [],
            "procedural": proc_list or []
        }
    except Exception as e:
        print(f"Failed to extract memories using Gemini: {e}")
        return {"semantic": [], "episodic": [], "procedural": []}


def simulate_memory_extraction(user_query: str) -> dict:
    """Fallback offline logic to extract memories based on query keywords."""
    query_lower = user_query.lower()
    memories = {"semantic": [], "episodic": [], "procedural": []}
    
    if any(k in query_lower for k in ["pain", "hurt", "sore", "injury", "knee", "shoulder", "back"]):
        body_part = "knees" if "knee" in query_lower else ("shoulders" if "shoulder" in query_lower else ("lower back" if "back" in query_lower else "joints"))
        memories["episodic"].append(f"User experienced discomfort or pain in the {body_part} during training and was advised to rest.")
        memories["semantic"].append("Ligaments and tendons have lower blood supply than muscles, causing them to recover and adapt slower to load.")
        memories["procedural"].append(f"Joint Pain Recovery Protocol: 1. Reduce workout load immediately. 2. Switch to low-impact exercise variations. 3. Monitor pain levels for 48 hours.")
    elif any(k in query_lower for k in ["squat", "leg", "quad", "glute"]):
        memories["episodic"].append("User queried about lower body training (squats/legs) and is focused on quad/glute development.")
        memories["semantic"].append("Squats target the quadriceps, glutes, and core. Keeping knees in line with toes prevents patellar shear stress.")
        memories["procedural"].append("Barbell Squat Form Guide: 1. Place bar on upper traps. 2. Set feet shoulder-width apart. 3. Hinge at hips and sit back. 4. Keep knees tracking over toes. 5. Push through mid-foot to stand.")
    elif any(k in query_lower for k in ["deadlift", "back", "lats", "pullup", "row"]):
        memories["episodic"].append("User asked about back exercises or deadlift technique, highlighting posterior chain training.")
        memories["semantic"].append("Deadlifts engage the posterior chain (hamstrings, glutes, lower back). A rounded spine under load causes lumbar compression.")
        memories["procedural"].append("Deadlift Form Protocol: 1. Stand with mid-foot under the bar. 2. Bend and grip the bar. 3. Drop hips slightly and flatten back. 4. Pull slack out of the bar. 5. Drive legs into the floor and stand.")
    elif any(k in query_lower for k in ["protein", "eat", "diet", "nutrition", "meal", "calorie", "bulk", "cut"]):
        memories["episodic"].append("User checked nutrition and dietary recommendations, focusing on protein intake or weight goals.")
        memories["semantic"].append("Daily protein intake for muscle building should be 1.6 to 2.2 grams per kilogram of body weight, spread across meals.")
        memories["procedural"].append("Daily Nutrition Setup: 1. Calculate target daily caloric intake. 2. Set protein target (1.8g/kg). 3. Divide protein intake into 4 equal meals. 4. Track hydration (3-4 liters daily).")
    elif any(k in query_lower for k in ["routine", "split", "plan", "program", "schedule", "week"]):
        memories["episodic"].append("User is designing or adjusting their weekly workout schedule and training split.")
        memories["semantic"].append("Muscle groups require 48 to 72 hours of rest between intense training sessions to optimize recovery and growth.")
        memories["procedural"].append("Weekly PPL Split Setup: 1. Day 1: Push (Chest, Shoulders, Triceps). 2. Day 2: Pull (Back, Biceps). 3. Day 3: Legs (Quads, Hamstrings, Calves). 4. Day 4: Rest. 5. Repeat or Rest.")
    else:
        memories["episodic"].append("User initiated conversation about general fitness goals and workout consistency.")
        memories["semantic"].append("Progressive overload (increasing weight, reps, or reducing rest) is required to trigger muscle hypertrophy.")
        memories["procedural"].append("Progressive Overload Application: 1. Keep a workout log. 2. Aim to add 1 rep or small weight increment each session. 3. Maintain strict form.")
        
    return memories


def generate_coach_response(user_query: str, username: str, use_memory: bool = True) -> dict:
    """
    Generates a response from the coach. Instantiates and calls DietCoach
    (integrating user profile and memories fetched live from ChromaDB).
    """
    # 1. Fetch API Key
    api_key = None
    try:
        import streamlit as st
        api_key = st.session_state.get("gemini_api_key")
    except Exception:
        pass

    if not api_key:
        try:
            import streamlit as st
            if "GEMINI_API_KEY" in st.secrets:
                api_key = st.secrets["GEMINI_API_KEY"]
            elif "gemini_api_key" in st.secrets:
                api_key = st.secrets["gemini_api_key"]
        except Exception:
            pass

    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")

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

    # 2. Get user profile and memories from ChromaDB
    if use_memory:
        profile = get_user_profile_dict(username)
        memories_str = get_user_memories_string(username)
    else:
        profile = {}
        memories_str = ""

    # 3. Instantiate and run DietCoach
    from fitness_coach import DietCoach
    
    coach_session = None
    try:
        import streamlit as st
        if "coach_session" not in st.session_state:
            st.session_state["coach_session"] = {
                "coaching_paused": False,
                "pause_reason": None,
                "last_safety_event": None,
                "turn_count": 0
            }
        coach_session = st.session_state["coach_session"]
    except Exception:
        pass
        
    if not coach_session:
        coach_session = {
            "coaching_paused": False,
            "pause_reason": None,
            "last_safety_event": None,
            "turn_count": 0
        }

    # Increment turn count
    coach_session["turn_count"] = coach_session.get("turn_count", 0) + 1

    # Instantiate DietCoach
    coach = DietCoach(api_key=api_key, mock=not bool(api_key))
    
    # Generate reply
    result = coach.chat(
        message=user_query,
        session=coach_session,
        profile=profile,
        memories=memories_str
    )
    
    response_text = result["reply"]

    # 4. Extract memories
    if api_key:
        extracted_memories = extract_memories(api_key, user_query, response_text)
    else:
        extracted_memories = simulate_memory_extraction(user_query)

    return {
        "response": response_text,
        "memories": extracted_memories
    }
