import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
# Workaround for streamlit cloud deployment protobuf descriptors error
# MUST be set before importing streamlit or any other library
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import time
from datetime import datetime

# Import modular components
import auth
import database_chromadb as database
import llm

# ── Cognee exercise knowledge-graph integration (optional) ────────────────────
# Works only in the cognee-2 conda env. The rest of the app is unaffected when
# Cognee is absent — the Knowledge Graph page just hides the Cognee tab.
try:
    import cognee_integration as _cog
    COGNEE_AVAILABLE = _cog.COGNEE_IMPORTABLE
    if not COGNEE_AVAILABLE:
        _COG_ERR = _cog._IMPORT_ERROR
    else:
        _COG_ERR = ""
except Exception as _ce:
    _cog = None
    COGNEE_AVAILABLE = False
    _COG_ERR = str(_ce)

# Set page config
st.set_page_config(
    page_title="AI Workout Coach",
    page_icon="🏋️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Neo4j credentials from Streamlit Secrets into os.environ so that
# cognee_integration.configure_env() picks them up (it reads from os.environ,
# not st.secrets directly).
_NEO4J_SECRET_KEYS = [
    "GRAPH_DATABASE_URL",
    "GRAPH_DATABASE_USERNAME",
    "GRAPH_DATABASE_PASSWORD",
    "GRAPH_DATABASE_NAME",
    "GRAPH_DATABASE_PROVIDER",
]
try:
    for _k in _NEO4J_SECRET_KEYS:
        _v = st.secrets.get(_k) or st.secrets.get(_k.lower())
        if _v:
            os.environ[_k] = str(_v)
except Exception:
    pass

# Inject custom modern CSS
css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("Visual styles stylesheet (styles.css) was not found in the workspace.")

# Initialize session state variables
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Chat Coach"
if "memory_mode" not in st.session_state:
    st.session_state["memory_mode"] = "Using Memory"

# ── Cognee session state (does not conflict with existing keys) ───────────────
if "cognee_data_ready" not in st.session_state:
    st.session_state["cognee_data_ready"] = None   # None=unchecked, True/False
if "cognee_last_result" not in st.session_state:
    st.session_state["cognee_last_result"] = {}    # {answer, scope, search_type, triples, total_tokens}
if "cognee_full_graph_html" not in st.session_state:
    st.session_state["cognee_full_graph_html"] = ""
if "cognee_chat_history" not in st.session_state:
    st.session_state["cognee_chat_history"] = []   # [(question, answer_meta)]

# ------------------------------------------------------------
# AUTHENTICATION PAGE
# ------------------------------------------------------------
def render_auth_page():
    st.markdown("<h1 class='landing-title' style='text-align: center; font-size: 2.6rem !important; font-weight: 800 !important; color: #0F172A !important; margin-top: 0.5rem !important; margin-bottom: 0.25rem !important; letter-spacing: -1.0px !important; line-height: 1.15 !important;'>AI Workout Coach</h1>", unsafe_allow_html=True)
    st.markdown("<p class='landing-subtitle' style='text-align: center; font-size: 1.0rem !important; font-weight: 500 !important; color: #475569 !important; margin-bottom: 1.2rem !important; line-height: 1.3 !important;'>Your smart fitness trainer with vector database memory</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_register = st.tabs(["Sign In", "Sign Up"])
        
        with tab_login:
            st.subheader("Login to Coach")
            login_user = st.text_input("Username", key="login_username").strip()
            login_pass = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Log In", key="login_btn", use_container_width=True):
                if not login_user or not login_pass:
                    st.error("Please enter both username and password.")
                else:
                    if auth.verify_user(login_user, login_pass):
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = login_user
                        database.warm_up_cache(login_user)
                        st.session_state["messages"] = [
                            {"role": "assistant", "text": "Hello, I am your AI Workout Coach. Ask me about exercises, workout splits, form tips, or nutrition! I can store and query your records."}
                        ]
                        st.success("Successfully logged in!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Invalid username or password. Please try again.")
                        
        with tab_register:
            st.subheader("Create New Account")
            reg_user = st.text_input("Choose Username", key="reg_username").strip()
            reg_pass = st.text_input("Password (min 4 characters)", type="password", key="reg_password")
            reg_pass_conf = st.text_input("Confirm Password", type="password", key="reg_password_conf")
            
            st.markdown("---")
            st.markdown("<h3 style='color: #0F172A;'>Profile Setup & Onboarding</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color: #475569; font-size: 1rem; margin-bottom: 15px;'>Please answer these onboarding questions. They will be saved in your Semantic Memory under explicit tags to customize your training plan.</p>", unsafe_allow_html=True)
            
            # Onboarding Questions
            q1 = st.selectbox(
                "1. What is your current life stage? (Select the option that best describes you)",
                options=["Student / Teenager", "Working Professional", "Middle-Aged", "Retired"],
                key="reg_q1"
            )
            q2 = st.selectbox(
                "2. What is your primary fitness goal? (What do you want to achieve?)",
                options=["Weight Loss", "Improve General Fitness & Health", "Build Muscle & Strength", "Prepare for Competition / Sports"],
                key="reg_q2"
            )
            q3 = st.selectbox(
                "3. How often do you currently work out? (Be honest so we can plan accordingly!)",
                options=["Daily", "2-3 times a week", "Once a week", "Rarely / Starting fresh"],
                key="reg_q3"
            )
            q4 = st.multiselect(
                "4. Do you have any injuries or health conditions? (Select all that apply)",
                options=[
                    "No injuries (I'm good to go!)", 
                    "Knee / Joint issues", 
                    "Back / Neck pain", 
                    "Asthma / Respiratory issues", 
                    "Heart condition / Cardiovascular issues", 
                    "High blood pressure (Hypertension)", 
                    "Diabetes", 
                    "Shoulder impingement / issues", 
                    "Herniated disc / spinal issues", 
                    "Arthritis / Joint inflammation", 
                    "Other"
                ],
                default=["No injuries (I'm good to go!)"],
                key="reg_q4"
            )
            
            other_injury = ""
            if "Other" in q4:
                other_injury = st.text_input("Please specify other conditions/injuries:", key="reg_other_injury").strip()
                
            q5 = st.selectbox(
                "5. How would you describe your workout level?",
                options=["Beginner (New to fitness)", "Intermediate (Comfortable with most exercises)", "Advanced / Pro (Experienced and looking for a challenge)"],
                key="reg_q5"
            )
            q6 = st.selectbox(
                "6. Where do you prefer to train?",
                options=["At Home", "At the Gym", "Outdoors"],
                key="reg_q6"
            )
            q7 = st.selectbox(
                "7. Which area would you like to focus on? (Select your main priority)",
                options=["Upper Body (Arms, Chest, Back)", "Lower Body (Legs and Glutes)", "Cardio & Endurance", "Full Body Core & Flexibility"],
                key="reg_q7"
            )
            q8 = st.selectbox(
                "8. How much time can you dedicate to a single workout? (Suggested Question for better planning)",
                options=["15-30 mins", "30-60 mins", "60+ mins"],
                key="reg_q8"
            )
            q9 = st.selectbox(
                "9. What is your diet preference? (Vegan or Non Vegan)",
                options=["Vegan", "Non Vegan"],
                key="reg_q9"
            )
            q10 = st.text_input("10. What is your current height (in cm)?", key="reg_q10").strip()
            q11 = st.text_input("11. What is your current weight (in kg)?", key="reg_q11").strip()
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Register", key="reg_btn", use_container_width=True):
                if not reg_user:
                    st.error("Username cannot be empty.")
                elif reg_pass != reg_pass_conf:
                    st.error("Passwords do not match.")
                elif len(reg_pass) < 4:
                    st.error("Password must be at least 4 characters.")
                elif not q10:
                    st.error("Please enter your current height.")
                elif not q11:
                    st.error("Please enter your current weight.")
                else:
                    success, msg = auth.register_user(reg_user, reg_pass)
                    if success:
                        # Compile injuries answer
                        injuries_list = [item for item in q4 if item != "Other"]
                        if "Other" in q4 and other_injury:
                            injuries_list.append(other_injury)
                        injuries_str = ", ".join(injuries_list) if injuries_list else "None"
                        
                        # Save explicit semantic memories
                        database.save_memory(reg_user, "semantic", "What is your current life stage?", q1, subtag="explicit")
                        database.save_memory(reg_user, "semantic", "What is your primary fitness goal?", q2, subtag="explicit")
                        database.save_memory(reg_user, "semantic", "How often do you currently work out?", q3, subtag="explicit")
                        database.save_memory(reg_user, "semantic", "Do you have any injuries or health conditions?", injuries_str, subtag="explicit")
                        database.save_memory(reg_user, "semantic", "How would you describe your workout level?", q5, subtag="explicit")
                        database.save_memory(reg_user, "semantic", "Where do you prefer to train?", q6, subtag="explicit")
                        database.save_memory(reg_user, "semantic", "Which area would you like to focus on?", q7, subtag="explicit")
                        database.save_memory(reg_user, "semantic", "How much time can you dedicate to a single workout?", q8, subtag="explicit")
                        database.save_memory(reg_user, "semantic", "What is your diet preference?", q9, subtag="explicit")
                        database.save_memory(reg_user, "semantic", "What is your current height (in cm)?", q10, subtag="explicit")
                        database.save_memory(reg_user, "semantic", "What is your current weight (in kg)?", q11, subtag="explicit")
                        
                        st.success("Account created successfully! Please sign in using the Sign In tab.")
                    else:
                        st.error(msg)

# ------------------------------------------------------------
# MAIN APPLICATION PAGE
# ------------------------------------------------------------
def render_main_app():
    username = st.session_state["username"]
    
    # ── SIDEBAR NAVIGATION ──
    st.sidebar.markdown("<div class='logo-container'>", unsafe_allow_html=True)
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=170)
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    
    st.sidebar.markdown(f"<h3 style='text-align: center; margin-top: 0; margin-bottom: 2px; font-size: 2.15rem; color: #FFFFFF;'>AI Workout Coach</h3>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<p style='text-align: center; color: #0284C7; font-weight: 600; margin-bottom: 8px;'>Active Athlete: {username}</p>", unsafe_allow_html=True)
    
    st.sidebar.markdown("<h4 style='padding-left: 10px; margin-top: 5px; margin-bottom: 4px; font-size: 0.95rem; color: #FFFFFF;'>Navigation Dashboard</h4>", unsafe_allow_html=True)
    
    pages = [
        "Chat Coach", 
        "Semantic Memory", 
        "Episodic Memory", 
        "Procedural Memory", 
        "Knowledge Graph", 
        "System Diagnostics"
    ]
    
    for p in pages:
        is_active = st.session_state["current_page"] == p
        nav_class = "sidebar-nav-active" if is_active else "sidebar-nav-inactive"
        st.sidebar.markdown(f"<div class='{nav_class}'>", unsafe_allow_html=True)
        if st.sidebar.button(p, key=f"nav_{p.replace(' ', '_').lower()}", use_container_width=True):
            st.session_state["current_page"] = p
            st.rerun()
        st.sidebar.markdown("</div>", unsafe_allow_html=True)
            
    st.sidebar.markdown("---")
    with st.sidebar.expander("API Configuration"):
        st.text_input("Gemini API Key", type="password", key="gemini_api_key")
    
    st.sidebar.markdown("<div class='sidebar-logout-container'>", unsafe_allow_html=True)
    if st.sidebar.button("Sign Out", key="logout_btn", use_container_width=True):
        st.session_state["logged_in"] = False
        st.session_state["username"] = None
        st.session_state["messages"] = []
        st.session_state["current_page"] = "Chat Coach"
        st.success("Successfully logged out.")
        time.sleep(0.5)
        st.rerun()
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    # ── TOP RIGHT PROFILE SHORTCUT EMOJI & TEXT ──
    col_space, col_profile_icon = st.columns([10, 1.3])
    with col_profile_icon:
        st.markdown(
            """
            <style>
            div[data-testid="stColumn"]:last-child button {
                background-color: transparent !important;
                border: none !important;
                font-size: 1.15rem !important;
                font-weight: 700 !important;
                padding: 0px !important;
                box-shadow: none !important;
                cursor: pointer !important;
                color: #475569 !important;
                text-align: right !important;
                justify-content: flex-end !important;
                display: flex !important;
                align-items: center !important;
                gap: 4px !important;
            }
            div[data-testid="stColumn"]:last-child button:hover {
                color: #0284C7 !important;
                transform: scale(1.05) !important;
                background-color: transparent !important;
                box-shadow: none !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        if st.button("👤 Profile", key="profile_shortcut_btn", help="View / Edit Profile"):
            st.session_state["current_page"] = "👤 Profile"
            st.rerun()

    current_page = st.session_state["current_page"]
    
    if current_page == "Chat Coach":
        render_chat_page(username)
    elif current_page == "👤 Profile":
        render_profile_page(username)
    elif current_page == "Semantic Memory":
        render_memory_page(username, "semantic")
    elif current_page == "Episodic Memory":
        render_memory_page(username, "episodic")
    elif current_page == "Procedural Memory":
        render_memory_page(username, "procedural")
    elif current_page == "Knowledge Graph":
        render_knowledge_graph(username)
    elif current_page == "System Diagnostics":
        render_diagnostics_page()

# ── CHAT COACH PAGE ──
def render_chat_page(username: str):
    st.markdown("<h1 style='font-size: 2.2rem; font-weight: 800; color: #0F172A; margin-top: 0px; margin-bottom: 5px; line-height: 1.1;'>Chat Coach</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.05rem; font-weight: 500; color: #475569; margin-top: 4px; margin-bottom: 15px;'>Interact with your fitness trainer. Ask queries about nutrition, schedules, routines, or symptoms.</p>", unsafe_allow_html=True)
    
    memory_mode = st.radio(
        "Memory Settings",
        options=["Using Memory", "Without Memory"],
        key="memory_mode_selection",
        horizontal=True,
        index=0 if st.session_state["memory_mode"] == "Using Memory" else 1
    )
    st.session_state["memory_mode"] = memory_mode
    
    st.markdown("---")
    
    # Chat History
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-container">
                    <div class="chat-sender sender-user">👤 Human</div>
                    <div class="chat-bubble chat-user">{msg['text']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-container">
                    <div class="chat-sender sender-bot">🤖 Bot</div>
                    <div class="chat-bubble chat-bot">{msg['text']}</div>
                </div>
                """, unsafe_allow_html=True)
                
    user_input = st.chat_input("Ask your workout coach...")
    
    if user_input:
        st.session_state["messages"].append({"role": "user", "text": user_input})
        st.rerun()

    if len(st.session_state["messages"]) > 0 and st.session_state["messages"][-1]["role"] == "user":
        last_query = st.session_state["messages"][-1]["text"]
        
        # Thinking Indicator animation
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("""
        <div class="thinking-box">
            Thinking
            <div class="thinking-dots">
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        time.sleep(1.2)
        
        result = llm.generate_coach_response(last_query, username)
        response_text = result["response"]
        extracted_memories = result["memories"]
        
        thinking_placeholder.empty()
        
        st.session_state["messages"].append({"role": "assistant", "text": response_text})
        
        if st.session_state["memory_mode"] == "Using Memory":
            # Save extracted memories
            for tag, list_of_texts in extracted_memories.items():
                for text in list_of_texts:
                    database.save_memory(username, tag, last_query, text, subtag="implicit")
                    
        st.rerun()

# ── MEMORY DETAIL PAGE ──
def render_memory_page(username: str, tag: str):
    tag_capitalized = tag.capitalize()
    st.markdown(f"<h1 style='font-size: 2.2rem; font-weight: 800; color: #0F172A; margin-top: 0px; margin-bottom: 5px; line-height: 1.1;'>{tag_capitalized} Memory Database</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 1.05rem; font-weight: 500; color: #475569; margin-top: 4px; margin-bottom: 15px;'>Displaying stored {tag} database entries filtered for user <strong>{username}</strong> in chronological order.</p>", unsafe_allow_html=True)
    
    # Semantic Memory Filter UX
    subtag_filter = "All"
    if tag == "semantic":
        st.markdown("<h4 style='margin-top: 15px; margin-bottom: 8px; font-size: 1.45rem; font-weight: 700; color: #0F172A;'>Filter Semantic Memory Classification</h4>", unsafe_allow_html=True)
        subtag_filter = st.radio(
            "Filter Semantic Memory Classification",
            options=["All", "Onboarding Profile (Explicit)", "Chat Context (Implicit)"],
            horizontal=True,
            label_visibility="collapsed"
        )
    
    # Vector Query Search Box
    search_query = st.text_input(f"Search inside {tag} memories using Vector Query...", key=f"search_{tag}")
    
    st.markdown("---")
    
    if search_query.strip():
        memories = database.vector_query_memories(username, tag, search_query.strip())
        st.subheader(f"Vector Query Results for: '{search_query}'")
    else:
        memories = database.get_memories_by_tag(username, tag)
        st.subheader("All Chronological Entries (Newest First)")
        
    if tag == "semantic" and subtag_filter != "All":
        target = "explicit" if "Onboarding" in subtag_filter else "implicit"
        memories = [m for m in memories if m.get("subtag") == target]
        
    if not memories:
        st.info(f"No records found in {tag} memory matching these criteria.")
    else:
        for idx, m in enumerate(memories):
            try:
                dt = datetime.fromisoformat(m["timestamp"])
                formatted_time = dt.strftime("%b %d, %Y - %H:%M:%S")
            except Exception:
                formatted_time = m["timestamp"]
                
            subtag_val = m.get("subtag", "implicit")
            subtag_badge = f'<span class="subtag-badge subtag-{subtag_val}">{subtag_val}</span>'
            
            st.markdown(f"""
            <div class="memory-card">
                <div class="memory-header">
                    <div>
                        <span class="memory-tag tag-{tag}">{tag_capitalized}</span>
                        {subtag_badge}
                    </div>
                    <span class="memory-time">{formatted_time}</span>
                </div>
                <div class="memory-body">
                    <div class="memory-q">Query: {m['query']}</div>
                    <div class="memory-r">Response: {m['response']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── HELPER: resolve the Gemini API key from all sources ──────────────────────
def _get_api_key() -> str:
    key = st.session_state.get("gemini_api_key") or ""
    if not key:
        try:
            key = (st.secrets.get("GEMINI_API_KEY")
                   or st.secrets.get("gemini_api_key") or "")
        except Exception:
            pass
    if not key:
        key = (os.environ.get("GEMINI_API_KEY")
               or os.environ.get("LLM_API_KEY") or "")
    return key.strip()


# Path to the pre-built graph HTML committed in the repo (resolved relative to app.py)
_KG_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".artifacts", "exercises_graph.html")


# ── KNOWLEDGE GRAPH PAGE ──────────────────────────────────────────────────────
def render_knowledge_graph(username: str):
    st.markdown(
        "<h1 style='font-size:2.2rem;font-weight:800;color:#0F172A;"
        "margin-top:0;margin-bottom:5px;line-height:1.1;'>"
        "Interactive Knowledge Graph</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:1.05rem;font-weight:500;color:#475569;"
        "margin-top:4px;margin-bottom:15px;'>"
        "Explore the Cognee exercise knowledge graph.</p>",
        unsafe_allow_html=True,
    )
    import streamlit.components.v1 as components

    # ── Auto-load pre-built graph directly from disk (no Cognee / API key needed) ──
    if not st.session_state.get("cognee_full_graph_html"):
        if os.path.exists(_KG_HTML_PATH):
            try:
                with open(_KG_HTML_PATH, "r", encoding="utf-8") as _f:
                    _html = _f.read()
            except UnicodeDecodeError:
                with open(_KG_HTML_PATH, "r", encoding="cp1252", errors="replace") as _f:
                    _html = _f.read()
            if _html:
                st.session_state["cognee_full_graph_html"] = _html
                st.session_state["cognee_data_ready"] = True
                if COGNEE_AVAILABLE:
                    _cognee_server_state()["ready"] = True
        elif COGNEE_AVAILABLE:
            prebuilt = _cog.get_prebuilt_graph_html()
            if prebuilt:
                st.session_state["cognee_full_graph_html"] = prebuilt
                st.session_state["cognee_data_ready"] = True
                _cognee_server_state()["ready"] = True

    full_html = st.session_state.get("cognee_full_graph_html", "")
    data_ready = bool(full_html)

    if full_html:
        st.success("**Exercise graph is ready.**")
        with st.expander("Full Exercise Knowledge Graph", expanded=True):
            st.caption(
                "Click and drag nodes · scroll to zoom · hover for details. "
                "Use the search box (top-left of the graph) to highlight nodes by name."
            )
            components.html(full_html, height=700, scrolling=True)
    else:
        st.error(
            "Pre-built graph not found. Ensure `.artifacts/exercises_graph.html` "
            "is present in the repository."
        )

    st.markdown("---")

    # ── Q&A requires Cognee + API key ─────────────────────────────────────
    if not COGNEE_AVAILABLE:
        st.info(
            "**Live Q&A is not available** — Cognee is not installed in this environment.\n\n"
            f"Import error: `{_COG_ERR}`"
        )
        return

    api_key = _get_api_key()
    if not api_key:
        st.warning(
            "Enter your **Gemini API key** in the sidebar API Configuration panel "
            "to query the knowledge graph."
        )
        return

    # ── 3. Chat + Context Graph panel ─────────────────────────────────────
    st.markdown(
        "<h3 style='color:#0F172A;margin-bottom:4px;'>Query the Exercise Knowledge Graph</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Ask any fitness question. The right panel shows the knowledge-graph triplets "
        "Cognee retrieved to ground the answer."
    )

    col_chat, col_ctx = st.columns([1, 1], gap="medium")

    with col_chat:
        st.markdown("**Ask a question**")
        question = st.text_input(
            "Exercise question",
            placeholder="e.g. What muscles do sit-ups target?",
            label_visibility="collapsed",
            key="cognee_question_input",
            disabled=not data_ready,
        )
        ask_btn = st.button(
            "Ask ➤",
            key="cognee_ask_btn",
            disabled=not data_ready or not question.strip(),
            use_container_width=False,
        )

        # Example chips
        st.caption("Try an example:")
        examples = [
            "What muscles do sit-ups target?",
            "Tell me instructions for a proper squat",
            "I have a knee injury — what upper-body exercises are safe?",
            "What exercise categories exist in this dataset?",
            "Suggest a beginner ab exercise with no equipment",
        ]
        for ex in examples:
            if st.button(ex, key=f"cog_ex_{ex[:20]}", use_container_width=True):
                st.session_state["cognee_prefill"] = ex
                st.rerun()

        # Handle prefill from example buttons
        if "cognee_prefill" in st.session_state and st.session_state["cognee_prefill"]:
            question = st.session_state.pop("cognee_prefill")
            ask_btn  = True   # treat as if Ask was clicked

    with col_ctx:
        st.markdown("**Context Subgraph**")
        st.caption("Triplets retrieved from the knowledge graph to answer your question. Blue = source · Green = target.")
        ctx_placeholder = st.empty()

    # Process query — render inline in the SAME script run.
    # Calling st.rerun() here would block ~60s on Cognee; by the time it
    # fires, the Streamlit WebSocket has often already dropped and the
    # client never sees the answer.
    if ask_btn and question.strip() and data_ready:
        with st.spinner("Querying exercise knowledge graph…"):
            try:
                result = _cog.query_exercise_graph(question.strip(), api_key)
                st.session_state["cognee_last_result"] = result
                st.session_state["cognee_chat_history"].append(
                    (question.strip(), result)
                )
            except Exception as exc:
                st.error(f"Query failed: {exc}")

    # Render last result
    last = st.session_state.get("cognee_last_result", {})
    if last:
        with col_chat:
            st.markdown("**Answer**")
            st.markdown(
                f"<div style='background:#F0F9FF;border-left:4px solid #0284C7;"
                f"padding:12px 14px;border-radius:6px;margin-top:6px;font-size:0.95rem;'>"
                f"{last['answer']}</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Scope: **{last.get('scope','')}** → `{last.get('search_type','')}` "
                f"| Tokens: {last.get('total_tokens', 0)} "
                f"| Graph triplets: {len(last.get('triples', []))}"
            )

        triples = last.get("triples", [])
        if triples:
            ctx_html = _cog.triples_to_html(triples, height=400)
            with ctx_placeholder:
                components.html(ctx_html, height=450, scrolling=False)
        else:
            ctx_placeholder.info("No graph context retrieved for this query.")
    else:
        ctx_placeholder.markdown(
            "<div style='display:flex;align-items:center;justify-content:center;"
            "height:300px;color:#aaa;font-size:14px;border:2px dashed #e0e0e0;border-radius:8px;'>"
            "Context graph will appear here after you ask a question.</div>",
            unsafe_allow_html=True,
        )

    # ── 4. Chat history (collapsible) ─────────────────────────────────────
    history = st.session_state.get("cognee_chat_history", [])
    if len(history) > 1:
        with st.expander(f"Chat History ({len(history)} questions)", expanded=False):
            for i, (q, r) in enumerate(reversed(history)):
                st.markdown(f"**Q{len(history)-i}:** {q}")
                st.markdown(
                    f"<div style='background:#F8FAFC;border:1px solid #E2E8F0;"
                    f"padding:8px 12px;border-radius:6px;margin-bottom:8px;font-size:0.9rem;'>"
                    f"{r['answer'][:300]}{'…' if len(r['answer'])>300 else ''}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Scope: {r.get('scope','')} | Search: {r.get('search_type','')} | "
                    f"Tokens: {r.get('total_tokens',0)} | Triplets: {len(r.get('triples',[]))}"
                )

    if history:
        if st.button(" Clear Cognee Chat History", key="cog_clear_history"):
            st.session_state["cognee_chat_history"] = []
            st.session_state["cognee_last_result"]  = {}
            st.rerun()


# ── SYSTEM DIAGNOSTICS PAGE ──
def render_diagnostics_page():
    st.markdown("<h1 style='font-size: 2.2rem; font-weight: 800; color: #0F172A; margin-top: 0px; margin-bottom: 5px; line-height: 1.1;'>System Diagnostics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.05rem; font-weight: 500; color: #475569; margin-top: 4px; margin-bottom: 15px;'>Vector Database status, file validation metrics, and active runtime information.</p>", unsafe_allow_html=True)
    
    status = database.get_db_status()
    st.markdown("---")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Database Mode", status["engine_mode"])
    c2.metric("Total Records", f"{status['total_records']:,}")
    c3.metric("Vectorized Records", f"{status['records_with_vectors']:,}")
    c4.metric("Active Athletes", f"{status['active_users']:,}")
    
    st.markdown("### Engine Components Diagnostics")
    
    lib_label = "ChromaDB Library" if "chroma_library" in status else "FAISS Library"
    lib_val = status.get("chroma_library") if "chroma_library" in status else status.get("faiss_library", "Missing")
    
    st.markdown(f"""
    <div class="diagnostic-container">
        <div class="diagnostic-item">
            <div class="diagnostic-label">{lib_label}</div>
            <div class="diagnostic-val">{lib_val}</div>
        </div>
        <div class="diagnostic-item">
            <div class="diagnostic-label">Sentence-Transformers Package</div>
            <div class="diagnostic-val">{status.get('sentence_transformers_library', 'Available')}</div>
        </div>
        <div class="diagnostic-item">
            <div class="diagnostic-label">Embedding Model Loaded</div>
            <div class="diagnostic-val">{status.get('model_loaded', 'Yes')}</div>
        </div>
        <div class="diagnostic-item">
            <div class="diagnostic-label">SQLite File Path</div>
            <div class="diagnostic-val">{database.DB_PATH}</div>
        </div>
        <div class="diagnostic-item">
            <div class="diagnostic-label">Active Memory Tags</div>
            <div class="diagnostic-val">{', '.join(status.get('memory_tags', [])) if status.get('memory_tags') else 'None'}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Interactive Vector Similarity Sandbox")
    st.markdown("Input a search phrase to test the similarity match against all records.")
    
    test_username = st.session_state["username"]
    test_tag = st.selectbox("Select Memory Tag to Query", ["semantic", "episodic", "procedural"])
    test_query = st.text_input("Enter Sandbox Query Test...")
    
    if test_query.strip():
        results = database.vector_query_memories(test_username, test_tag, test_query)
        if not results:
            st.info("No matching records returned from query.")
        else:
            st.write(f"Found {len(results)} matches. Displaying sorted newest first:")
            for idx, r in enumerate(results):
                st.info(f"Match #{idx+1} | Timestamp: {r['timestamp']} | Subtag: {r.get('subtag', 'implicit')}\n\nQuery: {r['query']}\n\nResponse: {r['response']}")

# ── PROFILE PAGE ──
def render_profile_page(username: str):
    st.markdown("<h1 style='font-size: 2.2rem; font-weight: 800; color: #0F172A; margin-top: 0px; margin-bottom: 5px; line-height: 1.1;'>👤 Edit Athlete Profile</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.05rem; font-weight: 500; color: #475569; margin-top: 4px; margin-bottom: 15px;'>Modify your onboarding answers and fitness choices. Your vector database semantic memories will be updated automatically.</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 1. Fetch current profile entries
    memories = database.get_memories_by_tag(username, "semantic")
    explicit_map = {}
    for m in memories:
        if m.get("subtag") == "explicit":
            explicit_map[m["query"]] = m["response"]
            
    # 2. Build form controls pre-populated with existing choices
    
    # Question 1: Life stage
    q1_val = explicit_map.get("What is your current life stage?", "Working Professional")
    q1_options = ["Student / Teenager", "Working Professional", "Middle-Aged", "Retired"]
    q1_index = q1_options.index(q1_val) if q1_val in q1_options else 1
    p_q1 = st.selectbox(
        "1. What is your current life stage?",
        options=q1_options,
        index=q1_index,
        key="prof_q1"
    )
    
    # Question 2: Fitness Goal
    q2_val = explicit_map.get("What is your primary fitness goal?", "Improve General Fitness & Health")
    q2_options = ["Weight Loss", "Improve General Fitness & Health", "Build Muscle & Strength", "Prepare for Competition / Sports"]
    q2_index = q2_options.index(q2_val) if q2_val in q2_options else 1
    p_q2 = st.selectbox(
        "2. What is your primary fitness goal?",
        options=q2_options,
        index=q2_index,
        key="prof_q2"
    )
    
    # Question 3: Frequency
    q3_val = explicit_map.get("How often do you currently work out?", "2-3 times a week")
    q3_options = ["Daily", "2-3 times a week", "Once a week", "Rarely / Starting fresh"]
    q3_index = q3_options.index(q3_val) if q3_val in q3_options else 1
    p_q3 = st.selectbox(
        "3. How often do you currently work out?",
        options=q3_options,
        index=q3_index,
        key="prof_q3"
    )
    
    # Question 4: Injuries
    raw_injuries = explicit_map.get("Do you have any injuries or health conditions?", "No injuries (I'm good to go!)")
    standard_options = [
        "No injuries (I'm good to go!)", 
        "Knee / Joint issues", 
        "Back / Neck pain", 
        "Asthma / Respiratory issues", 
        "Heart condition / Cardiovascular issues", 
        "High blood pressure (Hypertension)", 
        "Diabetes", 
        "Shoulder impingement / issues", 
        "Herniated disc / spinal issues", 
        "Arthritis / Joint inflammation", 
        "Other"
    ]
    
    if raw_injuries == "None" or raw_injuries == "No injuries (I'm good to go!)":
        default_injuries = ["No injuries (I'm good to go!)"]
        other_val = ""
    else:
        items = [item.strip() for item in raw_injuries.split(",") if item.strip()]
        default_injuries = []
        other_items = []
        for item in items:
            if item in standard_options[:-1]: # exclude 'Other'
                default_injuries.append(item)
            else:
                other_items.append(item)
        if other_items:
            default_injuries.append("Other")
            other_val = ", ".join(other_items)
        else:
            other_val = ""
            
    p_q4 = st.multiselect(
        "4. Do you have any injuries or health conditions? (Select all that apply)",
        options=standard_options,
        default=default_injuries,
        key="prof_q4"
    )
    
    p_other_injury = ""
    if "Other" in p_q4:
        p_other_injury = st.text_input("Please specify other conditions/injuries:", value=other_val, key="prof_other_injury").strip()
        
    # Question 5: Workout level
    q5_val = explicit_map.get("How would you describe your workout level?", "Beginner (New to fitness)")
    q5_options = ["Beginner (New to fitness)", "Intermediate (Comfortable with most exercises)", "Advanced / Pro (Experienced and looking for a challenge)"]
    q5_index = q5_options.index(q5_val) if q5_val in q5_options else 0
    p_q5 = st.selectbox(
        "5. How would you describe your workout level?",
        options=q5_options,
        index=q5_index,
        key="prof_q5"
    )
    
    # Question 6: Train location
    q6_val = explicit_map.get("Where do you prefer to train?", "At Home")
    q6_options = ["At Home", "At the Gym", "Outdoors"]
    q6_index = q6_options.index(q6_val) if q6_val in q6_options else 0
    p_q6 = st.selectbox(
        "6. Where do you prefer to train?",
        options=q6_options,
        index=q6_index,
        key="prof_q6"
    )
    
    # Question 7: Focus Area
    q7_val = explicit_map.get("Which area would you like to focus on?", "Full Body Core & Flexibility")
    q7_options = ["Upper Body (Arms, Chest, Back)", "Lower Body (Legs and Glutes)", "Cardio & Endurance", "Full Body Core & Flexibility"]
    q7_index = q7_options.index(q7_val) if q7_val in q7_options else 3
    p_q7 = st.selectbox(
        "7. Which area would you like to focus on?",
        options=q7_options,
        index=q7_index,
        key="prof_q7"
    )
    
    # Question 8: Dedicated Time
    q8_val = explicit_map.get("How much time can you dedicate to a single workout?", "30-60 mins")
    q8_options = ["15-30 mins", "30-60 mins", "60+ mins"]
    q8_index = q8_options.index(q8_val) if q8_val in q8_options else 1
    p_q8 = st.selectbox(
        "8. How much time can you dedicate to a single workout?",
        options=q8_options,
        index=q8_index,
        key="prof_q8"
    )
    
    # Question 9: Diet preference
    q9_val = explicit_map.get("What is your diet preference?", "Non Vegan")
    q9_options = ["Vegan", "Non Vegan"]
    q9_index = q9_options.index(q9_val) if q9_val in q9_options else 1
    p_q9 = st.selectbox(
        "9. What is your diet preference?",
        options=q9_options,
        index=q9_index,
        key="prof_q9"
    )
    
    # Question 10: Height
    q10_val = explicit_map.get("What is your current height (in cm)?", "")
    p_q10 = st.text_input("10. What is your current height (in cm)?", value=q10_val, key="prof_q10").strip()
    
    # Question 11: Weight
    q11_val = explicit_map.get("What is your current weight (in kg)?", "")
    p_q11 = st.text_input("11. What is your current weight (in kg)?", value=q11_val, key="prof_q11").strip()
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Save Profile Settings", key="prof_save_btn", use_container_width=True):
        if not p_q10:
            st.error("Please enter your current height.")
        elif not p_q11:
            st.error("Please enter your current weight.")
        else:
            # Compile injuries answer
            injuries_list = [item for item in p_q4 if item != "Other"]
            if "Other" in p_q4 and p_other_injury:
                injuries_list.append(p_other_injury)
            injuries_str = ", ".join(injuries_list) if injuries_list else "None"
            
            # Save or update explicit memories
            database.save_or_update_explicit_memory(username, "What is your current life stage?", p_q1)
            database.save_or_update_explicit_memory(username, "What is your primary fitness goal?", p_q2)
            database.save_or_update_explicit_memory(username, "How often do you currently work out?", p_q3)
            database.save_or_update_explicit_memory(username, "Do you have any injuries or health conditions?", injuries_str)
            database.save_or_update_explicit_memory(username, "How would you describe your workout level?", p_q5)
            database.save_or_update_explicit_memory(username, "Where do you prefer to train?", p_q6)
            database.save_or_update_explicit_memory(username, "Which area would you like to focus on?", p_q7)
            database.save_or_update_explicit_memory(username, "How much time can you dedicate to a single workout?", p_q8)
            database.save_or_update_explicit_memory(username, "What is your diet preference?", p_q9)
            database.save_or_update_explicit_memory(username, "What is your current height (in cm)?", p_q10)
            database.save_or_update_explicit_memory(username, "What is your current weight (in kg)?", p_q11)
            
            st.success("Profile updated successfully!")
            time.sleep(0.5)
            st.rerun()

# ------------------------------------------------------------
# BOOTSTRAP RUNNER
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI models (first run may take 1–2 minutes)...")
def preload_models_once():
    import threading
    auth.init_auth_db()
    database.init_db()
    try:
        database.get_transformer_model()  # block until the 547 MB nomic model is ready
        threading.Thread(target=database.get_reranker_model, daemon=True).start()
        return True
    except Exception as e:
        print(f"Background pre-load failed: {e}")
        return False


@st.cache_resource(show_spinner=False)
def _cognee_server_state() -> dict:
    """
    Mutable dict cached at the server process level — shared across all user sessions.
    Prevents re-ingestion on every relogin when data already exists in Neo4j/LanceDB.
    Keys: ready (None=unchecked, True=ready, False=failed)
    """
    return {"ready": None}

if __name__ == "__main__":
    preload_models_once()
    
    if not st.session_state["logged_in"]:
        render_auth_page()
    else:
        database.warm_up_cache(st.session_state["username"])
        render_main_app()
