"""
app.py
======
Streamlit frontend for Marginalia — Your literary assistant.
Loads all pipeline artifacts from Hugging Face Dataset on startup.

Run locally:
    streamlit run app.py

Environment variables required (set in HF Spaces secrets):
    HF_TOKEN          — Hugging Face token (read access to private dataset)
    HF_DATASET_REPO   — e.g. "nuria/marginalia-data"
    OPENAI_API_KEY    — OpenAI API key

Part of: Marginalia — Intelligent Personal Literary Analysis System
Bootcamp: Data Science & Machine Learning — Final Project
"""

import logging
import os
import pickle

import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

from notes_manager import NotesManager
from rag_engine import RAGEngine


# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# PAGE CONFIG — must be the first Streamlit call in the script
# =============================================================================
st.set_page_config(
    page_title="Marginalia",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM THEME — Marginalia palette
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=IM+Fell+English:ital@0;1&display=swap');

/* --- Root palette --- */
:root {
    --forest:   #2D4A2D;
    --ochre:    #C8922A;
    --slate:    #8A8FA8;
    --khaki:    #B8B89A;
    --cream:    #F0EBE0;
    --cream-dk: #E2DAC8;
    --ink:      #1C2B1C;
}

/* --- Background --- */
.stApp {
    background-color: var(--cream);
    font-family: 'IM Fell English', Georgia, serif;
    color: var(--ink);
}

/* --- Sidebar --- */
[data-testid="stSidebar"] {
    background-color: var(--forest) !important;
}
[data-testid="stSidebar"] * {
    color: var(--cream) !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: var(--cream) !important;
}

/* --- Headings --- */
h1, h2, h3 {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: var(--forest) !important;
}
h1 { font-size: 2.2rem !important; font-weight: 700 !important; }
h2 { font-size: 1.5rem !important; }
h3 { font-size: 1.2rem !important; }

/* --- Buttons --- */
.stButton > button {
    background-color: var(--forest) !important;
    color: var(--cream) !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'IM Fell English', Georgia, serif !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: background-color 0.2s ease !important;
}
.stButton > button:hover {
    background-color: var(--ochre) !important;
    color: var(--cream) !important;
}

/* --- Text input & text area --- */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: var(--cream-dk) !important;
    border: 1px solid var(--khaki) !important;
    border-radius: 3px !important;
    color: var(--ink) !important;
    font-family: 'IM Fell English', Georgia, serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--forest) !important;
    box-shadow: 0 0 0 1px var(--forest) !important;
}

/* --- Tabs --- */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 2px solid var(--khaki) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 1rem !important;
    color: var(--forest) !important;
    background-color: transparent !important;
    border: none !important;
    padding: 0.5rem 1.5rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--ochre) !important;
    border-bottom: 2px solid var(--ochre) !important;
}

/* --- Cards / expanders --- */
.stExpander {
    background-color: var(--cream-dk) !important;
    border: 1px solid var(--khaki) !important;
    border-radius: 4px !important;
}

/* --- Dividers --- */
hr {
    border-color: var(--khaki) !important;
    opacity: 0.6 !important;
}

/* --- Chat messages --- */
[data-testid="stChatMessage"] {
    background-color: var(--cream-dk) !important;
    border-radius: 4px !important;
    border-left: 3px solid var(--forest) !important;
}

/* --- Scrollbar --- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--cream); }
::-webkit-scrollbar-thumb { background: var(--khaki); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--forest); }

/* --- Input labels --- */
.stTextInput label, 
.stTextArea label,
.stSlider label,
.stDateInput label {
    color: var(--forest) !important;
    font-family: 'IM Fell English', Georgia, serif !important;
    font-size: 0.85rem !important;
    font-style: italic !important;
}

/* --- Placeholder text --- */
::placeholder {
    color: var(--khaki) !important;
    opacity: 1 !important;
}

/* --- Slider — eliminar rojo --- */
[data-testid="stSlider"] > div > div > div > div {
    background-color: var(--forest) !important;
}
[data-testid="stSlider"] > div > div > div > div > div {
    background-color: var(--ochre) !important;
    border-color: var(--ochre) !important;
}

/* --- Date input --- */
.stDateInput > div > div > input {
    background-color: var(--cream-dk) !important;
    color: var(--ink) !important;
    border: 1px solid var(--khaki) !important;
}

/* --- Info / warning / success boxes --- */
[data-testid="stAlert"] {
    background-color: var(--cream-dk) !important;
    border-left: 3px solid var(--forest) !important;
    color: var(--ink) !important;
    border-radius: 3px !important;
}
[data-testid="stAlert"] p {
    color: var(--ink) !important;
}

/* --- Save note button — full width override --- */
[data-testid="stFormSubmitButton"] > button,
[data-testid="stBaseButton-secondaryFormSubmit"] > button {
    background-color: var(--forest) !important;
    color: var(--cream) !important;
    width: 100% !important;
    border-radius: 3px !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background-color: var(--ochre) !important;
}

</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================
# In HF Spaces, secrets are injected as environment variables.
# Locally you can export them in your shell before running streamlit:
#   export HF_TOKEN="hf_..."
#   export HF_DATASET_REPO="your-username/literary-profile-data"
#   export OPENAI_API_KEY="sk-..."

HF_TOKEN        = os.environ.get("HF_TOKEN", "")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "")
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")

# Files that must be downloaded from HF Dataset at startup
ARTIFACT_FILES = [
    "bert_embeddings.pkl",
    "catalog_embeddings.pkl",
    "cluster_model.pkl",
    "reader_profile.pkl",
    "recommendations.pkl",
    "tfidf_matrix.pkl",
    "tfidf_vectorizer.pkl",
    "user_notes_clustered.csv",
    "books_clean.csv",  # catalog
]

# Local directory where downloaded files are cached inside the HF Space.
# /tmp is always writable in HF Spaces (unlike the app root).
DATA_DIR = "/tmp/literary_data"

# Each cluster gets a fixed color and icon for visual consistency
# across sidebar, recommendation cards, and chat.
# These are purely cosmetic — they do not affect pipeline logic.
CLUSTER_STYLE = {
    0: {"color": "#2D4A2D", "icon": "◈"},   # Inner Worlds — forest
    1: {"color": "#C8922A", "icon": "◈"},   # Speculative Minds — ochre
    2: {"color": "#8A8FA8", "icon": "◈"},   # Political & Dystopian — slate
    3: {"color": "#B8B89A", "icon": "◈"},   # The Seeker's Path — khaki
    4: {"color": "#7A6A4A", "icon": "◈"},   # The Dark & The Strange - warm brown
}

# Suggested questions shown as clickable buttons above the chat input
SUGGESTED_QUESTIONS = [
    "What recurring themes appear most in my reading notes?",
    "Recommend books similar to what I enjoy most.",
    "What ideas from my past readings might I have forgotten?",
    "¿Qué libros me recomendarías para seguir explorando mis temas favoritos?",
]


# =============================================================================
# DATA LOADING — cached so each function runs only once per app session
# =============================================================================

@st.cache_resource(show_spinner="Loading your reading profile...")
def load_artifacts() -> dict:
    """
    Downloads all pipeline artifacts from HF Dataset and loads them into memory.

    @st.cache_resource ensures this runs only once per session — not on every
    user interaction. Without caching, loading ~28 MB of embeddings + the BERT
    model would happen on every button click.

    Returns:
        dict with all loaded objects keyed by filename without extension.
        Example keys: "reader_profile", "recommendations", "user_notes_clustered"
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    artifacts = {}

    for filename in ARTIFACT_FILES:
        local_path = os.path.join(DATA_DIR, filename)

        # Download from HF Dataset only if not already cached locally.
        # On first startup this downloads all files; on restarts within
        # the same Space instance the files are already in /tmp.
        if not os.path.exists(local_path):
            logger.info("Downloading %s from HF Dataset...", filename)
            hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=filename,
                repo_type="dataset",
                token=HF_TOKEN,
                local_dir=DATA_DIR,
            )
            logger.info("Downloaded: %s", filename)

        # Load into memory
        if filename.endswith(".pkl"):
            with open(local_path, "rb") as f:
                key = filename.replace(".pkl", "")
                artifacts[key] = pickle.load(f)
                logger.info("Loaded pickle: %s", filename)

        elif filename.endswith(".csv"):
            key = filename.replace(".csv", "")
            artifacts[key] = pd.read_csv(local_path)
            logger.info(
                "Loaded CSV: %s  (%d rows)", filename, len(artifacts[key])
            )

    return artifacts


@st.cache_resource(show_spinner="Initializing RAG engine...")
def load_rag_engine() -> RAGEngine:
    """
    Initializes the RAGEngine once per session.
    Cached separately from artifacts because RAGEngine holds the BERT model
    in memory — the heaviest object to initialize (~1-2 min on first load).

    Returns:
        Initialized RAGEngine instance ready to call .ask()
    """
    return RAGEngine(drive_path=DATA_DIR, openai_api_key=OPENAI_API_KEY)


def load_notes_manager() -> NotesManager:
    """
    Returns a fresh NotesManager pointing to DATA_DIR.

    NOT cached — NotesManager must always read the current state of
    user_notes.csv from disk so newly added notes are reflected immediately.

    Returns:
        NotesManager instance with current notes loaded.
    """
    return NotesManager(drive_path=DATA_DIR)


# =============================================================================
# SIDEBAR — READER PROFILE
# =============================================================================

def render_sidebar(artifacts: dict):
    """
    Renders the persistent reader profile panel in the sidebar.
    Shows the clusters with label, books, theme keywords, and avg rating.
    Visible on all tabs simultaneously.
    """
    reader_profile = artifacts["reader_profile"]
    bg_map = {
        0: "#1E3320",   # forest green oscuro
        1: "#7A5518",   # ochre oscuro
        2: "#4A4E5E",   # slate oscuro
        3: "#6B6B52",   # khaki oscuro
        4: "#4A3E2A",   # warm brown oscuro
    }
    
    clusters_html = ""
    for i, (key, cluster) in enumerate(reader_profile.items()):
        cid    = cluster["cluster_id"]
        label  = cluster.get("label", f"Cluster {cid}")
        books  = cluster.get("books", [])
        words  = cluster.get("top_words", [])[:5]
        rating = cluster.get("avg_rating", "-")
        bg     = bg_map.get(cid, "#2D4A2D")

        books_str = " · ".join(books) if books else ""
        tags_str  = "".join(
            f'<span style="background:#FFFFFF22;color:#F0EBE0;'
            f'border:1px solid #FFFFFF44;border-radius:4px;'
            f'padding:1px 7px;font-size:0.72em;margin:2px;'
            f'display:inline-block;">{w}</span>'
            for w in words
        )

        clusters_html += (
            f'<div style="background:{bg};border-radius:6px;'
            f'padding:10px 12px;margin-bottom:8px;">'
            f'<div style="color:#F0EBE0;font-weight:700;font-size:0.95em;'
            f'font-family:\'Playfair Display\',Georgia,serif;'
            f'margin-bottom:5px;">{label}</div>'
            f'<div style="color:#C8D8C8;font-size:0.76em;font-style:italic;'
            f'margin-bottom:6px;">{books_str}</div>'
            f'<div style="margin-bottom:6px;">{tags_str}</div>'
            f'<div style="color:#C8B890;font-size:0.73em;">'
            f'Avg rating: {rating} / 5.0</div>'
            f'</div>'
        )

    footer_html = (
        '<div style="border-top:1px solid #4A6A4A;margin-top:8px;'
        'padding-top:12px;text-align:center;">'
        '<div style="display:flex;justify-content:space-around;margin-bottom:8px;">'
        '<div style="text-align:center;">'
        '<div style="color:#C8922A;font-weight:700;font-size:1.05em;">50</div>'
        '<div style="color:#C8D8C8;font-size:0.68em;">notes</div>'
        '</div>'
        '<div style="text-align:center;">'
        '<div style="color:#C8922A;font-weight:700;font-size:1.05em;">78,811</div>'
        '<div style="color:#C8D8C8;font-size:0.68em;">books</div>'
        '</div>'
        '<div style="text-align:center;">'
        '<div style="color:#C8922A;font-weight:700;font-size:1.05em;">K=5</div>'
        '<div style="color:#C8D8C8;font-size:0.68em;">clusters</div>'
        '</div>'
        '</div>'
        '<div style="color:#8A8FA8;font-size:0.70em;font-style:italic;">'
        'Marginalia — your literary assistant'
        '</div>'
        '</div>'
    )

    with st.sidebar:
        st.markdown("## Your Reading Profile")
        st.markdown(
            "*Built from your reading notes — your words, emotions, "
            "and ideas distilled into a portrait of who you are as a reader.*"
        )
        st.divider()
        st.markdown(clusters_html + footer_html, unsafe_allow_html=True)


# =============================================================================
# TAB 1 — RECOMMENDATIONS
# =============================================================================

def render_recommendations(artifacts: dict):
    """
    Renders pre-calculated top-10 book recommendations per cluster.
    Uses recommendations.pkl from Phase 5 — no real-time computation needed.

    Args:
        artifacts: dict returned by load_artifacts()
    """
    recommendations = artifacts["recommendations"]
    reader_profile  = artifacts["reader_profile"]

    st.markdown("## Recommended for You")
    st.markdown(
        "Below are the top 6 books selected for each of your reading clusters — "
        "groups of books that share emotional and thematic patterns drawn from your "
        "personal notes. Each recommendation was chosen by comparing the semantic "
        "fingerprint of your cluster against 79,000 titles, surfacing the books most "
        "likely to resonate with how you actually read and what moves you."
    )
    st.markdown(
        "<div style='border-left:3px solid #B8B89A;padding-left:12px;"
        "margin-top:8px;color:#6B6B52;font-style:italic;font-size:0.9em;'>"
        "A continuación encontrarás los 6 libros más recomendados para cada uno de "
        "tus clusters de lectura — grupos de libros que comparten patrones emocionales "
        "y temáticos extraídos de tus notas personales. Cada recomendación fue "
        "seleccionada comparando la huella semántica de tu cluster contra 79,000 "
        "títulos, identificando los libros con mayor probabilidad de resonar con tu "
        "forma de leer y lo que te mueve."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    
    for cluster_key, rec_list in recommendations.items():
        if not rec_list:
            continue

        # Get cluster metadata from reader_profile for label and style
        cluster_info = reader_profile.get(cluster_key, {})
        cid          = cluster_info.get("cluster_id", 0)
        label        = cluster_info.get("label", cluster_key)
        style        = CLUSTER_STYLE.get(cid, {"color": "#555", "icon": "•"})
        top_words    = cluster_info.get("top_words", [])[:4]

        # Section header
        st.markdown(
            f"<h3 style='color:{style['color']}; margin-bottom:4px;'>"
            f"{style['icon']} {label}</h3>",
            unsafe_allow_html=True,
        )
        if top_words:
            st.caption("Themes: " + "  ·  ".join(top_words))

        st.markdown("<br>", unsafe_allow_html=True)

        # Book cards in 2 columns — top 6 books per cluster.
        # i % 2 alternates between cols[0] and cols[1]:
        # i=0 → col 0, i=1 → col 1, i=2 → col 0, i=3 → col 1 ...
        cols = st.columns(2)

        for i, rec in enumerate(rec_list[:6]):
            col   = cols[i % 2]
            score = rec.get("similarity_score", 0)

            with col:
                # Color hex + "0A" = ~4% opacity — subtle background tint
                st.markdown(
                    f"<div style='"
                    f"border-left:3px solid {style['color']};"
                    f"padding:12px 16px;"
                    f"margin-bottom:12px;"
                    f"background:{style['color']}0A;"
                    f"border-radius:0 6px 6px 0;'>"
                
                    f"<div style='font-weight:600; font-size:0.95em; margin-bottom:2px;'>"
                    f"#{rec['rank']} {rec['title']}</div>"
                
                    f"<div style='color:{style['color']}; font-size:0.82em; "
                    f"font-style:italic; margin-bottom:6px;'>"
                    f"{str(rec.get('author_names', ''))[:80]}{'...' if len(str(rec.get('author_names', ''))) > 80 else ''}</div>"
                
                    f"<div style='font-size:0.82em; color:#5A5A4A; margin-bottom:8px; "
                    f"line-height:1.5;'>"
                    f"{str(rec.get('description', ''))}</div>"
                
                    f"<div style='display:flex; align-items:center; gap:8px;'>"
                    f"<div style='"
                    f"height:3px; width:{int(score * 100)}%;"
                    f"background:{style['color']}; border-radius:2px;'></div>"
                    f"<span style='font-size:0.73em; color:{style['color']};'>"
                    f"{score:.2f}</span>"
                    f"</div>"
                
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.divider()


# =============================================================================
# TAB 2 — ASK YOUR LIBRARY (RAG CHAT)
# =============================================================================

def render_chat(rag: RAGEngine):
    """
    Renders the conversational RAG interface.
    Manages chat history via st.session_state so messages persist
    across Streamlit re-runs within the same browser session.

    Args:
        rag: initialized RAGEngine instance from load_rag_engine()
    """
    st.markdown("## Ask Your Library")
    st.markdown(
        "Ask anything about your reading history, patterns, or "
        "get personalized recommendations. Works in English and Spanish."
    )

    # ------------------------------------------------------------------
    # SESSION STATE INITIALIZATION
    # ------------------------------------------------------------------
    # st.session_state persists across re-runs for this user session.
    # We initialize the keys only on the very first run (not found in dict).
    # After that we just append to them — Streamlit keeps them in memory.

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # pending_query stores a suggested question clicked by the user.
    # We cannot submit it directly in the button callback because Streamlit
    # processes the full script before rendering. We store it here and
    # consume it in the query-processing block below.
    if "pending_query" not in st.session_state:
        st.session_state["pending_query"] = ""

    # ------------------------------------------------------------------
    # SUGGESTED QUESTIONS
    # ------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Try one of these:")

    suggestion_cols = st.columns(len(SUGGESTED_QUESTIONS))

    for i, question in enumerate(SUGGESTED_QUESTIONS):
        with suggestion_cols[i]:
            label = question[:40] + "..." if len(question) > 40 else question
            if st.button(label, key=f"suggestion_{i}", use_container_width=True):
                st.session_state["pending_query"] = question

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # CHAT HISTORY DISPLAY
    # ------------------------------------------------------------------
    # We replay the full history on every re-run — Streamlit is stateless,
    # so everything is rendered from scratch from session_state each time.

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ------------------------------------------------------------------
    # CHAT INPUT + QUERY PROCESSING
    # ------------------------------------------------------------------
    # st.chat_input() renders a fixed input bar at the bottom of the page.
    # Returns the submitted text, or None if nothing was submitted this run.

    user_input = st.chat_input("Ask about your reading history...")

    # Determine the active query this run:
    # Priority 1 — a suggestion button was clicked (pending_query is set)
    # Priority 2 — the user typed and submitted in the chat input
    query = st.session_state["pending_query"] or user_input

    # Clear pending_query immediately so it does not re-trigger next run
    if st.session_state["pending_query"]:
        st.session_state["pending_query"] = ""

    if query:
        # 1. Add user message to history and render it
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # 2. Call the RAG engine
        # st.spinner() shows a loading indicator while GPT-4o responds (~3-8s)
        with st.chat_message("assistant"):
            with st.spinner("Reading your notes..."):
                try:
                    response = rag.ask(query)
                except Exception as e:
                    # Never show a raw traceback to the user.
                    # Log the full error for debugging, show a clean message.
                    logger.error("RAG error for query '%s': %s", query, e)
                    response = (
                        "Something went wrong while processing your question. "
                        "Please try again in a moment."
                    )

            st.markdown(response)

        # 3. Save assistant response to history
        st.session_state["messages"].append(
            {"role": "assistant", "content": response}
        )

        # 4. Force a re-run so the new messages appear in the history section
        # rendered above. Without this, the bubbles would not update until
        # the next user interaction.
        st.rerun()

    # ------------------------------------------------------------------
    # CLEAR CONVERSATION BUTTON
    # ------------------------------------------------------------------
    if st.session_state["messages"]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state["messages"] = []
            st.rerun()


# =============================================================================
# TAB 3 — ADD A NOTE
# =============================================================================

def render_add_note():
    """
    Renders a form for adding new reading notes via NotesManager.
    Notes are saved to user_notes.csv in DATA_DIR.

    Note on persistence: in HF Spaces, /tmp is ephemeral — notes added here
    do not persist between Space restarts. The 40 base notes live in the
    HF Dataset and are loaded at startup. This tab is useful for demo
    purposes and for testing the NotesManager integration.
    """
    st.markdown("## Add a Reading Note")
    st.markdown(
        "Record your reflections on a book. "
        "Notes added here feed into future pipeline updates."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    nm = load_notes_manager()

    # ------------------------------------------------------------------
    # INPUT FORM
    # ------------------------------------------------------------------
    # st.form() groups all inputs into a single submission block.
    # The re-run is triggered only when Submit is pressed — not on every
    # keystroke. This prevents unnecessary re-runs while the user is typing.
    # clear_on_submit=True resets all fields after a successful submission.

    with st.form("add_note_form", clear_on_submit=True):
        col1, col2 = st.columns(2)

        with col1:
            title = st.text_input(
                "Book title *",
                placeholder="e.g. The Brothers Karamazov",
            )
            author = st.text_input(
                "Author *",
                placeholder="e.g. Fyodor Dostoevsky",
            )
            rating = st.slider(
                "Rating *",
                min_value=1,
                max_value=5,
                value=4,
                help="1 = did not connect with it, 5 = life-changing",
            )

        with col2:
            tags = st.text_input(
                "Tags",
                placeholder="e.g. philosophy, identity, guilt  (optional)",
            )
            date_read = st.date_input(
                "Date read",
                help="Approximate date is fine. Defaults to today.",
            )

        note_text = st.text_area(
            "Your reflection *",
            placeholder=(
                "Write your personal thoughts about this book. "
                "What moved you? What ideas stayed with you? "
                "Minimum 20 characters."
            ),
            height=160,
        )

        submitted = st.form_submit_button(
            "Save note",
            use_container_width=True,
            type="primary",
        )

    # ------------------------------------------------------------------
    # FORM SUBMISSION HANDLING
    # ------------------------------------------------------------------
    # This block runs only when the user clicks "Save note".
    # NotesManager.add_note() handles all field validation internally
    # and returns True on success, False on validation failure.

    if submitted:
        success = nm.add_note(
            title=title,
            author=author,
            note_text=note_text,
            rating=int(rating),
            tags=tags,
            date_read=str(date_read),
        )

        if success:
            st.success(
                f"Note for '{title}' saved successfully. "
                "Re-run Phases 3-5 in Colab to update your profile."
            )
        else:
            # NotesManager logs the specific validation error via logging.
            # We show a generic message here for clean UX.
            st.error(
                "Could not save the note. Please check that all required "
                "fields are filled and the note is at least 20 characters."
            )

    # ------------------------------------------------------------------
    # EXISTING NOTES TABLE
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("### Your notes so far")

    df_notes = nm.get_notes_dataframe()

    if len(df_notes) == 0:
        st.info("No notes yet. Add your first one above.")
    else:
        display_cols = ["title", "author", "rating", "tags", "date_read"]
        available    = [c for c in display_cols if c in df_notes.columns]

        st.dataframe(
            df_notes[available].sort_values("date_read", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"{len(df_notes)} notes total.")


# =============================================================================
# MAIN — APP ASSEMBLY
# =============================================================================

def main():
    """
    Entry point for the Streamlit app.
    Validates secrets, loads all resources, and routes to the correct tab.

    Layout:
        Sidebar  — persistent reader profile (all 4 clusters, always visible)
        Tab 1    — Recommendations (pre-calculated top-10 per cluster)
        Tab 2    — Ask your library (RAG conversational chat)
        Tab 3    — Add a note (NotesManager form)
    """
    # ------------------------------------------------------------------
    # GUARD: validate required secrets before loading anything heavy
    # ------------------------------------------------------------------
    # If secrets are missing, we stop early with a clear message instead
    # of letting the app crash deep inside load_artifacts() or RAGEngine.

    missing = []
    if not HF_TOKEN:
        missing.append("HF_TOKEN")
    if not HF_DATASET_REPO:
        missing.append("HF_DATASET_REPO")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")

    if missing:
        st.error(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Set them in HF Spaces > Settings > Repository secrets."
        )
        st.stop()
        # st.stop() halts execution — nothing below runs.
        # Without it the app would continue and crash with a cryptic error.

    # ------------------------------------------------------------------
    # LOAD ALL RESOURCES
    # ------------------------------------------------------------------
    # Both calls are cached — they execute only once per session.
    # First load: ~1-2 min (BERT model download + artifact loading).
    # Subsequent interactions: instant (served from @st.cache_resource).

    artifacts = load_artifacts()
    rag = load_rag_engine()

    # ------------------------------------------------------------------
    # Logo banner
    # ------------------------------------------------------------------
    logo_path = os.path.join(DATA_DIR, "marginalia_logo.png")
    if not os.path.exists(logo_path):
        hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename="marginalia_logo.png",
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir=DATA_DIR,
        )
    if os.path.exists(logo_path):
        st.image(logo_path, width=380)
        st.markdown("<br>", unsafe_allow_html=True)
    
    # ------------------------------------------------------------------
    # SIDEBAR
    # ------------------------------------------------------------------
    render_sidebar(artifacts)

    # ------------------------------------------------------------------
    # MAIN CONTENT — TABS
    # ------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs([
        "Recommendations",
        "Ask your library",
        "Add a note",
    ])

    with tab1:
        render_recommendations(artifacts)

    with tab2:
        render_chat(rag)

    with tab3:
        render_add_note()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
