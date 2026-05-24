import streamlit as st
import json
import numpy as np
import sys
import os
from collections import Counter
import pandas as pd

sys.path.append("src")

from fetch_patents import load_patents, get_sample_patents, save_patents
from embedder import load_model, create_embeddings, build_faiss_index, search_patents, load_index, save_index

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="PatentSight AI",
    page_icon="✈️",
    layout="wide"
)

# ─────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1A3C6E;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-top: 0;
    }
    .patent-card {
        background: #f8f9fa;
        border-left: 4px solid #1A3C6E;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .patent-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1A3C6E;
    }
    .patent-meta {
        font-size: 0.82rem;
        color: #888;
        margin-top: 2px;
    }
    .score-badge {
        background: #e8f0fe;
        color: #1A3C6E;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .history-item {
        background: #f0f4ff;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.82rem;
        color: #333;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SESSION STATE — keeps data across reruns
# ─────────────────────────────────────────
# This is how Streamlit remembers things between searches
if "search_history" not in st.session_state:
    st.session_state.search_history = []

if "query" not in st.session_state:
    st.session_state.query = ""


# ─────────────────────────────────────────
# LOAD MODEL + DATA
# ─────────────────────────────────────────
@st.cache_resource
def load_ai_model():
    return load_model()

@st.cache_resource
def load_data_and_index():
    if os.path.exists("data/patents.json"):
        with open("data/patents.json", "r") as f:
            patents = json.load(f)
    else:
        patents = get_sample_patents()
        save_patents(patents)

    model = load_ai_model()
    index = load_index("data/faiss_index.bin")

    if index is None:
        embeddings = create_embeddings(patents, model)
        index = build_faiss_index(embeddings)
        save_index(index)

    return patents, index


# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<p class="main-title">✈️ PatentSight AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Aerospace Patent Intelligence Platform — Semantic Search powered by AI</p>', unsafe_allow_html=True)
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("🤖 sentence-transformers + FAISS")

st.divider()

# ─────────────────────────────────────────
# LOAD EVERYTHING
# ─────────────────────────────────────────
with st.spinner("Loading AI model and patent index..."):
    model = load_ai_model()
    patents, index = load_data_and_index()


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Search Settings")
    top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)

    st.markdown("---")

    # DATABASE STATS
    st.markdown("### 📊 Database Stats")
    st.metric("Total Patents", len(patents))
    st.metric("Domain", "Aerospace / UAV")
    st.metric("AI Model", "MiniLM-L6-v2")

    st.markdown("---")

    # PATENTS BY YEAR CHART
    st.markdown("### 📅 Patents by Year")
    years = [p["year"] for p in patents if p.get("year")]
    year_counts = Counter(years)
    year_df = pd.DataFrame({
        "Year": list(year_counts.keys()),
        "Patents": list(year_counts.values())
    }).sort_values("Year")
    st.bar_chart(year_df.set_index("Year"))

    st.markdown("---")

    # EXAMPLE QUERIES
    st.markdown("### 💡 Example Queries")
    examples = [
        "drone autonomous navigation",
        "fuel efficient jet propulsion",
        "satellite attitude control",
        "UAV swarm intelligence",
        "aircraft predictive maintenance"
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.query = ex

    st.markdown("---")

    # SEARCH HISTORY
    st.markdown("### 🕐 Search History")
    if st.session_state.search_history:
        for past_query in reversed(st.session_state.search_history[-8:]):
            st.markdown(f'<div class="history-item">🔍 {past_query}</div>', unsafe_allow_html=True)
        if st.button("Clear History", use_container_width=True):
            st.session_state.search_history = []
    else:
        st.caption("No searches yet")


# ─────────────────────────────────────────
# MAIN — TABS
# ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Search Patents", "🤖 AI Assistant", "📋 Browse All Patents"])

# ── TAB 1: SEARCH ──
with tab1:
    st.markdown("### 🔍 Search Aerospace Patents by Meaning")

    query = st.text_input(
        "Enter any concept, technology, or question:",
        value=st.session_state.query,
        placeholder="e.g. AI for unmanned aircraft navigation..."
    )

    search_clicked = st.button("🔍 Search", type="primary")

    if query and search_clicked:
        # Save to history
        if query not in st.session_state.search_history:
            st.session_state.search_history.append(query)

        with st.spinner(f"Searching for '{query}'..."):
            results = search_patents(query, patents, model, index, top_k=top_k)

        st.markdown(f"### 📋 Top {len(results)} Results for: *\"{query}\"*")
        st.caption("Ranked by semantic similarity — not keyword matching")

        for r in results:
            score = r["similarity_score"]
            if score < 0.9:
                match_label = "🟢 Strong Match"
            elif score < 1.2:
                match_label = "🟡 Good Match"
            else:
                match_label = "🔵 Related"

            with st.container():
                st.markdown(f"""
                <div class="patent-card">
                    <div class="patent-title">#{r['rank']} — {r['title']}</div>
                    <div class="patent-meta">
                        📄 Patent No. {r['number']} &nbsp;|&nbsp;
                        📅 {r['date']} &nbsp;|&nbsp;
                        {match_label} &nbsp;|&nbsp;
                        <span class="score-badge">Score: {score:.3f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("Read Abstract"):
                    st.write(r["abstract"])

        st.divider()
        st.caption("💡 Tip: Try different phrasings — this system understands meaning, not just words")

    elif not query:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 🧠 Semantic Search")
            st.write("Finds patents by meaning — not keyword matching.")
        with col2:
            st.markdown("#### ⚡ Instant Results")
            st.write("FAISS vector search returns results in milliseconds.")
        with col3:
            st.markdown("#### ✈️ Aerospace Focused")
            st.write("Specialized on aerospace, UAV, drone, and propulsion patents.")


# ── TAB 2: BROWSE ALL ──
# ── TAB 2: AI ASSISTANT ──
with tab2:
    st.markdown("### 🤖 Ask the Patent Intelligence Assistant")
    st.caption("Powered by RAG — answers are grounded in real patent data, not guesswork")

    # Import RAG
    from rag import rag_search_and_answer

    # Example questions
    st.markdown("**Example questions you can ask:**")
    example_questions = [
        "Which patents use AI or machine learning for drone control?",
        "What are the main approaches to fuel efficiency in aerospace?",
        "Which patents focus on autonomous navigation systems?",
        "What safety systems exist for UAV operations in urban areas?",
        "Which patents deal with satellite positioning and control?"
    ]

    col1, col2 = st.columns(2)
    for i, eq in enumerate(example_questions):
        if i % 2 == 0:
            with col1:
                if st.button(eq, use_container_width=True, key=f"eq_{i}"):
                    st.session_state.rag_question = eq
        else:
            with col2:
                if st.button(eq, use_container_width=True, key=f"eq_{i}"):
                    st.session_state.rag_question = eq

    st.markdown("---")

    # Initialize session state for RAG
    if "rag_question" not in st.session_state:
        st.session_state.rag_question = ""

    rag_question = st.text_area(
        "Ask anything about aerospace patents:",
        value=st.session_state.rag_question,
        placeholder="e.g. Which patents use machine learning for UAV control?",
        height=80
    )

    ask_clicked = st.button("🤖 Ask AI Assistant", type="primary")

    if rag_question and ask_clicked:
        with st.spinner("Retrieving patents and generating answer..."):
            result = rag_search_and_answer(
                rag_question, patents, model, index, top_k=5
            )

        # Show AI answer
        st.markdown("### 💡 AI Analysis")
        st.markdown(f"""
        <div style="background:#f0f7ff; border-left:4px solid #1A3C6E;
        padding:1.2rem; border-radius:6px; margin-bottom:1rem;">
        {result['answer'].replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)

        # Show patents used
        st.markdown("### 📄 Patents Analyzed")
        for p in result["patents_used"]:
            score = p["similarity_score"]
            with st.expander(f"📄 {p['title']} — Patent #{p['number']}"):
                st.write(p["abstract"])
                st.caption(f"Date: {p['date']} | Similarity Score: {score:.3f}")

        # Add to search history
        if rag_question not in st.session_state.search_history:
            st.session_state.search_history.append(f"[AI] {rag_question}")

    elif not rag_question:
        st.info("💡 Type a question above or click an example to get an AI-powered analysis of the patent database.")