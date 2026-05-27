import streamlit as st
import json
import numpy as np
import sys
import os
from collections import Counter
import pandas as pd
import plotly.express as px

sys.path.append("src")

from fetch_patents import load_patents, get_sample_patents, save_patents
from embedder import load_model, create_embeddings, build_faiss_index, search_patents, load_index, save_index
from analytics import get_patents_by_year, get_domain_breakdown, get_top_keywords, get_summary_stats, TECH_DOMAINS
try:
    from rag import rag_search_and_answer
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(page_title="PatentSight AI", page_icon="✈️", layout="wide")

st.markdown("""
<style>
    .main-title { font-size: 2.8rem; font-weight: 700; color: #1A3C6E; margin-bottom: 0; }
    .subtitle { font-size: 1.1rem; color: #666; margin-top: 0; }
    .patent-card { background: #f8f9fa; border-left: 4px solid #1A3C6E; padding: 1rem 1.2rem; border-radius: 6px; margin-bottom: 1rem; }
    .patent-title { font-size: 1.05rem; font-weight: 600; color: #1A3C6E; }
    .patent-meta { font-size: 0.82rem; color: #888; margin-top: 2px; }
    .score-badge { background: #e8f0fe; color: #1A3C6E; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
    .history-item { background: #f0f4ff; padding: 4px 10px; border-radius: 8px; font-size: 0.82rem; color: #333; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "query" not in st.session_state:
    st.session_state.query = ""
if "rag_question" not in st.session_state:
    st.session_state.rag_question = ""

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
    st.markdown("### 📊 Database Stats")
    st.metric("Total Patents", len(patents))
    st.metric("Domain", "Aerospace / UAV")
    st.metric("AI Model", "MiniLM-L6-v2")
    st.markdown("---")
    st.markdown("### 📅 Patents by Year")
    years = [p["year"] for p in patents if p.get("year")]
    year_counts = Counter(years)
    year_df = pd.DataFrame({
        "Year": list(year_counts.keys()),
        "Patents": list(year_counts.values())
    }).sort_values("Year")
    st.bar_chart(year_df.set_index("Year"))
    st.markdown("---")
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
    st.markdown("### 🕐 Search History")
    if st.session_state.search_history:
        for past_query in reversed(st.session_state.search_history[-8:]):
            st.markdown(f'<div class="history-item">🔍 {past_query}</div>', unsafe_allow_html=True)
        if st.button("Clear History", use_container_width=True):
            st.session_state.search_history = []
    else:
        st.caption("No searches yet")

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Search Patents", "🤖 AI Assistant", "📈 Analytics", "🕸️ Knowledge Graph", "📋 Browse All Patents"])

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

# ── TAB 2: AI ASSISTANT ──
# ── TAB 2: AI ASSISTANT ──
with tab2:
    st.markdown("### 🤖 Ask the Patent Intelligence Assistant")
    st.caption("Powered by RAG — answers are grounded in real patent data, not guesswork")

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
    rag_question = st.text_area(
        "Ask anything about aerospace patents:",
        value=st.session_state.rag_question,
        placeholder="e.g. Which patents use machine learning for UAV control?",
        height=80
    )
    ask_clicked = st.button("🤖 Ask AI Assistant", type="primary")

    if rag_question and ask_clicked:
        with st.spinner("Retrieving patents and generating answer..."):
            try:
                result = rag_search_and_answer(rag_question, patents, model, index, top_k=5)
                st.markdown("### 💡 AI Analysis")
                st.markdown(f"""
                <div style="background:#f0f7ff; border-left:4px solid #1A3C6E;
                padding:1.2rem; border-radius:6px; margin-bottom:1rem;">
                {result['answer'].replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)
                st.markdown("### 📄 Patents Analyzed")
                for p in result["patents_used"]:
                    score = p["similarity_score"]
                    with st.expander(f"📄 {p['title']} — Patent #{p['number']}"):
                        st.write(p["abstract"])
                        st.caption(f"Date: {p['date']} | Similarity Score: {score:.3f}")
                if rag_question not in st.session_state.search_history:
                    st.session_state.search_history.append(f"[AI] {rag_question}")
            except Exception as e:
                st.error(f"AI Assistant error: {str(e)}")
    elif not rag_question:
        st.info("💡 Type a question above or click an example to get an AI-powered analysis.")
        
# ── TAB 3: ANALYTICS ──
with tab3:
    st.markdown("### 📈 Aerospace Patent Analytics")
    st.caption("Insights derived from the patent database")

    stats = get_summary_stats(patents)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Patents", stats["total"])
    m2.metric("Year Range", stats["year_range"])
    m3.metric("Latest Year", stats["newest"])
    m4.metric("Tech Domains", stats["domains"])

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📅 Patent Filing Trend")
        year_df = get_patents_by_year(patents)
        fig1 = px.bar(year_df, x="Year", y="Patents Filed",
                      color="Patents Filed", color_continuous_scale="Blues",
                      title="Patents Filed per Year")
        fig1.update_layout(plot_bgcolor="white", showlegend=False,
                           height=350, coloraxis_showscale=False)
        fig1.update_traces(marker_line_width=0)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("#### 🔬 Technology Domain Breakdown")
        domain_df = get_domain_breakdown(patents)
        fig2 = px.bar(domain_df, x="Patent Count", y="Domain",
                      orientation="h", color="Patent Count",
                      color_continuous_scale="Blues",
                      title="Patents by Technology Domain")
        fig2.update_layout(plot_bgcolor="white", showlegend=False,
                           height=350, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔑 Top Keywords Across All Patents")
    keywords_df = get_top_keywords(patents, top_n=15)
    fig3 = px.bar(keywords_df, x="Frequency", y="Keyword",
                  orientation="h", color="Frequency",
                  color_continuous_scale="Teal",
                  title="Most Frequent Technical Terms")
    fig3.update_layout(plot_bgcolor="white", showlegend=False,
                       height=400, coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔍 Explore Patents by Domain")
    selected_domain = st.selectbox("Select a technology domain:", list(TECH_DOMAINS.keys()))
    domain_keywords = TECH_DOMAINS[selected_domain]
    domain_patents = []
    for p in patents:
        text = (p["title"] + " " + p["abstract"]).lower()
        if any(kw.lower() in text for kw in domain_keywords):
            domain_patents.append(p)
    st.markdown(f"**{len(domain_patents)} patents** in *{selected_domain}*")
    for p in domain_patents:
        with st.expander(f"📄 {p['title']} ({p['date']})"):
            st.write(p["abstract"])
            st.caption(f"Patent #{p['number']}")
            
# ── TAB 4: KNOWLEDGE GRAPH + EXPLAINABILITY ──
with tab4:
    from explainer import build_topic_network, graph_to_plotly, explain_search_result

    st.markdown("### 🕸️ Technology Domain Network")
    st.caption("Shows how aerospace technology domains connect based on patent co-occurrence")

    # Build and show network graph
    G, topic_patents = build_topic_network(patents)
    fig_network = graph_to_plotly(G)

    if fig_network:
        st.plotly_chart(fig_network, use_container_width=True)
    
    # Domain details below graph
    st.markdown("---")
    st.markdown("### 📊 Domain Connection Details")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Patents per Domain**")
        for node in G.nodes():
            count = G.nodes[node].get("count", 0)
            st.markdown(f"- **{node}**: {count} patents")

    with col2:
        st.markdown("**Strongest Connections**")
        edges = [(u, v, d["weight"]) for u, v, d in G.edges(data=True)]
        edges_sorted = sorted(edges, key=lambda x: x[2], reverse=True)
        for u, v, w in edges_sorted[:6]:
            st.markdown(f"- {u} ↔ {v}: **{w}** shared patents")

    st.markdown("---")

    # Explainability section
    st.markdown("### 🔍 Search Result Explainer")
    st.caption("See exactly WHY a patent matched your search query")

    explain_query = st.text_input(
        "Enter a search query to explain:",
        placeholder="e.g. AI for drone navigation",
        key="explain_query"
    )

    if explain_query:
        from embedder import search_patents
        results = search_patents(explain_query, patents, model, index, top_k=3)

        st.markdown(f"**Top 3 results for:** *\"{explain_query}\"* — with explanations")

        for r in results:
            explanation = explain_search_result(explain_query, r)

            with st.expander(f"#{r['rank']} — {r['title']}"):
                # Explanation text
                st.markdown(explanation["explanation"])

                # Keyword importance bars
                st.markdown("**Key matching terms:**")
                if explanation["keywords"]:
                    kw_df = pd.DataFrame({
                        "Keyword": explanation["keywords"],
                        "Relevance": explanation["scores"]
                    })
                    st.bar_chart(kw_df.set_index("Keyword"))

                st.caption(f"Patent #{r['number']} | Date: {r['date']} | Score: {r['similarity_score']:.3f}")


# ── TAB 5: BROWSE ALL ──
with tab5:
    st.markdown("### 📋 All Patents in Database")
    st.caption(f"Showing all {len(patents)} aerospace patents")
    all_years = sorted(set(p["year"] for p in patents if p.get("year")), reverse=True)
    selected_year = st.selectbox("Filter by year", ["All years"] + all_years)
    filtered = patents if selected_year == "All years" else [
        p for p in patents if p.get("year") == selected_year
    ]
    st.markdown(f"**{len(filtered)} patents**")
    st.markdown("---")
    for p in filtered:
        with st.expander(f"📄 {p['title']} ({p['date']})"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Abstract:** {p['abstract']}")
            with col2:
                st.markdown(f"**Patent No.:** `{p['number']}`")
                st.markdown(f"**Date:** {p['date']}")
                st.markdown(f"**Year:** {p['year']}")