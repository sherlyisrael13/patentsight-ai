# ✈️ PatentSight AI — Aerospace Patent Intelligence Platform

An AI-powered semantic search engine for aerospace patents, built with sentence-transformers, FAISS, and Streamlit.

## What it does
- Searches aerospace patents by **meaning**, not keywords
- Type "fuel efficient flight" and find propulsion patents — even without exact word matches
- Powered by `all-MiniLM-L6-v2` embeddings + FAISS vector search
- Browse patents by year with analytics dashboard

## Tech Stack
- **Python** — core language
- **sentence-transformers** — converts patent text to semantic embeddings
- **FAISS** — vector similarity search (millisecond retrieval)
- **Streamlit** — interactive web interface
- **Pandas** — data processing and analytics

## Project Structure
patentsight-ai/
├── app.py              ← Streamlit UI
├── src/
│   ├── fetch_patents.py  ← Data ingestion layer
│   └── embedder.py       ← AI embeddings + FAISS search
├── data/
│   └── patents.json      ← Aerospace patent dataset
└── requirements.txt

## How to run locally
```bash
git clone https://github.com/YOUR_USERNAME/patentsight-ai
cd patentsight-ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Built by
Josephine Sherly P — B.Tech CSE (AI & ML), SRM IST Trichy
