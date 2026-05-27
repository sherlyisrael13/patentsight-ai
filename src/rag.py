import os
from groq import Groq
from dotenv import load_dotenv

# Load the API key from .env file safely
load_dotenv()

def get_groq_client():
    import streamlit as st
    api_key = (
        os.getenv("GROQ_API_KEY") or
        os.getenv("GROQAPIKEY") or
        os.getenv("GROQKEY") or
        os.getenv("GROQTOKEN") or
        os.getenv("GROQ")
    )
    if not api_key:
        raise ValueError("No Groq API key found in any environment variable")
    
    # Validate key format
    if not api_key.startswith("gsk_"):
        raise ValueError(f"Invalid key format — starts with: {api_key[:6]}")
    
    return Groq(api_key=api_key)

def build_context(patents):
    """
    Takes a list of retrieved patents and formats them as context
    for the LLM to read.
    
    This is the RETRIEVAL part of RAG.
    The LLM will only answer based on this context — no hallucination.
    """
    context = ""
    for i, patent in enumerate(patents):
        context += f"""
PATENT {i+1}:
- Number: {patent['number']}
- Title: {patent['title']}
- Date: {patent['date']}
- Abstract: {patent['abstract']}
---
"""
    return context


def ask_patent_assistant(question, retrieved_patents):
    """
    The core RAG function.
    
    Takes:
    - question: what the user asked
    - retrieved_patents: patents found by FAISS search
    
    Returns:
    - answer: LLM's intelligent response with citations
    
    This is the GENERATION part of RAG.
    """
    client = get_groq_client()
    
    # Build context from retrieved patents
    context = build_context(retrieved_patents)
    
    # This is the PROMPT — instructions to the LLM
    # We tell it to ONLY use the provided patents, not its own knowledge
    system_prompt = """You are PatentSight AI, an expert aerospace patent analyst.

Your job is to answer questions about aerospace patents based ONLY on the patents provided to you.

Rules:
- Only use information from the provided patents
- Always cite patent numbers when making claims (e.g. "Patent #11584539 shows...")
- Be concise and technically precise
- If the patents don't contain enough information, say so honestly
- Structure your answer clearly
- End with a brief "Key Patents:" summary listing the most relevant ones"""

    user_prompt = f"""Based on the following aerospace patents, please answer this question:

Question: {question}

Patents to analyze:
{context}

Please provide a clear, analytical answer with specific citations to patent numbers."""

    # Send to Groq LLM
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Current Groq model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,  # Lower = more factual, less creative
        max_tokens=800
    )
    
    # Extract the text answer
    answer = response.choices[0].message.content
    return answer


def rag_search_and_answer(question, patents, model, index, top_k=5):
    """
    Complete RAG pipeline in one function:
    1. Search patents semantically (retrieval)
    2. Ask LLM to analyze them (generation)
    3. Return both results and answer
    """
    # Import here to avoid circular imports
    from embedder import search_patents
    
    # Step 1: Retrieve relevant patents using FAISS
    retrieved = search_patents(question, patents, model, index, top_k=top_k)
    
    # Step 2: Generate intelligent answer using LLM
    answer = ask_patent_assistant(question, retrieved)
    
    return {
        "question": question,
        "answer": answer,
        "patents_used": retrieved
    }


if __name__ == "__main__":
    # Quick test
    import json
    import sys
    sys.path.append(".")
    
    from embedder import load_model, load_index
    
    with open("data/patents.json", "r") as f:
        patents = json.load(f)
    
    model = load_model()
    index = load_index("data/faiss_index.bin")
    
    question = "Which patents use AI or machine learning for drone control?"
    print(f"Question: {question}\n")
    
    result = rag_search_and_answer(question, patents, model, index)
    
    print("ANSWER:")
    print(result["answer"])
    print("\nPATENTS USED:")
    for p in result["patents_used"]:
        print(f"  - {p['title']} ({p['number']})")  