import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer

# This is the AI model that converts text → numbers
# It downloads automatically the first time (~90MB)
MODEL_NAME = "all-MiniLM-L6-v2"

def load_model():
    """Loads the sentence transformer model"""
    print("Loading AI model...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model ready!")
    return model


def create_embeddings(patents, model):
    """
    Converts patent text into number vectors (embeddings).
    
    Each patent becomes a list of 384 numbers that capture its meaning.
    Similar patents will have similar number lists.
    """
    print(f"Converting {len(patents)} patents to embeddings...")
    
    # Combine title + abstract for richer meaning
    # This is called "text preparation" or "preprocessing"
    texts = []
    for patent in patents:
        combined = f"{patent['title']}. {patent['abstract']}"
        texts.append(combined)
    
    # This is where the AI magic happens
    # The model reads each text and outputs 384 numbers
    embeddings = model.encode(texts, show_progress_bar=True)
    
    print(f"Created embeddings — shape: {embeddings.shape}")
    # Shape will be (15, 384) meaning:
    # 15 patents, each represented as 384 numbers
    
    return embeddings


def build_faiss_index(embeddings):
    """
    Builds a FAISS index — a super fast search structure.
    
    Think of it like a sorted filing cabinet.
    Instead of checking every patent one by one,
    FAISS finds the closest matches instantly.
    """
    print("Building FAISS search index...")
    
    # Get the size of each embedding vector (384)
    dimension = embeddings.shape[1]
    
    # Create a FAISS index that uses cosine-like similarity
    # IndexFlatL2 = finds nearest neighbors using L2 distance
    index = faiss.IndexFlatL2(dimension)
    
    # FAISS needs float32 format
    embeddings_float = np.array(embeddings).astype("float32")
    
    # Add all patent embeddings to the index
    index.add(embeddings_float)
    
    print(f"FAISS index built — {index.ntotal} patents indexed!")
    return index


def search_patents(query, patents, model, index, top_k=5):
    """
    The main search function.
    
    Takes a plain English query like "drone navigation AI"
    Returns the most semantically similar patents.
    
    This is semantic search — finds meaning, not just keywords.
    """
    print(f"\nSearching for: '{query}'")
    
    # Step 1: Convert the query to a number vector
    # Same process as with patents — text → 384 numbers
    query_embedding = model.encode([query])
    query_float = np.array(query_embedding).astype("float32")
    
    # Step 2: Ask FAISS to find the top_k most similar patents
    # distances = how similar (lower = more similar)
    # indices = which patents in our list
    distances, indices = index.search(query_float, top_k)
    
    # Step 3: Build results list
    results = []
    for i, idx in enumerate(indices[0]):
        patent = patents[idx]
        results.append({
            "rank": i + 1,
            "number": patent["number"],
            "title": patent["title"],
            "abstract": patent["abstract"],
            "date": patent["date"],
            "similarity_score": float(distances[0][i])
        })
    
    return results


def save_index(index, filepath="data/faiss_index.bin"):
    """Saves the FAISS index to disk so we don't rebuild every time"""
    faiss.write_index(index, filepath)
    print(f"Index saved to {filepath}")


def load_index(filepath="data/faiss_index.bin"):
    """Loads a previously saved FAISS index"""
    if not os.path.exists(filepath):
        return None
    index = faiss.read_index(filepath)
    print(f"Index loaded — {index.ntotal} patents")
    return index


if __name__ == "__main__":
    # Load patents
    with open("data/patents.json", "r") as f:
        patents = json.load(f)

    # Load AI model
    model = load_model()

    # Create embeddings
    embeddings = create_embeddings(patents, model)

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Save index
    save_index(index)

    # Test search
    print("\n" + "="*50)
    print("TESTING SEMANTIC SEARCH")
    print("="*50)

    test_queries = [
        "AI for unmanned aircraft navigation",
        "fuel efficiency in jet engines",
        "satellite positioning system"
    ]

    for query in test_queries:
        results = search_patents(query, patents, model, index, top_k=3)
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        for r in results:
            print(f"  [{r['rank']}] {r['title']}")
            print(f"       Score: {r['similarity_score']:.4f} | Date: {r['date']}")