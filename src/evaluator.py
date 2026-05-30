import time
import numpy as np


def evaluate_retrieval(patents, model, index):
    """
    Runs evaluation queries and measures system performance.
    Returns metrics that demonstrate system quality.
    """
    from embedder import search_patents

    # Test queries with known relevant patents
    test_queries = [
        {"query": "drone autonomous navigation", "expected_keywords": ["drone", "navigation", "autonomous"]},
        {"query": "machine learning UAV control", "expected_keywords": ["machine learning", "uav", "learning"]},
        {"query": "satellite attitude control", "expected_keywords": ["satellite", "attitude", "control"]},
        {"query": "fuel efficient jet propulsion", "expected_keywords": ["fuel", "propulsion", "engine"]},
        {"query": "aerospace structural health monitoring", "expected_keywords": ["structural", "sensor", "composite"]},
    ]

    latencies = []
    similarity_scores = []
    relevance_scores = []

    for test in test_queries:
        # Measure retrieval latency
        start = time.time()
        results = search_patents(test["query"], patents, model, index, top_k=5)
        end = time.time()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

        # Collect similarity scores
        for r in results:
            similarity_scores.append(r["similarity_score"])

        # Simple relevance check — does top result contain expected keywords?
        if results:
            top_title = results[0]["title"].lower()
            top_abstract = results[0]["abstract"].lower()
            top_text = top_title + " " + top_abstract
            matches = sum(1 for kw in test["expected_keywords"] if kw in top_text)
            relevance = matches / len(test["expected_keywords"])
            relevance_scores.append(relevance)

    return {
        "avg_latency_ms": round(np.mean(latencies), 2),
        "min_latency_ms": round(np.min(latencies), 2),
        "max_latency_ms": round(np.max(latencies), 2),
        "avg_similarity_score": round(np.mean(similarity_scores), 4),
        "avg_relevance_score": round(np.mean(relevance_scores), 4),
        "retrieval_precision": round(np.mean(relevance_scores) * 100, 1),
        "total_queries_tested": len(test_queries),
        "total_patents_indexed": len(patents),
    }