import numpy as np
from collections import defaultdict

STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "of", "in", "to", "with",
    "using", "based", "system", "method", "device", "apparatus",
    "comprising", "includes", "wherein", "said", "plurality",
    "at", "by", "on", "is", "are", "that", "which", "from",
    "its", "this", "be", "has", "have", "each", "multiple",
    "one", "two", "use", "used", "provide", "providing", "also",
    "can", "may", "when", "where", "while", "about", "into",
    "through", "during", "before", "after", "above", "below"
}


def get_query_overlap_words(query, patent_text, top_n=8):
    def extract_words(text):
        text = text.lower()
        for ch in [",", ".", "-", "(", ")", ";", ":", "/"]:
            text = text.replace(ch, " ")
        words = text.split()
        return [w for w in words if len(w) > 3 and w not in STOPWORDS]

    query_words = set(extract_words(query))
    patent_words = extract_words(patent_text)

    scored = {}
    for word in patent_words:
        if word in scored:
            continue
        score = 0
        if word in query_words:
            score += 3
        for qw in query_words:
            if word in qw or qw in word:
                score += 2
                break
        score += len(word) * 0.1
        scored[word] = score

    top_words = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_words


def explain_search_result(query, patent, top_n=8):
    patent_text = patent["title"] + " " + patent["abstract"]
    top_words = get_query_overlap_words(query, patent_text, top_n=top_n)

    keywords = [w for w, s in top_words if s > 0]
    scores = [s for w, s in top_words if s > 0]

    if scores:
        max_score = max(scores)
        normalized = [s / max_score for s in scores]
    else:
        normalized = []

    return {
        "keywords": keywords,
        "scores": normalized,
        "explanation": build_explanation(query, patent, keywords)
    }


def build_explanation(query, patent, keywords):
    if not keywords:
        return "This patent was retrieved based on semantic similarity to your query."

    top_3 = keywords[:3]
    keyword_str = ", ".join(f'"{k}"' for k in top_3)

    return (
        f"This patent matched your query about **{query}** "
        f"because it strongly relates to {keyword_str}. "
        f"The semantic embedding model found deep conceptual similarity "
        f"between your search and this patent's technical content."
    )


def build_topic_network(patents):
    import networkx as nx

    G = nx.Graph()
    topic_patents = defaultdict(list)

    for patent in patents:
        text = (patent["title"] + " " + patent["abstract"]).lower()

        from analytics import TECH_DOMAINS
        patent_domains = []
        for domain, keywords in TECH_DOMAINS.items():
            for kw in keywords:
                if kw.lower() in text:
                    patent_domains.append(domain)
                    break

        for domain in patent_domains:
            if not G.has_node(domain):
                G.add_node(domain, count=0)
            G.nodes[domain]["count"] = G.nodes[domain].get("count", 0) + 1
            topic_patents[domain].append(patent["title"])

        for i in range(len(patent_domains)):
            for j in range(i + 1, len(patent_domains)):
                d1, d2 = patent_domains[i], patent_domains[j]
                if G.has_edge(d1, d2):
                    G[d1][d2]["weight"] += 1
                else:
                    G.add_edge(d1, d2, weight=1)

    return G, topic_patents


def graph_to_plotly(G):
    import networkx as nx
    import plotly.graph_objects as go

    if len(G.nodes) == 0:
        return None

    pos = nx.spring_layout(G, seed=42, k=2)

    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2].get("weight", 1)
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=weight * 1.5, color="#aac4e0"),
                hoverinfo="none",
                showlegend=False
            )
        )

    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        count = G.nodes[node].get("count", 1)
        node_text.append(f"{node}<br>{count} patents")
        node_size.append(20 + count * 8)
        node_color.append(count)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=[n.replace(" ", "<br>") for n in G.nodes()],
        textposition="top center",
        hovertext=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Patent Count"),
            line=dict(width=2, color="white")
        ),
        showlegend=False
    )

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=dict(
                text="Technology Domain Network — Patent Co-occurrence Graph",
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode="closest",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(b=20, l=5, r=5, t=50),
            height=500,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    return fig


if __name__ == "__main__":
    import json
    with open("data/patents.json", "r") as f:
        patents = json.load(f)

    query = "machine learning for drone navigation"
    patent = patents[0]
    result = explain_search_result(query, patent)
    print("Keywords:", result["keywords"])
    print("Explanation:", result["explanation"])

    G, topic_patents = build_topic_network(patents)
    print(f"\nNetwork: {len(G.nodes)} nodes, {len(G.edges)} edges")
    for node in G.nodes():
        print(f"  {node}: {G.nodes[node].get('count', 0)} patents")