from collections import Counter
import pandas as pd

# These are the main technology domains we track
# Each domain has keywords we look for in patent titles/abstracts
TECH_DOMAINS = {
    "Autonomous Navigation": ["navigation", "autonomous", "path planning", "obstacle"],
    "Propulsion Systems": ["propulsion", "thrust", "engine", "fuel", "combustion"],
    "AI & Machine Learning": ["machine learning", "artificial intelligence", "deep learning", "reinforcement"],
    "UAV / Drone": ["drone", "uav", "unmanned", "aerial vehicle"],
    "Satellite Systems": ["satellite", "orbital", "spacecraft", "ion thruster"],
    "Structural Systems": ["composite", "wing", "thermal", "morphing", "sensor"],
    "Power Systems": ["fuel cell", "solar", "electric", "battery", "power"],
}


def get_patents_by_year(patents):
    """
    Returns a DataFrame of patent counts per year.
    Used for the filing trend line chart.
    """
    years = [p["year"] for p in patents if p.get("year")]
    year_counts = Counter(years)

    df = pd.DataFrame({
        "Year": list(year_counts.keys()),
        "Patents Filed": list(year_counts.values())
    }).sort_values("Year").reset_index(drop=True)

    return df


def get_domain_breakdown(patents):
    """
    Classifies each patent into technology domains.
    A patent can belong to multiple domains.
    Returns a DataFrame of domain counts.
    """
    domain_counts = {domain: 0 for domain in TECH_DOMAINS}

    for patent in patents:
        # Combine title and abstract for searching
        text = (patent["title"] + " " + patent["abstract"]).lower()

        for domain, keywords in TECH_DOMAINS.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    domain_counts[domain] += 1
                    break  # Count each domain once per patent

    df = pd.DataFrame({
        "Domain": list(domain_counts.keys()),
        "Patent Count": list(domain_counts.values())
    }).sort_values("Patent Count", ascending=True)

    return df


def get_top_keywords(patents, top_n=15):
    """
    Extracts the most frequent meaningful words across all patents.
    This shows what topics dominate the patent database.
    """
    # Words to ignore — too common to be useful
    stopwords = {
        "the", "a", "an", "and", "or", "for", "of", "in", "to", "with",
        "using", "based", "system", "method", "device", "apparatus",
        "comprising", "includes", "wherein", "said", "plurality",
        "at", "by", "on", "is", "are", "that", "which", "from",
        "its", "this", "be", "has", "have", "each", "multiple",
        "one", "two", "use", "used", "provide", "providing"
    }

    word_counts = Counter()

    for patent in patents:
        text = (patent["title"] + " " + patent["abstract"]).lower()
        # Split into words, clean punctuation
        words = text.replace(",", " ").replace(".", " ").replace(
            "-", " ").replace("(", " ").replace(")", " ").split()

        for word in words:
            if len(word) > 4 and word not in stopwords:
                word_counts[word] += 1

    top_words = word_counts.most_common(top_n)
    df = pd.DataFrame(top_words, columns=["Keyword", "Frequency"])
    return df


def get_summary_stats(patents):
    """
    Returns key summary numbers for the dashboard header.
    """
    years = [int(p["year"]) for p in patents if p.get("year")]
    
    return {
        "total": len(patents),
        "year_range": f"{min(years)} – {max(years)}" if years else "N/A",
        "newest": max(years) if years else "N/A",
        "domains": len(TECH_DOMAINS)
    }