from collections import defaultdict
from sklearn.datasets import fetch_20newsgroups


def load_20newsgroups_data(limit: int = 100):
    """Load a subset of the 20 Newsgroups dataset and generate synthetic queries and qrels."""
    try:
        news = fetch_20newsgroups(
            subset="all",
            remove=("headers", "footers", "quotes"),
        )

        passages = {}
        for i, doc in enumerate(news.data[:limit]):
            if doc.strip():
                passages[str(i)] = doc.strip()

        queries = {
            f"q{idx + 1}": cat.replace(".", " ")
            for idx, cat in enumerate(news.target_names)
        }

        qrels = defaultdict(list)
        for i, label in enumerate(news.target[:limit]):
            if str(i) in passages:
                qrels[f"q{label + 1}"].append(str(i))

        return passages, queries, qrels
    except Exception as e:
        print(f"Error loading 20 Newsgroups data: {e}")
        return {}, {}, defaultdict(list)
