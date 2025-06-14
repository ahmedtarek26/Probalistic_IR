import math
from collections import defaultdict
from typing import List
from index import InvertedIndex, preprocess


def bm25_score(query: str, index: InvertedIndex, passages: dict, k1: float = 1.5, b: float = 0.75):
    """Compute BM25 scores for passages and return top results."""
    scores = defaultdict(float)
    query_terms = preprocess(query)
    N = index.total_docs
    avgdl = index.avg_doc_length
    for term in query_terms:
        if term in index.index:
            df = len(index.index[term])
            idf = math.log(N / df) if df > 0 else 0
            for doc_id, tf in index.index[term]:
                score = idf * ((k1 + 1) * tf) / (k1 * ((1 - b) + b * (index.doc_lengths[doc_id] / avgdl)) + tf)
                scores[doc_id] += score
    results = [
        (doc_id, score, passages.get(doc_id, "Not found")[:100] + "...")
        for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return results


def bim_score(query: str, index: InvertedIndex, passages: dict):
    """Compute BIM scores for passages."""
    scores = defaultdict(float)
    query_terms = preprocess(query)
    N = index.total_docs
    for term in query_terms:
        if term in index.index:
            df = len(index.index[term])
            p_i = df / N if df > 0 else 0.5
            u_i = 1 - p_i
            c_i = math.log((p_i / u_i) * ((1 - u_i) / (1 - p_i))) if p_i > 0 and u_i > 0 else 0
            for doc_id, _ in index.index[term]:
                scores[doc_id] += c_i
    results = [
        (doc_id, score, passages.get(doc_id, "Not found")[:100] + "...")
        for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return results


def relevance_feedback(index: InvertedIndex, query: str, relevant_docs: List[str], passages: dict):
    """Update rankings using explicit relevance feedback."""
    query_terms = preprocess(query)
    N = index.total_docs
    VR = set(relevant_docs)
    scores = defaultdict(float)
    for term in query_terms:
        if term in index.index:
            df = len(index.index[term])
            VR_i = len([doc_id for doc_id, _ in index.index[term] if doc_id in VR])
            p_i = (VR_i + 0.5) / (len(VR) + 1)
            u_i = (df - VR_i + 0.5) / (N - len(VR) + 1)
            for doc_id, tf in index.index[term]:
                c_i = math.log(p_i / (1 - p_i) * (1 - u_i) / u_i) if 0 < p_i < 1 and u_i > 0 else 0
                scores[doc_id] += c_i * tf
    results = [
        (doc_id, score, passages.get(doc_id, "Not found")[:100] + "...")
        for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return results


def pseudo_relevance_feedback(index: InvertedIndex, query: str, passages: dict, k: int = 2):
    """Apply pseudo-relevance feedback using top k passages."""
    initial_ranking = bm25_score(query, index, passages)[:k]
    relevant_docs = [doc_id for doc_id, _, _ in initial_ranking]
    return relevance_feedback(index, query, relevant_docs, passages)


def evaluate_system(index: InvertedIndex, queries: dict, qrels: dict):
    """Evaluate the system using Mean Average Precision (MAP)."""
    map_score = 0
    for query_id, query in queries.items():
        ranking = bm25_score(query, index, passages={})
        relevant_docs = set(qrels.get(query_id, []))
        relevant_retrieved = 0
        precision_sum = 0
        for i, (doc_id, _, _) in enumerate(ranking, 1):
            if doc_id in relevant_docs:
                relevant_retrieved += 1
                precision_sum += relevant_retrieved / i
        avg_precision = precision_sum / len(relevant_docs) if relevant_docs else 0
        map_score += avg_precision
    return map_score / len(queries) if queries else 0

