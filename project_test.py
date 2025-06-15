import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import json
from collections import defaultdict
import math
import os
import sys
from sklearn.datasets import fetch_20newsgroupss

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")
    sys.exit(1)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    """
    Preprocess input text for indexing or querying:
    - Tokenize
    - Lowercase
    - Remove stopwords and non-alphanumeric tokens
    - Apply stemming
    """
    try:
        tokens = word_tokenize(text.lower())
        return [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    except Exception:
        return []

def load_20newsgroups_data(limit=100):
    """
    Load a subset of the 20 Newsgroups dataset and prepare:
    - A mapping from doc ID to document text (passages)
    - Synthetic queries
    - Query relevance judgments (qrels)
    """
    try:
        # Fetch only the categories needed for our example queries
        news = fetch_20newsgroups(
            subset='all',
            #categories=['sci.space', 'comp.graphics'],
            remove=('headers', 'footers', 'quotes')
        )
        passages = {}
        for i, doc in enumerate(news.data[:limit]):  # Limit for performance
            if doc.strip():  # Skip empty documents
                passages[str(i)] = doc.strip()

        # Generate synthetic queries and qrels based on categories
        queries = {
            "q1": "space exploration",
            "q2": "computer graphics"
        }
        qrels = defaultdict(list)
        label_map = {i: cat for i, cat in enumerate(news.target_names)}
        for i, label in enumerate(news.target[:limit]):
            if str(i) in passages:
                category = label_map.get(label, "")
                if category == "sci.space":
                    qrels["q1"].append(str(i))
                elif category == "comp.graphics":
                    qrels["q2"].append(str(i))

        return passages, queries, qrels
    except Exception as e:
        print(f"Error loading 20 Newsgroups data: {e}")
        return {}, {}, defaultdict(list)

class InvertedIndex:
    """
    A basic inverted index supporting BM25/BIM scoring.
    Stores term frequencies per document and global collection statistics.
    """
    def __init__(self):
        self.index = defaultdict(list)
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.total_docs = 0

    def add_document(self, doc_id, text):
        """
        Preprocess and index a single document.
        Update average document length and total document count.
        """
        tokens = preprocess(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.total_docs += 1
        self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs
        term_counts = defaultdict(int)
        for token in tokens:
            term_counts[token] += 1
        for term, freq in term_counts.items():
            self.index[term].append((doc_id, freq))

    def save(self, filename):
        """
        Serialize the index and statistics to disk.
        Useful for reusing the index without reprocessing.
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'index': dict(self.index),
                    'doc_lengths': self.doc_lengths,
                    'avg_doc_length': self.avg_doc_length,
                    'total_docs': self.total_docs
                }, f)
        except Exception as e:
            print(f"Error saving index: {e}")

def bm25_score(query, index, passages, k1=1.5, b=0.75):
    """
    Score documents using the BM25 formula.
    - Accounts for term frequency and document length normalization.
    - Returns top documents with their scores and snippets.
    """
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
    # Return top 5 results with truncated text (first 100 chars) for readability
    results = [(doc_id, score, passages.get(doc_id, "Not found")[:100] + "...")
               for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:5]
    return results

def bim_score(query, index, passages):
    """
    Binary Independence Model (BIM) scoring based on term presence.
    - Uses odds ratio of term in relevant vs. non-relevant documents.
    - Assumes binary term presence and independence across terms.
    """
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
    results = [(doc_id, score, passages.get(doc_id, "Not found")[:100] + "...")
               for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:5]
    return results

def relevance_feedback(index, query, relevant_docs, passages):
    """
    Update term weights using relevance feedback (Rocchio-style logic).
    - Estimate new term relevance probabilities based on user-labeled docs.
    - Re-score documents accordingly.
    """
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
                c_i = math.log(p_i / (1 - p_i) * (1 - u_i) / u_i) if p_i > 0 and p_i < 1 and u_i > 0 else 0
                scores[doc_id] += c_i * tf
    results = [(doc_id, score, passages.get(doc_id, "Not found")[:100] + "...")
               for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:5]
    return results

def pseudo_relevance_feedback(index, query, passages, k=2):
    """
    Apply pseudo-relevance feedback:
    - Assume top-k documents from BM25 are relevant.
    - Re-compute term weights using estimated relevance.
    """
    initial_ranking = bm25_score(query, index, passages)[:k]
    relevant_docs = [doc_id for doc_id, _, _ in initial_ranking]
    return relevance_feedback(index, query, relevant_docs, passages)

def evaluate_system(index, queries, qrels):
    """
    Evaluate the ranking system using Mean Average Precision (MAP):
    - For each query, compute precision at each relevant doc.
    - Average over all queries.
    """
    map_score = 0
    for query_id, query in queries.items():
        ranking = bm25_score(query, index, passages={})  # Passages not needed for MAP
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

def print_ranking(title, ranking):
    """Format and display ranking results with score and truncated passage."""
    print(f"\n{title}:")
    if not ranking:
        print("No results found.")
        return
    for i, (doc_id, score, text) in enumerate(ranking, 1):
        print(f"{i}. Document ID: {doc_id}, Score: {score:.4f}")
        print(f"   Text: {text}")

if __name__ == "__main__":
    # Load 20 Newsgroups dataset
    print("Loading 20 Newsgroups dataset...")
    passages, queries, qrels = load_20newsgroups_data(limit=100)
    if not passages:
        print("Failed to load dataset. Exiting.")
        sys.exit(1)
    print(f"Loaded {len(passages)} documents and {len(queries)} queries.")

    # Build the index
    index = InvertedIndex()
    for doc_id, text in passages.items():
        index.add_document(doc_id, text)
    index.save("index.json")
    print("Index saved to 'index.json'.")

    # Prompt user for query
    print("\nSample queries: 'space exploration', 'computer graphics'")
    with open("results.txt", "w", encoding="utf-8") as f:
        while True:
            sample_query = input("\nEnter your query (or 'quit' to exit): ").strip()
            if sample_query.lower() == 'quit':
                print("Exiting program.")
                break
            if not sample_query:
                print("Error: Query cannot be empty. Please try again.")
                continue

            print(f"\nProcessing query: {sample_query}")
            f.write(f"\nQuery: {sample_query}\n")

            # BM25 ranking
            bm25_ranking = bm25_score(sample_query, index, passages)
            print_ranking("BM25 Ranking", bm25_ranking)
            f.write("\nBM25 Ranking:\n")
            for doc_id, score, text in bm25_ranking:
                f.write(f"Document ID: {doc_id}, Score: {score:.4f}, Text: {text}\n")

            # BIM ranking
            bim_ranking = bim_score(sample_query, index, passages)
            print_ranking("BIM Ranking", bim_ranking)
            f.write("\nBIM Ranking:\n")
            for doc_id, score, text in bim_ranking:
                f.write(f"Document ID: {doc_id}, Score: {score:.4f}, Text: {text}\n")

            # Relevance feedback
            if bm25_ranking:
                feedback_ranking = relevance_feedback(index, sample_query, [bm25_ranking[0][0]], passages)
                print_ranking("Relevance Feedback Ranking", feedback_ranking)
                f.write("\nRelevance Feedback Ranking:\n")
                for doc_id, score, text in feedback_ranking:
                    f.write(f"Document ID: {doc_id}, Score: {score:.4f}, Text: {text}\n")

            # Pseudo-relevance feedback
            pseudo_ranking = pseudo_relevance_feedback(index, sample_query, passages)
            print_ranking("Pseudo-Relevance Feedback Ranking", pseudo_ranking)
            f.write("\nPseudo-Relevance Feedback Ranking:\n")
            for doc_id, score, text in pseudo_ranking:
                f.write(f"Document ID: {doc_id}, Score: {score:.4f}, Text: {text}\n")

    # Evaluate with MAP for predefined queries
    map_score = evaluate_system(index, queries, qrels)
    print(f"\nMAP Score for predefined queries: {map_score:.4f}")
    with open("results.txt", "a", encoding="utf-8") as f:
        f.write(f"\nMAP Score for predefined queries: {map_score:.4f}\n")