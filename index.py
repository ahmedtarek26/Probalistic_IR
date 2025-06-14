import json
import nltk
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize


# Download NLTK resources when the module is imported
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess(text: str):
    """Preprocess text by tokenizing, removing stopwords and stemming."""
    try:
        tokens = word_tokenize(text.lower())
        return [stemmer.stem(t) for t in tokens if t.isalnum() and t not in stop_words]
    except Exception:
        return []


class InvertedIndex:
    """A simple inverted index for storing term-document mappings."""

    def __init__(self):
        self.index = defaultdict(list)
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.total_docs = 0

    def add_document(self, doc_id: str, text: str):
        tokens = preprocess(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.total_docs += 1
        self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs
        term_counts = defaultdict(int)
        for token in tokens:
            term_counts[token] += 1
        for term, freq in term_counts.items():
            self.index[term].append((doc_id, freq))

    def save(self, filename: str):
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "index": dict(self.index),
                        "doc_lengths": self.doc_lengths,
                        "avg_doc_length": self.avg_doc_length,
                        "total_docs": self.total_docs,
                    },
                    f,
                )
        except Exception as e:
            print(f"Error saving index: {e}")
