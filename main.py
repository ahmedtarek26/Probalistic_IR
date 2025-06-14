import sys

from dataset import load_20newsgroups_data
from index import InvertedIndex
from scoring import (
    bm25_score,
    bim_score,
    relevance_feedback,
    pseudo_relevance_feedback,
    evaluate_system,
)
 

def print_ranking(title: str, ranking):
     """Print ranking results in a formatted way."""
     print(f"\n{title}:")
     if not ranking:
         print("No results found.")
         return
     for i, (doc_id, score, text) in enumerate(ranking, 1):
         print(f"{i}. Document ID: {doc_id}, Score: {score:.4f}")
         print(f"   Text: {text}")
 

def main():
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

     print("\nSample queries: 'space exploration', 'computer graphics'")
     with open("results.txt", "w", encoding="utf-8") as f:
         while True:
             sample_query = input("\nEnter your query (or 'quit' to exit): ").strip()
            if sample_query.lower() == "quit":
                 print("Exiting program.")
                 break
             if not sample_query:
                 print("Error: Query cannot be empty. Please try again.")
                 continue
 
             print(f"\nProcessing query: {sample_query}")
             f.write(f"\nQuery: {sample_query}\n")
 
            bm25_ranking = bm25_score(sample_query, index, passages)[:5]
             print_ranking("BM25 Ranking", bm25_ranking)
             f.write("\nBM25 Ranking:\n")
             for doc_id, score, text in bm25_ranking:
                 f.write(f"Document ID: {doc_id}, Score: {score:.4f}, Text: {text}\n")
 
            bim_ranking = bim_score(sample_query, index, passages)[:5]
             print_ranking("BIM Ranking", bim_ranking)
             f.write("\nBIM Ranking:\n")
             for doc_id, score, text in bim_ranking:
                 f.write(f"Document ID: {doc_id}, Score: {score:.4f}, Text: {text}\n")
 
             if bm25_ranking:
                feedback_ranking = relevance_feedback(index, sample_query, [bm25_ranking[0][0]], passages)[:5]
                 print_ranking("Relevance Feedback Ranking", feedback_ranking)
                 f.write("\nRelevance Feedback Ranking:\n")
                 for doc_id, score, text in feedback_ranking:
                     f.write(f"Document ID: {doc_id}, Score: {score:.4f}, Text: {text}\n")
 
            pseudo_ranking = pseudo_relevance_feedback(index, sample_query, passages)[:5]
             print_ranking("Pseudo-Relevance Feedback Ranking", pseudo_ranking)
             f.write("\nPseudo-Relevance Feedback Ranking:\n")
             for doc_id, score, text in pseudo_ranking:
                 f.write(f"Document ID: {doc_id}, Score: {score:.4f}, Text: {text}\n")
 
     map_score = evaluate_system(index, queries, qrels)
     print(f"\nMAP Score for predefined queries: {map_score:.4f}")
     with open("results.txt", "a", encoding="utf-8") as f:
        f.write(f"\nMAP Score for predefined queries: {map_score:.4f}\n")

if __name__ == "__main__":
    main()
