"""
Embedding Quality Benchmark

Evaluates embedding models on:
- Cosine similarity distribution
- Retrieval accuracy
- Embedding speed
"""

import time
import numpy as np
from typing import Optional


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def benchmark_embedding_model(
    model_name: str,
    test_queries: list[str],
    test_documents: list[str],
    relevance_labels: list[list[int]]
) -> dict:
    """
    Benchmark an embedding model.
    
    Args:
        model_name: Name of the embedding model
        test_queries: List of test queries
        test_documents: List of documents
        relevance_labels: Binary relevance labels [query_idx][doc_idx]
        
    Returns:
        Dictionary with benchmark results
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Please install sentence-transformers")
    
    print(f"Benchmarking: {model_name}")
    
    # Load model
    model = SentenceTransformer(model_name)
    
    # Benchmark encoding speed
    start_time = time.time()
    query_embeddings = model.encode(test_queries)
    query_time = time.time() - start_time
    
    start_time = time.time()
    doc_embeddings = model.encode(test_documents)
    doc_time = time.time() - start_time
    
    # Calculate similarities
    similarities = np.zeros((len(test_queries), len(test_documents)))
    for i, q_emb in enumerate(query_embeddings):
        for j, d_emb in enumerate(doc_embeddings):
            similarities[i, j] = cosine_similarity(q_emb, d_emb)
    
    # Calculate retrieval metrics
    mrr_scores = []
    recall_at_5 = []
    
    for q_idx in range(len(test_queries)):
        ranked_docs = np.argsort(similarities[q_idx])[::-1]
        relevant_docs = [j for j, rel in enumerate(relevance_labels[q_idx]) if rel == 1]
        
        # MRR
        for rank, doc_idx in enumerate(ranked_docs):
            if doc_idx in relevant_docs:
                mrr_scores.append(1.0 / (rank + 1))
                break
        else:
            mrr_scores.append(0.0)
        
        # Recall@5
        top_5 = set(ranked_docs[:5])
        recall = len(top_5 & set(relevant_docs)) / len(relevant_docs) if relevant_docs else 0
        recall_at_5.append(recall)
    
    return {
        "model": model_name,
        "dimension": query_embeddings.shape[1],
        "query_encoding_time": query_time / len(test_queries),
        "doc_encoding_time": doc_time / len(test_documents),
        "mrr": np.mean(mrr_scores),
        "recall_at_5": np.mean(recall_at_5),
        "avg_similarity": np.mean(similarities),
        "similarity_std": np.std(similarities)
    }


def run_benchmark():
    """Run the complete embedding benchmark."""
    
    # Test data
    queries = [
        "What is machine learning?",
        "How does neural network training work?",
        "Explain gradient descent optimization",
        "What are transformers in NLP?",
        "How to implement attention mechanism?"
    ]
    
    documents = [
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Neural networks are trained using backpropagation to minimize loss.",
        "Gradient descent is an optimization algorithm that iteratively updates parameters.",
        "Transformers use self-attention to process sequential data in parallel.",
        "The attention mechanism computes weighted sums of values based on query-key similarity.",
        "Deep learning uses multiple layers of neural networks for feature extraction.",
        "Convolutional neural networks are designed for processing grid-like data.",
        "Recurrent neural networks handle sequential data with hidden state memory."
    ]
    
    # Relevance labels (1 = relevant, 0 = not relevant)
    relevance = [
        [1, 0, 0, 0, 0, 1, 0, 0],  # Query 0
        [0, 1, 1, 0, 0, 1, 0, 0],  # Query 1
        [0, 0, 1, 0, 0, 0, 0, 0],  # Query 2
        [0, 0, 0, 1, 1, 0, 0, 0],  # Query 3
        [0, 0, 0, 1, 1, 0, 0, 0],  # Query 4
    ]
    
    # Models to benchmark
    models = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-MiniLM-L6-v2"
    ]
    
    print("=" * 60)
    print("EMBEDDING QUALITY BENCHMARK")
    print("=" * 60)
    
    results = []
    for model_name in models:
        try:
            result = benchmark_embedding_model(
                model_name, queries, documents, relevance
            )
            results.append(result)
            
            print(f"\n{model_name}:")
            print(f"  Dimension: {result['dimension']}")
            print(f"  Encoding Speed: {result['query_encoding_time']*1000:.2f}ms/query")
            print(f"  MRR: {result['mrr']:.3f}")
            print(f"  Recall@5: {result['recall_at_5']:.3f}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Sort by MRR
    results.sort(key=lambda x: x["mrr"], reverse=True)
    print(f"\n{'Model':<30} {'MRR':>8} {'R@5':>8} {'Dim':>6}")
    print("-" * 55)
    for r in results:
        print(f"{r['model']:<30} {r['mrr']:>8.3f} {r['recall_at_5']:>8.3f} {r['dimension']:>6}")


if __name__ == "__main__":
    run_benchmark()

