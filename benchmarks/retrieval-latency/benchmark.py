"""
Retrieval Latency Benchmark

Compares latency of different vector stores:
- FAISS (various index types)
- ChromaDB
- In-memory search
"""

import time
import numpy as np
from typing import Optional


def generate_synthetic_data(
    num_documents: int = 10000,
    embedding_dim: int = 384
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic embeddings for benchmarking."""
    np.random.seed(42)
    
    # Generate random document embeddings
    doc_embeddings = np.random.randn(num_documents, embedding_dim).astype(np.float32)
    # Normalize
    doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    
    # Generate random query embeddings
    query_embeddings = np.random.randn(100, embedding_dim).astype(np.float32)
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    
    return doc_embeddings, query_embeddings


def benchmark_faiss_flat(doc_embeddings: np.ndarray, query_embeddings: np.ndarray, top_k: int = 10):
    """Benchmark FAISS Flat index."""
    try:
        import faiss
    except ImportError:
        return {"error": "FAISS not installed"}
    
    dim = doc_embeddings.shape[1]
    
    # Build index
    start = time.time()
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings)
    build_time = time.time() - start
    
    # Query
    latencies = []
    for query in query_embeddings:
        start = time.time()
        _, _ = index.search(query.reshape(1, -1), top_k)
        latencies.append((time.time() - start) * 1000)  # ms
    
    return {
        "index_type": "FAISS Flat",
        "build_time_ms": build_time * 1000,
        "p50_latency_ms": np.percentile(latencies, 50),
        "p99_latency_ms": np.percentile(latencies, 99),
        "avg_latency_ms": np.mean(latencies)
    }


def benchmark_faiss_ivf(doc_embeddings: np.ndarray, query_embeddings: np.ndarray, top_k: int = 10):
    """Benchmark FAISS IVF index."""
    try:
        import faiss
    except ImportError:
        return {"error": "FAISS not installed"}
    
    dim = doc_embeddings.shape[1]
    nlist = min(100, len(doc_embeddings) // 100)
    
    # Build index
    start = time.time()
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, max(1, nlist))
    index.train(doc_embeddings)
    index.add(doc_embeddings)
    index.nprobe = 10
    build_time = time.time() - start
    
    # Query
    latencies = []
    for query in query_embeddings:
        start = time.time()
        _, _ = index.search(query.reshape(1, -1), top_k)
        latencies.append((time.time() - start) * 1000)
    
    return {
        "index_type": "FAISS IVF",
        "build_time_ms": build_time * 1000,
        "p50_latency_ms": np.percentile(latencies, 50),
        "p99_latency_ms": np.percentile(latencies, 99),
        "avg_latency_ms": np.mean(latencies)
    }


def benchmark_faiss_hnsw(doc_embeddings: np.ndarray, query_embeddings: np.ndarray, top_k: int = 10):
    """Benchmark FAISS HNSW index."""
    try:
        import faiss
    except ImportError:
        return {"error": "FAISS not installed"}
    
    dim = doc_embeddings.shape[1]
    
    # Build index
    start = time.time()
    index = faiss.IndexHNSWFlat(dim, 32)
    index.add(doc_embeddings)
    build_time = time.time() - start
    
    # Query
    latencies = []
    for query in query_embeddings:
        start = time.time()
        _, _ = index.search(query.reshape(1, -1), top_k)
        latencies.append((time.time() - start) * 1000)
    
    return {
        "index_type": "FAISS HNSW",
        "build_time_ms": build_time * 1000,
        "p50_latency_ms": np.percentile(latencies, 50),
        "p99_latency_ms": np.percentile(latencies, 99),
        "avg_latency_ms": np.mean(latencies)
    }


def benchmark_chromadb(doc_embeddings: np.ndarray, query_embeddings: np.ndarray, top_k: int = 10):
    """Benchmark ChromaDB."""
    try:
        import chromadb
    except ImportError:
        return {"error": "ChromaDB not installed"}
    
    # Build index
    start = time.time()
    client = chromadb.Client()
    collection = client.create_collection("benchmark")
    
    # Add in batches
    batch_size = 1000
    for i in range(0, len(doc_embeddings), batch_size):
        batch = doc_embeddings[i:i+batch_size]
        ids = [str(j) for j in range(i, min(i+batch_size, len(doc_embeddings)))]
        collection.add(
            ids=ids,
            embeddings=batch.tolist()
        )
    build_time = time.time() - start
    
    # Query
    latencies = []
    for query in query_embeddings:
        start = time.time()
        _ = collection.query(query_embeddings=[query.tolist()], n_results=top_k)
        latencies.append((time.time() - start) * 1000)
    
    return {
        "index_type": "ChromaDB",
        "build_time_ms": build_time * 1000,
        "p50_latency_ms": np.percentile(latencies, 50),
        "p99_latency_ms": np.percentile(latencies, 99),
        "avg_latency_ms": np.mean(latencies)
    }


def benchmark_numpy_brute(doc_embeddings: np.ndarray, query_embeddings: np.ndarray, top_k: int = 10):
    """Benchmark brute-force NumPy search."""
    
    # No build time for brute force
    build_time = 0
    
    # Query
    latencies = []
    for query in query_embeddings:
        start = time.time()
        similarities = np.dot(doc_embeddings, query)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        latencies.append((time.time() - start) * 1000)
    
    return {
        "index_type": "NumPy Brute",
        "build_time_ms": build_time,
        "p50_latency_ms": np.percentile(latencies, 50),
        "p99_latency_ms": np.percentile(latencies, 99),
        "avg_latency_ms": np.mean(latencies)
    }


def run_benchmark(num_documents: int = 10000):
    """Run the complete latency benchmark."""
    
    print("=" * 70)
    print(f"RETRIEVAL LATENCY BENCHMARK ({num_documents:,} documents)")
    print("=" * 70)
    
    print("\nGenerating synthetic data...")
    doc_embeddings, query_embeddings = generate_synthetic_data(num_documents)
    print(f"Documents: {doc_embeddings.shape}")
    print(f"Queries: {query_embeddings.shape}")
    
    benchmarks = [
        benchmark_numpy_brute,
        benchmark_faiss_flat,
        benchmark_faiss_ivf,
        benchmark_faiss_hnsw,
        benchmark_chromadb,
    ]
    
    results = []
    for benchmark_fn in benchmarks:
        print(f"\nRunning {benchmark_fn.__name__}...")
        try:
            result = benchmark_fn(doc_embeddings, query_embeddings)
            results.append(result)
            
            if "error" not in result:
                print(f"  Build: {result['build_time_ms']:.2f}ms")
                print(f"  P50: {result['p50_latency_ms']:.2f}ms")
                print(f"  P99: {result['p99_latency_ms']:.2f}ms")
            else:
                print(f"  Error: {result['error']}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    valid_results = [r for r in results if "error" not in r]
    print(f"\n{'Index Type':<20} {'Build (ms)':>12} {'P50 (ms)':>10} {'P99 (ms)':>10}")
    print("-" * 55)
    for r in sorted(valid_results, key=lambda x: x["p50_latency_ms"]):
        print(f"{r['index_type']:<20} {r['build_time_ms']:>12.2f} {r['p50_latency_ms']:>10.2f} {r['p99_latency_ms']:>10.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=10000, help="Number of documents")
    args = parser.parse_args()
    
    run_benchmark(args.docs)

