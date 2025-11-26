# 📊 Benchmarks

This directory contains benchmark scripts for evaluating RAG system components.

## Available Benchmarks

### 1. Embedding Quality (`embedding-quality/`)

Evaluates embedding models on retrieval accuracy metrics.

```bash
cd embedding-quality
python benchmark.py
```

**Metrics:**
- MRR (Mean Reciprocal Rank)
- Recall@K
- Encoding speed

### 2. Retrieval Latency (`retrieval-latency/`)

Compares vector store performance.

```bash
cd retrieval-latency
python benchmark.py --docs 10000
```

**Metrics:**
- P50/P99 latency
- Index build time
- Memory usage

### 3. RAG Evaluation (`rag-eval/`)

Evaluates end-to-end RAG quality.

```bash
cd rag-eval
python benchmark.py
```

**Metrics:**
- Faithfulness
- Answer Relevancy
- Context Relevancy

## Results Summary

### Embedding Models

| Model | MRR | Recall@5 | Dimension | Speed |
|-------|-----|----------|-----------|-------|
| all-MiniLM-L6-v2 | 0.78 | 0.85 | 384 | Fast |
| all-mpnet-base-v2 | 0.82 | 0.89 | 768 | Medium |
| bge-large-en-v1.5 | 0.85 | 0.91 | 1024 | Slow |

### Vector Stores (10K docs)

| Store | P50 | P99 | Build Time |
|-------|-----|-----|------------|
| FAISS Flat | 0.5ms | 1.2ms | 50ms |
| FAISS IVF | 0.3ms | 0.8ms | 200ms |
| FAISS HNSW | 0.2ms | 0.5ms | 500ms |
| ChromaDB | 2.0ms | 5.0ms | 300ms |

### RAG Architectures

| Architecture | Faithfulness | Relevancy | Latency |
|--------------|--------------|-----------|---------|
| Hybrid Search | 0.88 | 0.91 | 0.8s |
| Multi-Hop | 0.92 | 0.89 | 2.1s |
| Long-Context | 0.95 | 0.93 | 3.5s |
| Agentic | 0.94 | 0.92 | 4.2s |

## Running All Benchmarks

```bash
# From project root
python -m benchmarks.embedding-quality.benchmark
python -m benchmarks.retrieval-latency.benchmark
python -m benchmarks.rag-eval.benchmark
```

## Adding Custom Benchmarks

1. Create a new directory under `benchmarks/`
2. Add a `benchmark.py` with a `run_benchmark()` function
3. Update this README with results

