"""
Search and Retrieval for RAG Systems

This module provides:
- VectorSearch: Dense vector similarity search
- BM25Search: Sparse keyword search
- HybridSearch: Combined dense + sparse
- Reranker: Cross-encoder reranking
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Single search result."""
    content: str
    score: float
    metadata: dict
    index: int


@dataclass
class SearchResults:
    """Collection of search results."""
    results: list[SearchResult]
    query: str
    method: str


class VectorSearch:
    """
    Dense vector similarity search using FAISS or ChromaDB.
    """
    
    def __init__(
        self,
        backend: str = "faiss",
        embedding_dim: int = 384,
        index_type: str = "flat"
    ):
        """
        Initialize vector search.
        
        Args:
            backend: Search backend (faiss, chroma)
            embedding_dim: Embedding dimension
            index_type: Index type (flat, ivf, hnsw)
        """
        self.backend = backend
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self._index = None
        self._documents = []
        self._embeddings = None
    
    def _init_faiss_index(self, embeddings: np.ndarray):
        """Initialize FAISS index."""
        try:
            import faiss
        except ImportError:
            raise ImportError("Please install faiss-cpu: pip install faiss-cpu")
        
        dim = embeddings.shape[1]
        
        if self.index_type == "flat":
            self._index = faiss.IndexFlatIP(dim)  # Inner product (for normalized vectors)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            nlist = min(100, len(embeddings) // 10)
            self._index = faiss.IndexIVFFlat(quantizer, dim, max(1, nlist))
            self._index.train(embeddings.astype(np.float32))
        elif self.index_type == "hnsw":
            self._index = faiss.IndexHNSWFlat(dim, 32)
        
        self._index.add(embeddings.astype(np.float32))
    
    def _init_chroma_index(self, documents: list[dict], embeddings: np.ndarray):
        """Initialize ChromaDB collection."""
        try:
            import chromadb
        except ImportError:
            raise ImportError("Please install chromadb: pip install chromadb")
        
        client = chromadb.Client()
        self._collection = client.create_collection("rag_documents")
        
        ids = [str(i) for i in range(len(documents))]
        contents = [doc["content"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        
        self._collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
    
    def index(
        self,
        documents: list[dict],
        embeddings: np.ndarray
    ):
        """
        Index documents with embeddings.
        
        Args:
            documents: List of documents with 'content' and 'metadata'
            embeddings: Document embeddings
        """
        self._documents = documents
        self._embeddings = embeddings
        
        if self.backend == "faiss":
            self._init_faiss_index(embeddings)
        elif self.backend == "chroma":
            self._init_chroma_index(documents, embeddings)
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> SearchResults:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            
        Returns:
            SearchResults object
        """
        if self.backend == "faiss":
            return self._search_faiss(query_embedding, top_k)
        elif self.backend == "chroma":
            return self._search_chroma(query_embedding, top_k)
    
    def _search_faiss(
        self,
        query_embedding: np.ndarray,
        top_k: int
    ) -> SearchResults:
        """Search using FAISS."""
        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(query, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index
                results.append(SearchResult(
                    content=self._documents[idx]["content"],
                    score=float(score),
                    metadata=self._documents[idx].get("metadata", {}),
                    index=int(idx)
                ))
        
        return SearchResults(
            results=results,
            query="",
            method="vector_faiss"
        )
    
    def _search_chroma(
        self,
        query_embedding: np.ndarray,
        top_k: int
    ) -> SearchResults:
        """Search using ChromaDB."""
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        search_results = []
        for i, (doc, distance) in enumerate(zip(
            results["documents"][0],
            results["distances"][0]
        )):
            search_results.append(SearchResult(
                content=doc,
                score=1 - distance,  # Convert distance to similarity
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                index=int(results["ids"][0][i])
            ))
        
        return SearchResults(
            results=search_results,
            query="",
            method="vector_chroma"
        )


class BM25Search:
    """
    Sparse BM25 keyword search.
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75
    ):
        """
        Initialize BM25 search.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self._bm25 = None
        self._documents = []
        self._tokenized_corpus = []
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        import re
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def index(self, documents: list[dict]):
        """
        Index documents for BM25 search.
        
        Args:
            documents: List of documents with 'content'
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Please install rank-bm25: pip install rank-bm25")
        
        self._documents = documents
        self._tokenized_corpus = [
            self._tokenize(doc["content"])
            for doc in documents
        ]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> SearchResults:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            SearchResults object
        """
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append(SearchResult(
                content=self._documents[idx]["content"],
                score=float(scores[idx]),
                metadata=self._documents[idx].get("metadata", {}),
                index=int(idx)
            ))
        
        return SearchResults(
            results=results,
            query=query,
            method="bm25"
        )


class HybridSearch:
    """
    Hybrid search combining dense and sparse retrieval.
    """
    
    def __init__(
        self,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        fusion_method: str = "rrf",
        rrf_k: int = 60
    ):
        """
        Initialize hybrid search.
        
        Args:
            vector_weight: Weight for vector search
            bm25_weight: Weight for BM25 search
            fusion_method: Fusion method (rrf, weighted)
            rrf_k: RRF constant
        """
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        
        self.vector_search = VectorSearch()
        self.bm25_search = BM25Search()
        self._documents = []
    
    def index(
        self,
        documents: list[dict],
        embeddings: np.ndarray
    ):
        """
        Index documents for hybrid search.
        
        Args:
            documents: List of documents
            embeddings: Document embeddings
        """
        self._documents = documents
        self.vector_search.index(documents, embeddings)
        self.bm25_search.index(documents)
    
    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        fetch_k: int = 20
    ) -> SearchResults:
        """
        Perform hybrid search.
        
        Args:
            query: Text query
            query_embedding: Query embedding
            top_k: Final number of results
            fetch_k: Number to fetch from each source
            
        Returns:
            SearchResults object
        """
        # Get results from both sources
        vector_results = self.vector_search.search(query_embedding, fetch_k)
        bm25_results = self.bm25_search.search(query, fetch_k)
        
        # Fuse results
        if self.fusion_method == "rrf":
            fused = self._rrf_fusion(vector_results, bm25_results, top_k)
        else:
            fused = self._weighted_fusion(vector_results, bm25_results, top_k)
        
        return SearchResults(
            results=fused,
            query=query,
            method=f"hybrid_{self.fusion_method}"
        )
    
    def _rrf_fusion(
        self,
        vector_results: SearchResults,
        bm25_results: SearchResults,
        top_k: int
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion."""
        scores = {}
        doc_map = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results.results):
            doc_id = result.index
            rrf_score = 1 / (self.rrf_k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0) + self.vector_weight * rrf_score
            doc_map[doc_id] = result
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results.results):
            doc_id = result.index
            rrf_score = 1 / (self.rrf_k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0) + self.bm25_weight * rrf_score
            if doc_id not in doc_map:
                doc_map[doc_id] = result
        
        # Sort by fused score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, score in sorted_docs:
            original = doc_map[doc_id]
            results.append(SearchResult(
                content=original.content,
                score=score,
                metadata=original.metadata,
                index=doc_id
            ))
        
        return results
    
    def _weighted_fusion(
        self,
        vector_results: SearchResults,
        bm25_results: SearchResults,
        top_k: int
    ) -> list[SearchResult]:
        """Weighted score fusion."""
        scores = {}
        doc_map = {}
        
        # Normalize and combine scores
        max_vector = max((r.score for r in vector_results.results), default=1)
        max_bm25 = max((r.score for r in bm25_results.results), default=1)
        
        for result in vector_results.results:
            doc_id = result.index
            norm_score = result.score / max_vector if max_vector > 0 else 0
            scores[doc_id] = self.vector_weight * norm_score
            doc_map[doc_id] = result
        
        for result in bm25_results.results:
            doc_id = result.index
            norm_score = result.score / max_bm25 if max_bm25 > 0 else 0
            scores[doc_id] = scores.get(doc_id, 0) + self.bm25_weight * norm_score
            if doc_id not in doc_map:
                doc_map[doc_id] = result
        
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, score in sorted_docs:
            original = doc_map[doc_id]
            results.append(SearchResult(
                content=original.content,
                score=score,
                metadata=original.metadata,
                index=doc_id
            ))
        
        return results


class Reranker:
    """
    Cross-encoder reranking for improved precision.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize reranker.
        
        Args:
            model_name: Cross-encoder model name
        """
        self.model_name = model_name
        self._model = None
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError("Please install sentence-transformers")
            self._model = CrossEncoder(self.model_name)
        return self._model
    
    def rerank(
        self,
        query: str,
        results: SearchResults,
        top_k: Optional[int] = None
    ) -> SearchResults:
        """
        Rerank search results.
        
        Args:
            query: Original query
            results: Search results to rerank
            top_k: Number of results to return (None = all)
            
        Returns:
            Reranked SearchResults
        """
        if not results.results:
            return results
        
        model = self._load_model()
        
        # Create query-document pairs
        pairs = [(query, r.content) for r in results.results]
        
        # Get reranking scores
        scores = model.predict(pairs)
        
        # Sort by reranking score
        reranked = sorted(
            zip(results.results, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        if top_k:
            reranked = reranked[:top_k]
        
        new_results = []
        for result, score in reranked:
            new_results.append(SearchResult(
                content=result.content,
                score=float(score),
                metadata={**result.metadata, "original_score": result.score},
                index=result.index
            ))
        
        return SearchResults(
            results=new_results,
            query=query,
            method=f"{results.method}+rerank"
        )


if __name__ == "__main__":
    print("Search modules initialized successfully!")
    print("Available: VectorSearch, BM25Search, HybridSearch, Reranker")

