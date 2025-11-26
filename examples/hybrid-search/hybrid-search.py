"""
Hybrid Search RAG Implementation

Combines dense vector search with sparse BM25 retrieval
using Reciprocal Rank Fusion for superior results.
"""

import os
import sys
from typing import Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv
import numpy as np

from utils.loaders import Document
from utils.chunking import RecursiveChunker, Chunk
from utils.embeddings import EmbeddingModel
from utils.llm import get_llm, create_rag_prompt
from utils.search import VectorSearch, BM25Search, HybridSearch, Reranker, SearchResult

load_dotenv()


@dataclass
class HybridSearchResult:
    """Result from hybrid search RAG."""
    question: str
    answer: str
    retrieved_chunks: list[dict]
    search_method: str
    scores: dict


class HybridSearchRAG:
    """
    Hybrid Search RAG combining dense and sparse retrieval.
    
    Uses Reciprocal Rank Fusion to merge results from
    vector search and BM25 for optimal retrieval.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gpt-4o",
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
        use_reranker: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5,
        fetch_k: int = 20
    ):
        """
        Initialize Hybrid Search RAG.
        
        Args:
            embedding_model: Model for dense embeddings
            llm_model: LLM for answer generation
            vector_weight: Weight for vector search scores
            bm25_weight: Weight for BM25 scores
            rrf_k: Constant for RRF formula
            use_reranker: Whether to use cross-encoder reranking
            reranker_model: Cross-encoder model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k: Final number of results
            fetch_k: Number to fetch before reranking
        """
        self.embedding_model = EmbeddingModel(embedding_model)
        self.llm = get_llm(llm_model)
        
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self.top_k = top_k
        self.fetch_k = fetch_k
        
        self.use_reranker = use_reranker
        self.reranker = Reranker(reranker_model) if use_reranker else None
        
        self.chunker = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize search components
        self.vector_search = VectorSearch(backend="faiss")
        self.bm25_search = BM25Search()
        
        self.chunks: list[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.indexed = False
    
    def index_documents(self, documents: list[Document | dict | str]):
        """
        Index documents for hybrid search.
        
        Args:
            documents: List of documents to index
        """
        # Normalize input
        normalized = []
        for doc in documents:
            if isinstance(doc, str):
                normalized.append({"content": doc, "metadata": {}})
            elif isinstance(doc, Document):
                normalized.append({"content": doc.content, "metadata": doc.metadata})
            else:
                normalized.append(doc)
        
        # Chunk documents
        self.chunks = []
        for doc in normalized:
            doc_chunks = self.chunker.chunk(
                doc["content"],
                metadata=doc.get("metadata", {})
            )
            self.chunks.extend(doc_chunks)
        
        # Prepare documents for indexing
        chunk_docs = [
            {"content": c.content, "metadata": c.metadata}
            for c in self.chunks
        ]
        
        # Generate embeddings for vector search
        print("Generating embeddings...")
        chunk_texts = [c.content for c in self.chunks]
        self.embeddings = self.embedding_model.embed(chunk_texts).embeddings
        
        # Index for vector search
        print("Building vector index...")
        self.vector_search.index(chunk_docs, self.embeddings)
        
        # Index for BM25
        print("Building BM25 index...")
        self.bm25_search.index(chunk_docs)
        
        self.indexed = True
        print(f"Indexed {len(self.chunks)} chunks from {len(documents)} documents")
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult]
    ) -> list[SearchResult]:
        """
        Fuse results using Reciprocal Rank Fusion.
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            
        Returns:
            Fused and sorted results
        """
        scores = {}
        doc_map = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results):
            doc_id = result.index
            rrf_score = self.vector_weight / (self.rrf_k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            doc_map[doc_id] = result
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results):
            doc_id = result.index
            rrf_score = self.bm25_weight / (self.rrf_k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            if doc_id not in doc_map:
                doc_map[doc_id] = result
        
        # Sort by fused score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for doc_id, score in sorted_docs:
            original = doc_map[doc_id]
            fused_results.append(SearchResult(
                content=original.content,
                score=score,
                metadata={
                    **original.metadata,
                    "rrf_score": score,
                    "original_score": original.score
                },
                index=doc_id
            ))
        
        return fused_results
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> list[SearchResult]:
        """
        Perform hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results (uses default if None)
            
        Returns:
            List of search results
        """
        if not self.indexed:
            raise RuntimeError("No documents indexed. Call index_documents() first.")
        
        top_k = top_k or self.top_k
        
        # Get query embedding
        query_embedding = self.embedding_model.embed(query).embeddings[0]
        
        # Vector search
        vector_results = self.vector_search.search(
            query_embedding,
            self.fetch_k
        ).results
        
        # BM25 search
        bm25_results = self.bm25_search.search(query, self.fetch_k).results
        
        # Fuse results
        fused_results = self._reciprocal_rank_fusion(vector_results, bm25_results)
        
        # Rerank if enabled
        if self.use_reranker and self.reranker:
            from utils.search import SearchResults
            search_results = SearchResults(
                results=fused_results[:self.fetch_k],
                query=query,
                method="hybrid_rrf"
            )
            reranked = self.reranker.rerank(query, search_results, top_k)
            return reranked.results
        
        return fused_results[:top_k]
    
    def query(self, question: str) -> HybridSearchResult:
        """
        Answer a question using hybrid search RAG.
        
        Args:
            question: The question to answer
            
        Returns:
            HybridSearchResult with answer and retrieval info
        """
        # Search
        results = self.search(question)
        
        # Build context
        context = "\n\n---\n\n".join([
            f"[Source {i+1}]\n{r.content}"
            for i, r in enumerate(results)
        ])
        
        # Generate answer
        system_prompt, user_prompt = create_rag_prompt(context, question)
        response = self.llm.generate(user_prompt, system_prompt=system_prompt)
        
        # Collect scores for analysis
        scores = {
            "avg_rrf_score": np.mean([r.score for r in results]),
            "max_rrf_score": max(r.score for r in results),
            "min_rrf_score": min(r.score for r in results)
        }
        
        return HybridSearchResult(
            question=question,
            answer=response.content,
            retrieved_chunks=[
                {
                    "content": r.content[:200] + "...",
                    "score": r.score,
                    "metadata": r.metadata
                }
                for r in results
            ],
            search_method="hybrid_rrf" + ("+rerank" if self.use_reranker else ""),
            scores=scores
        )


def main():
    """Example usage of Hybrid Search RAG."""
    
    # Sample technical documentation
    documents = [
        {
            "content": """
            API Rate Limits
            
            The /users endpoint has a rate limit of 100 requests per minute.
            Exceeding this limit will result in a 429 Too Many Requests error.
            
            For authenticated requests, the limit is increased to 1000 requests
            per minute. Use the X-RateLimit-Remaining header to track usage.
            """,
            "metadata": {"source": "api_docs.md", "section": "rate-limits"}
        },
        {
            "content": """
            User Authentication
            
            All API requests must include an Authorization header with a valid
            Bearer token. Tokens can be obtained from the /auth/token endpoint.
            
            Example:
            curl -H "Authorization: Bearer YOUR_TOKEN" https://api.example.com/users
            
            Tokens expire after 24 hours and must be refreshed.
            """,
            "metadata": {"source": "api_docs.md", "section": "authentication"}
        },
        {
            "content": """
            Error Handling
            
            The API returns standard HTTP status codes:
            - 200 OK: Request successful
            - 400 Bad Request: Invalid parameters
            - 401 Unauthorized: Missing or invalid token
            - 429 Too Many Requests: Rate limit exceeded
            - 500 Internal Server Error: Server-side error
            
            Error responses include a JSON body with error details.
            """,
            "metadata": {"source": "api_docs.md", "section": "errors"}
        },
        {
            "content": """
            Database Optimization
            
            For large datasets, consider using pagination with the /users endpoint.
            The maximum page size is 100 items. Use cursor-based pagination for
            optimal performance with the 'cursor' query parameter.
            
            Example: GET /users?cursor=abc123&limit=50
            """,
            "metadata": {"source": "best_practices.md", "section": "optimization"}
        },
        {
            "content": """
            User Management API
            
            The /users endpoint supports CRUD operations:
            - GET /users: List all users (paginated)
            - GET /users/{id}: Get specific user
            - POST /users: Create new user
            - PUT /users/{id}: Update user
            - DELETE /users/{id}: Delete user
            
            All write operations require admin scope in the token.
            """,
            "metadata": {"source": "api_docs.md", "section": "users"}
        }
    ]
    
    # Initialize Hybrid Search RAG
    print("🚀 Initializing Hybrid Search RAG...")
    rag = HybridSearchRAG(
        embedding_model="all-MiniLM-L6-v2",
        llm_model="gpt-4o",
        vector_weight=0.5,
        bm25_weight=0.5,
        use_reranker=True,
        top_k=3
    )
    
    # Index documents
    print("\n📚 Indexing documents...")
    rag.index_documents(documents)
    
    # Test queries
    queries = [
        "What is the rate limit for the /users endpoint?",
        "How do I handle 429 errors?",
        "What authentication is required for API requests?"
    ]
    
    for question in queries:
        print(f"\n{'='*60}")
        print(f"❓ Question: {question}")
        print("=" * 60)
        
        result = rag.query(question)
        
        print(f"\n🔍 Search Method: {result.search_method}")
        print(f"📊 Avg RRF Score: {result.scores['avg_rrf_score']:.4f}")
        
        print("\n📄 Retrieved Chunks:")
        for i, chunk in enumerate(result.retrieved_chunks):
            print(f"  [{i+1}] Score: {chunk['score']:.4f}")
            print(f"      {chunk['content'][:100]}...")
        
        print("\n💡 Answer:")
        print("-" * 40)
        print(result.answer)


if __name__ == "__main__":
    main()

