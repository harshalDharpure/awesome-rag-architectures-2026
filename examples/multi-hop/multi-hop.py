"""
Multi-Hop RAG Implementation

This module implements iterative retrieval and reasoning
for complex questions requiring multiple document sources.
"""

import os
import sys
from typing import Optional
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv
from utils.loaders import PDFLoader, WebLoader, Document
from utils.chunking import RecursiveChunker, Chunk
from utils.embeddings import EmbeddingModel, get_embeddings
from utils.llm import LLMWrapper, get_llm
from utils.search import VectorSearch, SearchResults

load_dotenv()


@dataclass
class HopResult:
    """Result from a single retrieval hop."""
    query: str
    retrieved_chunks: list[Chunk]
    analysis: str
    needs_more_info: bool
    sub_queries: list[str] = field(default_factory=list)


@dataclass
class MultiHopResult:
    """Final result from multi-hop RAG."""
    question: str
    answer: str
    hops: list[HopResult]
    total_chunks_used: int
    confidence: float


class MultiHopRAG:
    """
    Multi-Hop RAG for complex question answering.
    
    Iteratively retrieves and reasons across multiple documents
    until sufficient context is gathered to answer the question.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gpt-4o",
        max_hops: int = 3,
        top_k_per_hop: int = 5,
        confidence_threshold: float = 0.8,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Initialize Multi-Hop RAG.
        
        Args:
            embedding_model: Embedding model name
            llm_model: LLM model name
            max_hops: Maximum retrieval iterations
            top_k_per_hop: Documents to retrieve per hop
            confidence_threshold: Confidence level to stop retrieval
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.embedding_model = EmbeddingModel(embedding_model)
        self.llm = get_llm(llm_model)
        self.max_hops = max_hops
        self.top_k_per_hop = top_k_per_hop
        self.confidence_threshold = confidence_threshold
        
        self.chunker = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.vector_search = VectorSearch(backend="faiss")
        self.chunks: list[Chunk] = []
        self.indexed = False
    
    def index_documents(self, documents: list[Document | dict | str]):
        """
        Index documents for retrieval.
        
        Args:
            documents: List of Document objects, dicts with 'content', or strings
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
        
        # Generate embeddings
        chunk_texts = [c.content for c in self.chunks]
        embeddings = self.embedding_model.embed(chunk_texts).embeddings
        
        # Index
        chunk_docs = [
            {"content": c.content, "metadata": c.metadata}
            for c in self.chunks
        ]
        self.vector_search.index(chunk_docs, embeddings)
        self.indexed = True
        
        print(f"Indexed {len(self.chunks)} chunks from {len(documents)} documents")
    
    def _retrieve(self, query: str, top_k: int) -> list[Chunk]:
        """Retrieve relevant chunks for a query."""
        query_embedding = self.embedding_model.embed(query).embeddings[0]
        results = self.vector_search.search(query_embedding, top_k)
        
        retrieved_chunks = []
        for result in results.results:
            # Find matching chunk
            for chunk in self.chunks:
                if chunk.content == result.content:
                    retrieved_chunks.append(chunk)
                    break
        
        return retrieved_chunks
    
    def _analyze_context(
        self,
        question: str,
        context: str,
        hop_number: int
    ) -> tuple[str, bool, list[str], float]:
        """
        Analyze retrieved context and determine if more info is needed.
        
        Returns:
            Tuple of (analysis, needs_more_info, sub_queries, confidence)
        """
        prompt = f"""You are analyzing retrieved context to answer a question.

Question: {question}

Retrieved Context:
{context}

Current Hop: {hop_number}/{self.max_hops}

Analyze the context and respond in the following format:

ANALYSIS: [Your analysis of what information is available and what's missing]

CAN_ANSWER: [YES/NO - whether you have enough information to fully answer the question]

CONFIDENCE: [0.0-1.0 - your confidence in being able to answer]

SUB_QUERIES: [If CAN_ANSWER is NO, list 1-3 specific follow-up queries to gather missing information, one per line. If CAN_ANSWER is YES, write NONE]

Be thorough but concise."""

        response = self.llm.generate(prompt).content
        
        # Parse response
        lines = response.strip().split('\n')
        analysis = ""
        can_answer = False
        confidence = 0.5
        sub_queries = []
        
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith("ANALYSIS:"):
                current_section = "analysis"
                analysis = line.replace("ANALYSIS:", "").strip()
            elif line.startswith("CAN_ANSWER:"):
                current_section = "can_answer"
                can_answer = "YES" in line.upper()
            elif line.startswith("CONFIDENCE:"):
                current_section = "confidence"
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    confidence = 0.5
            elif line.startswith("SUB_QUERIES:"):
                current_section = "sub_queries"
                query_text = line.replace("SUB_QUERIES:", "").strip()
                if query_text and query_text.upper() != "NONE":
                    sub_queries.append(query_text)
            elif current_section == "analysis":
                analysis += " " + line
            elif current_section == "sub_queries" and line and line.upper() != "NONE":
                # Clean up numbered lists
                cleaned = line.lstrip("0123456789.-) ")
                if cleaned:
                    sub_queries.append(cleaned)
        
        needs_more_info = not can_answer and confidence < self.confidence_threshold
        
        return analysis, needs_more_info, sub_queries[:3], confidence
    
    def _generate_answer(
        self,
        question: str,
        all_context: str,
        hops: list[HopResult]
    ) -> str:
        """Generate final answer from accumulated context."""
        hop_summary = "\n".join([
            f"Hop {i+1}: {hop.query} -> Found {len(hop.retrieved_chunks)} relevant passages"
            for i, hop in enumerate(hops)
        ])
        
        prompt = f"""Based on the following context gathered through multiple retrieval steps, 
provide a comprehensive answer to the question.

Question: {question}

Retrieval Process:
{hop_summary}

Accumulated Context:
{all_context}

Instructions:
1. Synthesize information from all retrieved passages
2. Cite specific passages when making claims
3. Acknowledge any remaining uncertainty
4. Be comprehensive but concise

Answer:"""

        response = self.llm.generate(prompt)
        return response.content
    
    def query(self, question: str) -> MultiHopResult:
        """
        Answer a question using multi-hop retrieval.
        
        Args:
            question: The question to answer
            
        Returns:
            MultiHopResult with answer and retrieval trace
        """
        if not self.indexed:
            raise RuntimeError("No documents indexed. Call index_documents() first.")
        
        hops: list[HopResult] = []
        all_chunks: list[Chunk] = []
        current_query = question
        
        for hop_num in range(self.max_hops):
            print(f"\n🔍 Hop {hop_num + 1}: {current_query[:50]}...")
            
            # Retrieve
            retrieved = self._retrieve(current_query, self.top_k_per_hop)
            all_chunks.extend(retrieved)
            
            # Build context from all retrieved chunks so far
            context = "\n\n---\n\n".join([
                f"[Passage {i+1}]\n{chunk.content}"
                for i, chunk in enumerate(all_chunks)
            ])
            
            # Analyze
            analysis, needs_more, sub_queries, confidence = self._analyze_context(
                question, context, hop_num + 1
            )
            
            hop_result = HopResult(
                query=current_query,
                retrieved_chunks=retrieved,
                analysis=analysis,
                needs_more_info=needs_more,
                sub_queries=sub_queries
            )
            hops.append(hop_result)
            
            print(f"   Retrieved {len(retrieved)} chunks")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Needs more info: {needs_more}")
            
            # Check if we should stop
            if not needs_more or not sub_queries:
                print(f"✅ Stopping after hop {hop_num + 1}")
                break
            
            # Use first sub-query for next hop
            current_query = sub_queries[0]
        
        # Generate final answer
        print("\n📝 Generating final answer...")
        final_context = "\n\n---\n\n".join([
            f"[Passage {i+1}]\n{chunk.content}"
            for i, chunk in enumerate(all_chunks)
        ])
        
        answer = self._generate_answer(question, final_context, hops)
        
        return MultiHopResult(
            question=question,
            answer=answer,
            hops=hops,
            total_chunks_used=len(all_chunks),
            confidence=hops[-1].needs_more_info if hops else 0.0
        )


def main():
    """Example usage of Multi-Hop RAG."""
    
    # Sample documents for demonstration
    documents = [
        {
            "content": """
            TechCorp announced its Q4 2024 results on January 15, 2025. 
            The company reported revenue of $5.2 billion, up 12% year-over-year.
            CEO Jane Smith attributed the growth to strong performance in the 
            cloud services division, which saw 25% growth.
            """,
            "metadata": {"source": "earnings_report.pdf", "date": "2025-01-15"}
        },
        {
            "content": """
            In September 2024, TechCorp completed its acquisition of DataFlow Inc 
            for $800 million. DataFlow specializes in real-time data processing 
            and has 500 enterprise customers. The acquisition was expected to 
            add $200 million in annual recurring revenue.
            """,
            "metadata": {"source": "acquisition_news.pdf", "date": "2024-09-10"}
        },
        {
            "content": """
            DataFlow Inc was founded in 2018 and had grown to $180 million in 
            revenue by 2023. The company's flagship product, StreamPro, is used 
            by 40% of Fortune 500 companies for their data pipeline needs.
            Prior to the acquisition, DataFlow had raised $150 million in funding.
            """,
            "metadata": {"source": "dataflow_profile.pdf", "date": "2024-08-01"}
        },
        {
            "content": """
            TechCorp's cloud services division includes compute, storage, and 
            database offerings. In 2024, the division launched three new products:
            CloudCompute Pro, DataLake Enterprise, and AI Studio. These products
            contributed $400 million in new revenue.
            """,
            "metadata": {"source": "product_launch.pdf", "date": "2024-12-01"}
        },
        {
            "content": """
            Industry analysts predict that TechCorp's integration of DataFlow 
            will significantly enhance their data services offering. The combined
            entity is expected to capture 15% of the enterprise data market by 2026.
            Competitors include major cloud providers and specialized data companies.
            """,
            "metadata": {"source": "analyst_report.pdf", "date": "2025-01-20"}
        }
    ]
    
    # Initialize Multi-Hop RAG
    print("🚀 Initializing Multi-Hop RAG...")
    rag = MultiHopRAG(
        embedding_model="all-MiniLM-L6-v2",
        llm_model="gpt-4o",
        max_hops=3,
        top_k_per_hop=3
    )
    
    # Index documents
    print("\n📚 Indexing documents...")
    rag.index_documents(documents)
    
    # Complex question requiring multiple hops
    question = """
    How did TechCorp's acquisition of DataFlow impact their Q4 2024 revenue, 
    and what percentage of the revenue growth can be attributed to the acquisition 
    versus organic growth in their cloud services?
    """
    
    print(f"\n❓ Question: {question}")
    print("=" * 60)
    
    # Query
    result = rag.query(question)
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    
    print(f"\n🔢 Total hops: {len(result.hops)}")
    print(f"📄 Total chunks used: {result.total_chunks_used}")
    
    print("\n📜 Retrieval Trace:")
    for i, hop in enumerate(result.hops):
        print(f"\n  Hop {i+1}:")
        print(f"    Query: {hop.query[:60]}...")
        print(f"    Chunks retrieved: {len(hop.retrieved_chunks)}")
        print(f"    Analysis: {hop.analysis[:100]}...")
    
    print("\n" + "-" * 60)
    print("💡 ANSWER:")
    print("-" * 60)
    print(result.answer)


if __name__ == "__main__":
    main()

