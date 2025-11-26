"""
Knowledge Graph RAG Implementation

Combines structured knowledge graphs with vector search
for entity-aware retrieval and reasoning.
"""

import os
import sys
import re
from typing import Optional
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv
import numpy as np

try:
    import networkx as nx
except ImportError:
    raise ImportError("Please install networkx: pip install networkx")

from utils.loaders import Document
from utils.chunking import RecursiveChunker
from utils.embeddings import EmbeddingModel
from utils.llm import get_llm, create_rag_prompt
from utils.search import VectorSearch

load_dotenv()


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    name: str
    entity_type: str
    attributes: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source: str
    relation_type: str
    target: str
    attributes: dict = field(default_factory=dict)


@dataclass
class GraphContext:
    """Context retrieved from the knowledge graph."""
    entities: list[Entity]
    relationships: list[Relationship]
    paths: list[list[str]]
    summary: str


@dataclass
class KGRAGResult:
    """Result from Knowledge Graph RAG."""
    question: str
    answer: str
    graph_context: GraphContext
    vector_context: list[str]
    entities_found: list[str]


class KnowledgeGraphRAG:
    """
    Knowledge Graph RAG combining graph traversal with vector search.
    
    Uses entity extraction and linking to navigate a knowledge graph
    while also retrieving relevant unstructured context.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gpt-4o",
        graph_backend: str = "networkx",
        max_hops: int = 2,
        top_k_entities: int = 5,
        top_k_chunks: int = 5
    ):
        """
        Initialize Knowledge Graph RAG.
        
        Args:
            embedding_model: Model for embeddings
            llm_model: LLM for generation and NER
            graph_backend: Graph storage (networkx, neo4j)
            max_hops: Maximum hops in graph traversal
            top_k_entities: Max entities to retrieve
            top_k_chunks: Max chunks from vector search
        """
        self.embedding_model = EmbeddingModel(embedding_model)
        self.llm = get_llm(llm_model)
        self.max_hops = max_hops
        self.top_k_entities = top_k_entities
        self.top_k_chunks = top_k_chunks
        
        self.chunker = RecursiveChunker(chunk_size=512)
        self.vector_search = VectorSearch(backend="faiss")
        
        # Initialize graph
        if graph_backend == "networkx":
            self.graph = nx.DiGraph()
        else:
            raise ValueError(f"Unsupported graph backend: {graph_backend}")
        
        self.entities: dict[str, Entity] = {}
        self.entity_embeddings: Optional[np.ndarray] = None
        self.entity_names: list[str] = []
        self.chunks = []
        self.indexed = False
    
    def add_entity(
        self,
        name: str,
        attributes: dict = None,
        entity_type: str = "Entity"
    ) -> Entity:
        """
        Add an entity to the knowledge graph.
        
        Args:
            name: Entity name
            attributes: Entity attributes
            entity_type: Type of entity
            
        Returns:
            Created Entity object
        """
        attributes = attributes or {}
        
        # Generate embedding for entity
        embedding = self.embedding_model.embed(name).embeddings[0]
        
        entity = Entity(
            name=name,
            entity_type=entity_type,
            attributes=attributes,
            embedding=embedding
        )
        
        self.entities[name] = entity
        self.graph.add_node(name, **{"type": entity_type, **attributes})
        
        # Update entity embeddings index
        self._rebuild_entity_index()
        
        return entity
    
    def add_relationship(
        self,
        source: str,
        relation_type: str,
        target: str,
        attributes: dict = None
    ) -> Relationship:
        """
        Add a relationship between entities.
        
        Args:
            source: Source entity name
            relation_type: Type of relationship
            target: Target entity name
            attributes: Relationship attributes
            
        Returns:
            Created Relationship object
        """
        attributes = attributes or {}
        
        # Auto-create entities if they don't exist
        if source not in self.entities:
            self.add_entity(source)
        if target not in self.entities:
            self.add_entity(target)
        
        self.graph.add_edge(source, target, relation=relation_type, **attributes)
        
        return Relationship(
            source=source,
            relation_type=relation_type,
            target=target,
            attributes=attributes
        )
    
    def _rebuild_entity_index(self):
        """Rebuild the entity embedding index."""
        if not self.entities:
            return
        
        self.entity_names = list(self.entities.keys())
        embeddings = [self.entities[name].embedding for name in self.entity_names]
        self.entity_embeddings = np.array(embeddings)
    
    def index_documents(self, documents: list):
        """
        Index documents for vector search.
        
        Args:
            documents: List of documents to index
        """
        normalized = []
        for doc in documents:
            if isinstance(doc, str):
                normalized.append({"content": doc, "metadata": {}})
            elif isinstance(doc, Document):
                normalized.append({"content": doc.content, "metadata": doc.metadata})
            else:
                normalized.append(doc)
        
        self.chunks = []
        for doc in normalized:
            doc_chunks = self.chunker.chunk(doc["content"], metadata=doc.get("metadata", {}))
            self.chunks.extend(doc_chunks)
        
        chunk_texts = [c.content for c in self.chunks]
        embeddings = self.embedding_model.embed(chunk_texts).embeddings
        
        chunk_docs = [{"content": c.content, "metadata": c.metadata} for c in self.chunks]
        self.vector_search.index(chunk_docs, embeddings)
        
        self.indexed = True
        print(f"Indexed {len(self.chunks)} chunks")
    
    def _extract_entities(self, text: str) -> list[str]:
        """
        Extract entity mentions from text using LLM.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of entity names
        """
        prompt = f"""Extract all named entities (people, organizations, products, locations, etc.) from the following text.
Return only the entity names, one per line.

Text: {text}

Entities:"""

        response = self.llm.generate(prompt).content
        entities = [e.strip() for e in response.strip().split('\n') if e.strip()]
        return entities
    
    def _link_entities(self, mentions: list[str]) -> list[str]:
        """
        Link entity mentions to graph nodes.
        
        Args:
            mentions: Entity mentions from text
            
        Returns:
            List of matched entity names from graph
        """
        if not self.entity_embeddings is not None or len(self.entity_names) == 0:
            return []
        
        linked = []
        
        for mention in mentions:
            # Generate embedding for mention
            mention_emb = self.embedding_model.embed(mention).embeddings[0]
            
            # Find most similar entity
            similarities = np.dot(self.entity_embeddings, mention_emb)
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            # Threshold for matching
            if best_score > 0.7:
                linked.append(self.entity_names[best_idx])
            
            # Also check exact match
            if mention in self.entities and mention not in linked:
                linked.append(mention)
        
        return list(set(linked))
    
    def _traverse_graph(self, entities: list[str]) -> GraphContext:
        """
        Traverse graph from given entities.
        
        Args:
            entities: Starting entity names
            
        Returns:
            GraphContext with retrieved information
        """
        found_entities = []
        found_relationships = []
        paths = []
        
        for entity_name in entities:
            if entity_name not in self.graph:
                continue
            
            found_entities.append(self.entities.get(entity_name))
            
            # Get outgoing relationships
            for _, target, data in self.graph.out_edges(entity_name, data=True):
                found_relationships.append(Relationship(
                    source=entity_name,
                    relation_type=data.get("relation", "RELATED_TO"),
                    target=target,
                    attributes=data
                ))
                
                if target in self.entities:
                    found_entities.append(self.entities[target])
                
                # 2-hop traversal
                if self.max_hops >= 2:
                    for _, target2, data2 in self.graph.out_edges(target, data=True):
                        found_relationships.append(Relationship(
                            source=target,
                            relation_type=data2.get("relation", "RELATED_TO"),
                            target=target2,
                            attributes=data2
                        ))
                        if target2 in self.entities:
                            found_entities.append(self.entities[target2])
            
            # Get incoming relationships
            for source, _, data in self.graph.in_edges(entity_name, data=True):
                found_relationships.append(Relationship(
                    source=source,
                    relation_type=data.get("relation", "RELATED_TO"),
                    target=entity_name,
                    attributes=data
                ))
                
                if source in self.entities:
                    found_entities.append(self.entities[source])
        
        # Find paths between entities if multiple
        if len(entities) >= 2:
            for i, e1 in enumerate(entities):
                for e2 in entities[i+1:]:
                    try:
                        path = nx.shortest_path(self.graph, e1, e2)
                        paths.append(path)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass
        
        # Build summary
        summary_parts = []
        for rel in found_relationships[:10]:  # Limit for prompt
            summary_parts.append(f"{rel.source} --[{rel.relation_type}]--> {rel.target}")
        
        summary = "\n".join(summary_parts) if summary_parts else "No graph relationships found."
        
        return GraphContext(
            entities=list({e.name: e for e in found_entities if e}.values())[:self.top_k_entities],
            relationships=found_relationships[:20],
            paths=paths[:5],
            summary=summary
        )
    
    def query(self, question: str) -> KGRAGResult:
        """
        Answer a question using knowledge graph + vector search.
        
        Args:
            question: The question to answer
            
        Returns:
            KGRAGResult with answer and context
        """
        # Extract entities from question
        mentions = self._extract_entities(question)
        print(f"Extracted mentions: {mentions}")
        
        # Link to graph entities
        linked_entities = self._link_entities(mentions)
        print(f"Linked entities: {linked_entities}")
        
        # Traverse graph
        graph_context = self._traverse_graph(linked_entities)
        
        # Vector search
        vector_context = []
        if self.indexed:
            query_embedding = self.embedding_model.embed(question).embeddings[0]
            results = self.vector_search.search(query_embedding, self.top_k_chunks)
            vector_context = [r.content for r in results.results]
        
        # Build combined context
        context_parts = []
        
        if graph_context.summary and graph_context.summary != "No graph relationships found.":
            context_parts.append("=== Knowledge Graph Facts ===")
            context_parts.append(graph_context.summary)
            
            # Add entity attributes
            for entity in graph_context.entities[:5]:
                if entity and entity.attributes:
                    attrs = ", ".join(f"{k}: {v}" for k, v in entity.attributes.items())
                    context_parts.append(f"{entity.name} ({entity.entity_type}): {attrs}")
        
        if vector_context:
            context_parts.append("\n=== Retrieved Documents ===")
            for i, chunk in enumerate(vector_context):
                context_parts.append(f"[{i+1}] {chunk}")
        
        combined_context = "\n".join(context_parts)
        
        # Generate answer
        system_prompt, user_prompt = create_rag_prompt(combined_context, question)
        response = self.llm.generate(user_prompt, system_prompt=system_prompt)
        
        return KGRAGResult(
            question=question,
            answer=response.content,
            graph_context=graph_context,
            vector_context=vector_context,
            entities_found=linked_entities
        )


def main():
    """Example usage of Knowledge Graph RAG."""
    
    # Initialize
    print("🚀 Initializing Knowledge Graph RAG...")
    rag = KnowledgeGraphRAG(
        embedding_model="all-MiniLM-L6-v2",
        llm_model="gpt-4o",
        max_hops=2
    )
    
    # Build knowledge graph
    print("\n📊 Building knowledge graph...")
    
    # Add entities
    rag.add_entity("Apple Inc", {"type": "Company", "industry": "Technology", "founded": "1976"})
    rag.add_entity("Tim Cook", {"type": "Person", "role": "CEO", "nationality": "American"})
    rag.add_entity("Steve Jobs", {"type": "Person", "role": "Co-founder", "nationality": "American"})
    rag.add_entity("iPhone", {"type": "Product", "category": "Smartphone", "launched": "2007"})
    rag.add_entity("Mac", {"type": "Product", "category": "Computer", "launched": "1984"})
    rag.add_entity("California", {"type": "Location", "country": "USA"})
    rag.add_entity("Microsoft", {"type": "Company", "industry": "Technology", "founded": "1975"})
    rag.add_entity("Satya Nadella", {"type": "Person", "role": "CEO", "nationality": "Indian-American"})
    
    # Add relationships
    rag.add_relationship("Tim Cook", "CEO_OF", "Apple Inc")
    rag.add_relationship("Steve Jobs", "FOUNDED", "Apple Inc")
    rag.add_relationship("Apple Inc", "PRODUCES", "iPhone")
    rag.add_relationship("Apple Inc", "PRODUCES", "Mac")
    rag.add_relationship("Apple Inc", "HEADQUARTERED_IN", "California")
    rag.add_relationship("Steve Jobs", "BORN_IN", "California")
    rag.add_relationship("Tim Cook", "SUCCEEDED", "Steve Jobs")
    rag.add_relationship("Satya Nadella", "CEO_OF", "Microsoft")
    rag.add_relationship("Apple Inc", "COMPETITOR_OF", "Microsoft")
    
    # Add documents
    documents = [
        {
            "content": """
            Apple Inc. reported record revenue of $123.9 billion in Q1 2024.
            CEO Tim Cook attributed the success to strong iPhone sales and 
            growth in the services segment. The company continues to innovate
            in AI and augmented reality technologies.
            """,
            "metadata": {"source": "earnings.pdf"}
        },
        {
            "content": """
            Tim Cook has been the CEO of Apple since 2011, following the 
            passing of co-founder Steve Jobs. Under his leadership, Apple
            has become the world's most valuable company, with a market cap
            exceeding $3 trillion.
            """,
            "metadata": {"source": "biography.pdf"}
        }
    ]
    
    print("\n📚 Indexing documents...")
    rag.index_documents(documents)
    
    # Test queries
    queries = [
        "Who is the CEO of Apple?",
        "What products does Apple make?",
        "Where is Apple headquartered?",
        "Who did Tim Cook succeed as CEO?"
    ]
    
    for question in queries:
        print(f"\n{'='*60}")
        print(f"❓ Question: {question}")
        print("=" * 60)
        
        result = rag.query(question)
        
        print(f"\n🔍 Entities found: {result.entities_found}")
        print(f"\n📊 Graph context:")
        print(f"   - {len(result.graph_context.entities)} entities")
        print(f"   - {len(result.graph_context.relationships)} relationships")
        
        if result.graph_context.summary:
            print(f"\n   Knowledge Graph Facts:")
            for line in result.graph_context.summary.split('\n')[:5]:
                print(f"     {line}")
        
        print("\n💡 Answer:")
        print("-" * 40)
        print(result.answer)


if __name__ == "__main__":
    main()

