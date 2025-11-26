# Knowledge Graph RAG Architecture Diagram

## Overview

Knowledge Graph RAG combines structured knowledge graphs with vector search for entity-aware, relationship-rich retrieval.

## Architecture Diagram

```mermaid
flowchart TB
    subgraph Input
        Q[🔍 User Query]
    end

    subgraph "Entity Processing"
        NER[Named Entity Recognition]
        EL[Entity Linking]
        ED[Entity Disambiguation]
    end

    subgraph "Dual Retrieval"
        direction TB
        subgraph "Graph Path"
            KG[(Knowledge Graph)]
            GT[Graph Traversal]
            GP[Graph Patterns]
        end
        
        subgraph "Vector Path"
            VS[(Vector Store)]
            SIM[Similarity Search]
            TOP[Top-K Chunks]
        end
    end

    subgraph "Context Fusion"
        MERGE[Context Merger]
        RANK[Relevance Ranker]
        CTX[Unified Context]
    end

    subgraph "Generation"
        LLM[🤖 LLM]
        ANS[✅ Answer]
    end

    Q --> NER
    NER --> EL
    EL --> ED
    ED --> KG
    ED --> VS
    KG --> GT
    GT --> GP
    VS --> SIM
    SIM --> TOP
    GP --> MERGE
    TOP --> MERGE
    MERGE --> RANK
    RANK --> CTX
    CTX --> LLM
    LLM --> ANS

    style Q fill:#1a1a2e,stroke:#00d4ff,color:#fff
    style KG fill:#0f3460,stroke:#ff6b6b,color:#fff
    style VS fill:#0f3460,stroke:#00ff88,color:#fff
    style MERGE fill:#533483,stroke:#e94560,color:#fff
    style LLM fill:#e94560,stroke:#fff,color:#fff
    style ANS fill:#1a1a2e,stroke:#00ff88,color:#fff
```

## Knowledge Graph Structure

```mermaid
graph LR
    subgraph "Entities"
        E1((Apple Inc))
        E2((Tim Cook))
        E3((iPhone))
        E4((Steve Jobs))
        E5((California))
        E6((Technology))
    end

    E2 -->|CEO_OF| E1
    E4 -->|FOUNDED| E1
    E1 -->|PRODUCES| E3
    E1 -->|LOCATED_IN| E5
    E1 -->|INDUSTRY| E6
    E4 -->|BORN_IN| E5
    E2 -->|SUCCEEDED| E4

    style E1 fill:#e94560,stroke:#fff,color:#fff
    style E2 fill:#00ff88,stroke:#1a1a2e,color:#1a1a2e
    style E3 fill:#ffd700,stroke:#1a1a2e,color:#1a1a2e
    style E4 fill:#00ff88,stroke:#1a1a2e,color:#1a1a2e
    style E5 fill:#00d4ff,stroke:#1a1a2e,color:#1a1a2e
    style E6 fill:#533483,stroke:#fff,color:#fff
```

## Graph Traversal Patterns

```mermaid
flowchart TB
    subgraph "1-Hop Query"
        Q1["Who is the CEO of Apple?"]
        N1((Apple)) -->|CEO_OF| N2((Tim Cook))
    end

    subgraph "2-Hop Query"
        Q2["Where was Apple's founder born?"]
        N3((Apple)) -->|FOUNDED_BY| N4((Steve Jobs))
        N4 -->|BORN_IN| N5((California))
    end

    subgraph "Multi-Hop Query"
        Q3["What products are made where Jobs was born?"]
        N6((Steve Jobs)) -->|BORN_IN| N7((California))
        N8((Apple)) -->|LOCATED_IN| N7
        N8 -->|PRODUCES| N9((iPhone))
    end

    style N2 fill:#00ff88,stroke:#1a1a2e,color:#1a1a2e
    style N5 fill:#00ff88,stroke:#1a1a2e,color:#1a1a2e
    style N9 fill:#00ff88,stroke:#1a1a2e,color:#1a1a2e
```

## Entity Extraction Pipeline

```mermaid
flowchart LR
    subgraph "NER"
        T1[Input Text]
        T2[Token Classification]
        T3[Entity Spans]
    end

    subgraph "Entity Linking"
        EL1[Candidate Generation]
        EL2[Entity Embeddings]
        EL3[Similarity Matching]
        EL4[Disambiguation]
    end

    subgraph "Graph Lookup"
        GL1[Entity ID]
        GL2[Node Retrieval]
        GL3[Neighbor Expansion]
    end

    T1 --> T2 --> T3
    T3 --> EL1 --> EL2 --> EL3 --> EL4
    EL4 --> GL1 --> GL2 --> GL3

    style T3 fill:#ffd700,stroke:#1a1a2e,color:#1a1a2e
    style EL4 fill:#e94560,stroke:#fff,color:#fff
    style GL3 fill:#00ff88,stroke:#1a1a2e,color:#1a1a2e
```

## Hybrid Context Assembly

```mermaid
flowchart TB
    subgraph "Graph Context"
        GC1["Entity: Apple Inc"]
        GC2["Type: Technology Company"]
        GC3["Relations: CEO(Tim Cook), Founded(1976)"]
        GC4["Products: iPhone, Mac, iPad"]
    end

    subgraph "Vector Context"
        VC1["Chunk 1: Apple Q4 earnings..."]
        VC2["Chunk 2: iPhone 16 features..."]
        VC3["Chunk 3: Tim Cook interview..."]
    end

    subgraph "Merged Context"
        MC["
        [STRUCTURED KNOWLEDGE]
        - Apple Inc (Tech Company)
        - CEO: Tim Cook
        - Products: iPhone, Mac, iPad
        
        [RELEVANT PASSAGES]
        1. Apple Q4 earnings showed...
        2. iPhone 16 introduces...
        3. Tim Cook stated...
        "]
    end

    GC1 --> MC
    GC2 --> MC
    GC3 --> MC
    GC4 --> MC
    VC1 --> MC
    VC2 --> MC
    VC3 --> MC

    style MC fill:#533483,stroke:#e94560,color:#fff
```

## Graph Database Options

| Database | Type | Scalability | Query Language | Best For |
|----------|------|-------------|----------------|----------|
| Neo4j | Native Graph | High | Cypher | Enterprise |
| NetworkX | In-Memory | Medium | Python | Prototyping |
| Amazon Neptune | Cloud | Very High | Gremlin/SPARQL | AWS Stack |
| TigerGraph | Native Graph | Very High | GSQL | Analytics |

## When to Use

✅ **Use Knowledge Graph RAG when:**
- Domain has rich entity relationships
- Questions involve multi-hop reasoning
- Structured knowledge is critical
- Consistency and accuracy matter
- Enterprise knowledge management

❌ **Avoid when:**
- Simple keyword queries
- Unstructured domains
- Rapid prototyping needed
- Graph construction is costly
- Real-time updates required

