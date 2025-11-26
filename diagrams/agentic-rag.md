# Agentic RAG Architecture Diagram

## Overview

Agentic RAG transforms the retrieval system into an autonomous agent capable of planning, tool use, and self-correction.

## Architecture Diagram

```mermaid
flowchart TB
    subgraph Input
        Q[🔍 User Query]
    end

    subgraph "Planning Agent"
        PA[Planner LLM]
        PLAN[Execution Plan]
        STEPS[Step Breakdown]
    end

    subgraph "Tool Belt"
        T1[🔍 Retrieval Tool]
        T2[🧮 Calculator]
        T3[🌐 Web Search]
        T4[📊 SQL Query]
        T5[🐍 Code Executor]
    end

    subgraph "Execution Agent"
        EA[Executor LLM]
        ACT[Action Execution]
        OBS[Observation]
    end

    subgraph "Reflection Agent"
        RA[Critic LLM]
        VAL{Valid Result?}
        FB[Feedback Loop]
    end

    subgraph "Memory"
        STM[Short-Term Memory]
        LTM[Long-Term Memory]
    end

    subgraph Output
        ANS[✅ Final Answer]
    end

    Q --> PA
    PA --> PLAN
    PLAN --> STEPS
    STEPS --> EA
    EA --> T1
    EA --> T2
    EA --> T3
    EA --> T4
    EA --> T5
    T1 --> ACT
    T2 --> ACT
    T3 --> ACT
    T4 --> ACT
    T5 --> ACT
    ACT --> OBS
    OBS --> STM
    OBS --> RA
    RA --> VAL
    VAL -->|No| FB
    FB --> PA
    VAL -->|Yes| ANS
    STM --> EA
    LTM --> PA

    style Q fill:#1a1a2e,stroke:#00d4ff,color:#fff
    style PA fill:#e94560,stroke:#fff,color:#fff
    style EA fill:#533483,stroke:#e94560,color:#fff
    style RA fill:#0f3460,stroke:#ffd700,color:#fff
    style ANS fill:#1a1a2e,stroke:#00ff88,color:#fff
```

## ReAct Pattern

```mermaid
flowchart LR
    subgraph "ReAct Loop"
        direction TB
        TH[💭 Thought]
        AC[⚡ Action]
        OB[👁️ Observation]
        
        TH --> AC
        AC --> OB
        OB --> TH
    end

    subgraph "Example"
        E1["Thought: I need to find revenue data"]
        E2["Action: search('Q4 2024 revenue')"]
        E3["Observation: Found $5.2B revenue"]
        E4["Thought: Now calculate YoY growth"]
        E5["Action: calculate('(5.2-4.8)/4.8')"]
        E6["Observation: 8.33% growth"]
        E7["Thought: I have enough info"]
        E8["Action: respond(answer)"]
    end

    TH -.-> E1
    AC -.-> E2
    OB -.-> E3
    E1 --> E2 --> E3 --> E4 --> E5 --> E6 --> E7 --> E8

    style TH fill:#e94560,stroke:#fff,color:#fff
    style AC fill:#00ff88,stroke:#1a1a2e,color:#1a1a2e
    style OB fill:#ffd700,stroke:#1a1a2e,color:#1a1a2e
```

## Tool Orchestration

```mermaid
flowchart TB
    subgraph "Query Analysis"
        QA[Query Analyzer]
        INT[Intent Detection]
        ENT[Entity Extraction]
    end

    subgraph "Tool Selection"
        TS[Tool Selector]
        MATCH{Best Tool?}
    end

    subgraph "Tools"
        direction LR
        RT[Retrieval]
        WT[Web Search]
        CT[Calculator]
        ST[SQL]
        PT[Python]
    end

    subgraph "Tool Execution"
        TE[Tool Executor]
        RES[Result Parser]
        ERR[Error Handler]
    end

    QA --> INT
    INT --> ENT
    ENT --> TS
    TS --> MATCH
    MATCH -->|"Knowledge Query"| RT
    MATCH -->|"Current Events"| WT
    MATCH -->|"Math"| CT
    MATCH -->|"Data Query"| ST
    MATCH -->|"Complex Logic"| PT
    RT --> TE
    WT --> TE
    CT --> TE
    ST --> TE
    PT --> TE
    TE --> RES
    TE --> ERR
    ERR --> TS

    style MATCH fill:#533483,stroke:#e94560,color:#fff
    style TE fill:#e94560,stroke:#fff,color:#fff
```

## Self-Correction Loop

```mermaid
flowchart TB
    subgraph "Generation"
        G1[Initial Answer]
    end

    subgraph "Validation"
        V1[Fact Checker]
        V2[Consistency Check]
        V3[Completeness Check]
        V4{All Passed?}
    end

    subgraph "Correction"
        C1[Error Identification]
        C2[Correction Strategy]
        C3[Re-retrieval]
        C4[Answer Refinement]
    end

    subgraph "Output"
        O1[Validated Answer]
    end

    G1 --> V1
    G1 --> V2
    G1 --> V3
    V1 --> V4
    V2 --> V4
    V3 --> V4
    V4 -->|No| C1
    C1 --> C2
    C2 --> C3
    C3 --> C4
    C4 --> V1
    V4 -->|Yes| O1

    style V4 fill:#533483,stroke:#e94560,color:#fff
    style O1 fill:#00ff88,stroke:#1a1a2e,color:#1a1a2e
```

## Agent Types Comparison

| Agent Type | Planning | Tools | Memory | Self-Correction |
|------------|----------|-------|--------|-----------------|
| Simple RAG | ❌ | ❌ | ❌ | ❌ |
| ReAct Agent | ✅ | ✅ | Short-term | ❌ |
| Plan-Execute | ✅✅ | ✅ | Short-term | ❌ |
| Reflexion | ✅ | ✅ | Long-term | ✅ |
| **Full Agentic** | ✅✅ | ✅✅ | Both | ✅✅ |

## When to Use

✅ **Use Agentic RAG when:**
- Complex multi-step reasoning needed
- Multiple tools/data sources required
- Query requires real-time data
- Self-correction is valuable
- Tasks require planning and decomposition

❌ **Avoid when:**
- Simple factual queries
- Latency is critical (<2s)
- Deterministic answers required
- Cost is primary concern
- Debugging complexity is an issue

