"""
Agentic RAG Implementation

Autonomous RAG system with planning, tool use,
and self-correction capabilities using the ReAct pattern.
"""

import os
import sys
import json
import re
from typing import Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv
import numpy as np

from utils.loaders import Document
from utils.chunking import RecursiveChunker
from utils.embeddings import EmbeddingModel
from utils.llm import get_llm
from utils.search import VectorSearch

load_dotenv()


class ActionType(Enum):
    """Types of agent actions."""
    SEARCH = "search"
    CALCULATE = "calculate"
    WEB_SEARCH = "web_search"
    CODE_EXEC = "code_exec"
    RESPOND = "respond"
    THINK = "think"


@dataclass
class AgentAction:
    """Represents an agent action."""
    action_type: ActionType
    action_input: str
    thought: str


@dataclass
class AgentObservation:
    """Observation from executing an action."""
    action: AgentAction
    result: str
    success: bool


@dataclass
class ReasoningStep:
    """Single step in the reasoning trace."""
    step_num: int
    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class AgenticResult:
    """Result from agentic RAG."""
    question: str
    answer: str
    reasoning_trace: list[ReasoningStep]
    iterations: int
    tools_used: list[str]
    final_confidence: float


class Tool:
    """Base class for agent tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, input_str: str) -> str:
        raise NotImplementedError


class RetrievalTool(Tool):
    """Vector search retrieval tool."""
    
    def __init__(
        self,
        vector_search: VectorSearch,
        embedding_model: EmbeddingModel,
        chunks: list,
        top_k: int = 3
    ):
        super().__init__(
            "search",
            "Search the knowledge base for relevant information. Input should be a search query."
        )
        self.vector_search = vector_search
        self.embedding_model = embedding_model
        self.chunks = chunks
        self.top_k = top_k
    
    def execute(self, input_str: str) -> str:
        query_embedding = self.embedding_model.embed(input_str).embeddings[0]
        results = self.vector_search.search(query_embedding, self.top_k)
        
        if not results.results:
            return "No relevant documents found."
        
        output = []
        for i, result in enumerate(results.results):
            output.append(f"[{i+1}] {result.content}")
        
        return "\n\n".join(output)


class CalculatorTool(Tool):
    """Mathematical calculation tool."""
    
    def __init__(self):
        super().__init__(
            "calculate",
            "Perform mathematical calculations. Input should be a mathematical expression."
        )
    
    def execute(self, input_str: str) -> str:
        try:
            # Safe evaluation of mathematical expressions
            allowed_names = {
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "pow": pow, "sqrt": np.sqrt,
                "log": np.log, "log10": np.log10, "exp": np.exp,
                "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "pi": np.pi, "e": np.e
            }
            
            # Clean the input
            clean_input = input_str.replace('^', '**')
            
            result = eval(clean_input, {"__builtins__": {}}, allowed_names)
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation error: {str(e)}"


class WebSearchTool(Tool):
    """Simulated web search tool."""
    
    def __init__(self):
        super().__init__(
            "web_search",
            "Search the web for current information. Input should be a search query."
        )
    
    def execute(self, input_str: str) -> str:
        # In production, integrate with actual search API
        return f"[Simulated web search for: {input_str}] No results available in demo mode. Use the knowledge base search instead."


class CodeExecutorTool(Tool):
    """Python code execution tool."""
    
    def __init__(self):
        super().__init__(
            "code_exec",
            "Execute Python code for complex calculations. Input should be valid Python code."
        )
    
    def execute(self, input_str: str) -> str:
        try:
            # Create safe execution environment
            local_vars = {"np": np, "result": None}
            
            # Execute code
            exec(input_str, {"__builtins__": {"print": print, "range": range, "len": len}}, local_vars)
            
            if "result" in local_vars and local_vars["result"] is not None:
                return f"Code output: {local_vars['result']}"
            return "Code executed successfully (no explicit result)."
        except Exception as e:
            return f"Code execution error: {str(e)}"


class AgenticRAG:
    """
    Agentic RAG with planning, tool use, and self-correction.
    
    Implements the ReAct pattern for autonomous reasoning
    over a knowledge base with multiple tools.
    """
    
    def __init__(
        self,
        llm_model: str = "gpt-4o",
        embedding_model: str = "all-MiniLM-L6-v2",
        tools: list[str] = None,
        max_iterations: int = 5,
        chunk_size: int = 512
    ):
        """
        Initialize Agentic RAG.
        
        Args:
            llm_model: LLM model for reasoning
            embedding_model: Model for embeddings
            tools: List of tool names to enable
            max_iterations: Maximum reasoning iterations
            chunk_size: Size of text chunks
        """
        self.llm = get_llm(llm_model, temperature=0.3)
        self.embedding_model = EmbeddingModel(embedding_model)
        self.max_iterations = max_iterations
        
        self.chunker = RecursiveChunker(chunk_size=chunk_size)
        self.vector_search = VectorSearch(backend="faiss")
        self.chunks = []
        
        # Initialize tools
        self.tools: dict[str, Tool] = {}
        enabled_tools = tools or ["search", "calculate"]
        
        if "calculate" in enabled_tools:
            self.tools["calculate"] = CalculatorTool()
        if "web_search" in enabled_tools:
            self.tools["web_search"] = WebSearchTool()
        if "code_exec" in enabled_tools:
            self.tools["code_exec"] = CodeExecutorTool()
        
        self.indexed = False
    
    def index_documents(self, documents: list):
        """Index documents for retrieval."""
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
        
        # Add retrieval tool now that we have documents
        self.tools["search"] = RetrievalTool(
            self.vector_search, self.embedding_model, self.chunks
        )
        
        self.indexed = True
        print(f"Indexed {len(self.chunks)} chunks")
    
    def _build_tool_descriptions(self) -> str:
        """Build tool descriptions for the prompt."""
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        return "\n".join(descriptions)
    
    def _parse_action(self, response: str) -> Optional[AgentAction]:
        """Parse action from LLM response."""
        # Look for Thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        
        # Look for Action
        action_match = re.search(r'Action:\s*(\w+)\[(.+?)\]', response, re.DOTALL)
        
        if action_match:
            action_type_str = action_match.group(1).lower()
            action_input = action_match.group(2).strip()
            
            try:
                action_type = ActionType(action_type_str)
            except ValueError:
                action_type = ActionType.THINK
            
            return AgentAction(
                action_type=action_type,
                action_input=action_input,
                thought=thought
            )
        
        # Check for final answer
        if "Final Answer:" in response:
            answer_match = re.search(r'Final Answer:\s*(.+)', response, re.DOTALL)
            if answer_match:
                return AgentAction(
                    action_type=ActionType.RESPOND,
                    action_input=answer_match.group(1).strip(),
                    thought=thought
                )
        
        return None
    
    def _execute_action(self, action: AgentAction) -> AgentObservation:
        """Execute an action and return observation."""
        if action.action_type == ActionType.RESPOND:
            return AgentObservation(
                action=action,
                result=action.action_input,
                success=True
            )
        
        tool_name = action.action_type.value
        
        if tool_name in self.tools:
            try:
                result = self.tools[tool_name].execute(action.action_input)
                return AgentObservation(action=action, result=result, success=True)
            except Exception as e:
                return AgentObservation(
                    action=action,
                    result=f"Tool error: {str(e)}",
                    success=False
                )
        else:
            return AgentObservation(
                action=action,
                result=f"Unknown tool: {tool_name}",
                success=False
            )
    
    def _build_prompt(
        self,
        question: str,
        history: list[tuple[str, str, str]]
    ) -> str:
        """Build the reasoning prompt."""
        tool_desc = self._build_tool_descriptions()
        
        history_str = ""
        for thought, action, observation in history:
            history_str += f"\nThought: {thought}\nAction: {action}\nObservation: {observation}\n"
        
        prompt = f"""You are an intelligent assistant that answers questions by reasoning step-by-step and using tools.

Available Tools:
{tool_desc}
- respond: Provide the final answer. Use respond[your answer] when you have enough information.

Instructions:
1. Think about what information you need
2. Use tools to gather information
3. Reason about the results
4. Continue until you can provide a complete answer
5. Use respond[answer] to give your final answer

Always format your response as:
Thought: [your reasoning]
Action: [tool_name][input]

OR when ready to answer:
Thought: [final reasoning]
Final Answer: [your complete answer]

Question: {question}
{history_str}
"""
        return prompt
    
    def query(self, question: str) -> AgenticResult:
        """
        Answer a question using agentic reasoning.
        
        Args:
            question: The question to answer
            
        Returns:
            AgenticResult with answer and reasoning trace
        """
        if not self.indexed:
            raise RuntimeError("No documents indexed. Call index_documents() first.")
        
        history: list[tuple[str, str, str]] = []
        reasoning_trace: list[ReasoningStep] = []
        tools_used: list[str] = []
        
        for iteration in range(self.max_iterations):
            print(f"\n🔄 Iteration {iteration + 1}")
            
            # Build prompt and get LLM response
            prompt = self._build_prompt(question, history)
            response = self.llm.generate(prompt).content
            
            print(f"   Response: {response[:100]}...")
            
            # Parse action
            action = self._parse_action(response)
            
            if not action:
                print("   ⚠️ Could not parse action, retrying...")
                continue
            
            print(f"   Action: {action.action_type.value}[{action.action_input[:50]}...]")
            
            # Check for final answer
            if action.action_type == ActionType.RESPOND:
                reasoning_trace.append(ReasoningStep(
                    step_num=iteration + 1,
                    thought=action.thought,
                    action="respond",
                    action_input=action.action_input[:100],
                    observation="Final answer provided"
                ))
                
                return AgenticResult(
                    question=question,
                    answer=action.action_input,
                    reasoning_trace=reasoning_trace,
                    iterations=iteration + 1,
                    tools_used=list(set(tools_used)),
                    final_confidence=0.9
                )
            
            # Execute action
            observation = self._execute_action(action)
            
            print(f"   Observation: {observation.result[:100]}...")
            
            # Record tool usage
            tools_used.append(action.action_type.value)
            
            # Add to history
            action_str = f"{action.action_type.value}[{action.action_input}]"
            history.append((action.thought, action_str, observation.result))
            
            reasoning_trace.append(ReasoningStep(
                step_num=iteration + 1,
                thought=action.thought,
                action=action.action_type.value,
                action_input=action.action_input[:100],
                observation=observation.result[:200]
            ))
        
        # Max iterations reached - synthesize best answer
        final_prompt = f"""Based on your investigation, provide a final answer to the question.

Question: {question}

Your findings:
{chr(10).join([f"- {step.observation}" for step in reasoning_trace[-3:]])}

Final Answer:"""

        final_response = self.llm.generate(final_prompt).content
        
        return AgenticResult(
            question=question,
            answer=final_response,
            reasoning_trace=reasoning_trace,
            iterations=self.max_iterations,
            tools_used=list(set(tools_used)),
            final_confidence=0.7
        )


def main():
    """Example usage of Agentic RAG."""
    
    # Sample documents with financial data
    documents = [
        {
            "content": """
            Company X Financial Report 2024
            
            Q1 2024 Revenue: $4.2 billion
            Q2 2024 Revenue: $4.5 billion
            Q3 2024 Revenue: $4.8 billion
            Q4 2024 Revenue: $5.2 billion
            
            Total 2024 Revenue: $18.7 billion
            Year-over-year growth: 15%
            
            The growth was primarily driven by:
            - Cloud services expansion (40% of revenue)
            - Enterprise software (35% of revenue)
            - Consulting services (25% of revenue)
            """,
            "metadata": {"source": "annual_report.pdf", "year": "2024"}
        },
        {
            "content": """
            Company X Financial Report 2023
            
            Q1 2023 Revenue: $3.8 billion
            Q2 2023 Revenue: $4.0 billion
            Q3 2023 Revenue: $4.1 billion
            Q4 2023 Revenue: $4.4 billion
            
            Total 2023 Revenue: $16.3 billion
            Year-over-year growth: 12%
            """,
            "metadata": {"source": "annual_report.pdf", "year": "2023"}
        },
        {
            "content": """
            Company X Financial Report 2022
            
            Total 2022 Revenue: $14.5 billion
            Year-over-year growth: 18%
            
            Key highlights:
            - Launched new AI platform
            - Expanded to 15 new markets
            - Acquired 3 startups
            """,
            "metadata": {"source": "annual_report.pdf", "year": "2022"}
        }
    ]
    
    # Initialize Agentic RAG
    print("🚀 Initializing Agentic RAG...")
    rag = AgenticRAG(
        llm_model="gpt-4o",
        tools=["search", "calculate"],
        max_iterations=5
    )
    
    # Index documents
    print("\n📚 Indexing documents...")
    rag.index_documents(documents)
    
    # Complex query requiring reasoning
    question = """
    What is the compound annual growth rate (CAGR) of Company X's revenue 
    from 2022 to 2024? Show your calculation.
    """
    
    print(f"\n{'='*60}")
    print(f"❓ Question: {question}")
    print("=" * 60)
    
    result = rag.query(question)
    
    print(f"\n{'='*60}")
    print("📊 RESULTS")
    print("=" * 60)
    
    print(f"\n🔢 Total iterations: {result.iterations}")
    print(f"🛠️ Tools used: {', '.join(result.tools_used)}")
    print(f"📈 Confidence: {result.final_confidence:.0%}")
    
    print("\n📜 Reasoning Trace:")
    for step in result.reasoning_trace:
        print(f"\n  Step {step.step_num}:")
        print(f"    Thought: {step.thought[:80]}...")
        print(f"    Action: {step.action}[{step.action_input[:40]}...]")
        print(f"    Observation: {step.observation[:80]}...")
    
    print("\n" + "-" * 60)
    print("💡 FINAL ANSWER:")
    print("-" * 60)
    print(result.answer)


if __name__ == "__main__":
    main()

