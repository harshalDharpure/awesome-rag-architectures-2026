"""
RAG Evaluation Benchmark

Evaluates RAG systems on:
- Faithfulness (is the answer supported by context?)
- Answer Relevancy (does the answer address the question?)
- Context Relevancy (is the retrieved context relevant?)
"""

import os
import sys
from typing import Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv
load_dotenv()


@dataclass
class RAGEvalResult:
    """Results from RAG evaluation."""
    question: str
    answer: str
    context: str
    faithfulness: float
    answer_relevancy: float
    context_relevancy: float


def evaluate_faithfulness(answer: str, context: str, llm) -> float:
    """
    Evaluate if the answer is faithful to the context.
    
    Returns score from 0 to 1.
    """
    prompt = f"""Evaluate if the following answer is faithful to the given context.
A faithful answer only contains information that can be verified from the context.

Context:
{context}

Answer:
{answer}

Rate the faithfulness from 0 to 1:
- 1.0: Completely faithful, all claims supported by context
- 0.5: Partially faithful, some claims unsupported
- 0.0: Unfaithful, contains hallucinations

Respond with only a number between 0 and 1."""

    response = llm.generate(prompt).content.strip()
    
    try:
        score = float(response)
        return max(0, min(1, score))
    except ValueError:
        return 0.5


def evaluate_answer_relevancy(question: str, answer: str, llm) -> float:
    """
    Evaluate if the answer is relevant to the question.
    
    Returns score from 0 to 1.
    """
    prompt = f"""Evaluate if the following answer is relevant to the question.
A relevant answer directly addresses what was asked.

Question:
{question}

Answer:
{answer}

Rate the relevancy from 0 to 1:
- 1.0: Completely relevant, fully addresses the question
- 0.5: Partially relevant, addresses some aspects
- 0.0: Not relevant, doesn't address the question

Respond with only a number between 0 and 1."""

    response = llm.generate(prompt).content.strip()
    
    try:
        score = float(response)
        return max(0, min(1, score))
    except ValueError:
        return 0.5


def evaluate_context_relevancy(question: str, context: str, llm) -> float:
    """
    Evaluate if the retrieved context is relevant to the question.
    
    Returns score from 0 to 1.
    """
    prompt = f"""Evaluate if the following context is relevant for answering the question.
Relevant context contains information needed to answer the question.

Question:
{question}

Context:
{context}

Rate the context relevancy from 0 to 1:
- 1.0: Highly relevant, contains all needed information
- 0.5: Partially relevant, contains some useful information
- 0.0: Not relevant, no useful information

Respond with only a number between 0 and 1."""

    response = llm.generate(prompt).content.strip()
    
    try:
        score = float(response)
        return max(0, min(1, score))
    except ValueError:
        return 0.5


def evaluate_rag(
    question: str,
    answer: str,
    context: str,
    llm
) -> RAGEvalResult:
    """
    Run full RAG evaluation.
    
    Args:
        question: The original question
        answer: The generated answer
        context: The retrieved context
        llm: LLM wrapper for evaluation
        
    Returns:
        RAGEvalResult with all scores
    """
    faithfulness = evaluate_faithfulness(answer, context, llm)
    answer_relevancy = evaluate_answer_relevancy(question, answer, llm)
    context_relevancy = evaluate_context_relevancy(question, context, llm)
    
    return RAGEvalResult(
        question=question,
        answer=answer,
        context=context,
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        context_relevancy=context_relevancy
    )


def run_benchmark():
    """Run the RAG evaluation benchmark with sample data."""
    
    from utils.llm import get_llm
    
    print("=" * 60)
    print("RAG EVALUATION BENCHMARK")
    print("=" * 60)
    
    # Initialize evaluator LLM
    llm = get_llm("gpt-4o", temperature=0)
    
    # Test cases
    test_cases = [
        {
            "question": "What is the capital of France?",
            "context": "France is a country in Western Europe. Paris is the capital and largest city of France.",
            "answer": "The capital of France is Paris."
        },
        {
            "question": "What is the population of Tokyo?",
            "context": "Tokyo is the capital of Japan. It is one of the most populous cities in the world.",
            "answer": "Tokyo has a population of approximately 14 million people."  # Hallucination - not in context
        },
        {
            "question": "How does photosynthesis work?",
            "context": "The Eiffel Tower is a famous landmark in Paris, France.",
            "answer": "Plants convert sunlight into energy through photosynthesis."  # Irrelevant context
        },
        {
            "question": "What are the benefits of exercise?",
            "context": "Regular exercise has numerous health benefits including improved cardiovascular health, better mental health, weight management, and increased energy levels.",
            "answer": "Exercise provides many benefits including better heart health, improved mood, weight control, and more energy."
        }
    ]
    
    results = []
    
    for i, case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Q: {case['question'][:50]}...")
        
        result = evaluate_rag(
            case["question"],
            case["answer"],
            case["context"],
            llm
        )
        results.append(result)
        
        print(f"Faithfulness: {result.faithfulness:.2f}")
        print(f"Answer Relevancy: {result.answer_relevancy:.2f}")
        print(f"Context Relevancy: {result.context_relevancy:.2f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
    avg_answer_rel = sum(r.answer_relevancy for r in results) / len(results)
    avg_context_rel = sum(r.context_relevancy for r in results) / len(results)
    
    print(f"\nAverage Faithfulness: {avg_faithfulness:.2f}")
    print(f"Average Answer Relevancy: {avg_answer_rel:.2f}")
    print(f"Average Context Relevancy: {avg_context_rel:.2f}")
    print(f"\nOverall Score: {(avg_faithfulness + avg_answer_rel + avg_context_rel) / 3:.2f}")


if __name__ == "__main__":
    run_benchmark()

