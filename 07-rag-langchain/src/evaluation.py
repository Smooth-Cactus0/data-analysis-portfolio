"""
Evaluation Module
=================

RAG system evaluation using RAGAS metrics and custom metrics.
Measures retrieval quality, answer faithfulness, and relevance.
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    """Single evaluation sample."""

    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
        }


@dataclass
class EvaluationResult:
    """Evaluation results container."""

    # Overall scores
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

    # Per-sample results
    sample_results: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    num_samples: int = 0
    evaluation_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "num_samples": self.num_samples,
            "evaluation_time": self.evaluation_time,
        }

    def __str__(self) -> str:
        return (
            f"RAG Evaluation Results:\n"
            f"  Faithfulness: {self.faithfulness:.4f}\n"
            f"  Answer Relevancy: {self.answer_relevancy:.4f}\n"
            f"  Context Precision: {self.context_precision:.4f}\n"
            f"  Context Recall: {self.context_recall:.4f}\n"
            f"  Samples evaluated: {self.num_samples}"
        )


class SimpleEvaluator:
    """
    Simple RAG evaluator using heuristic metrics.

    Provides lightweight evaluation without requiring external LLM calls.
    Good for quick iteration during development.
    """

    def __init__(self, embeddings_generator=None):
        """
        Initialize the evaluator.

        Args:
            embeddings_generator: Optional EmbeddingGenerator for semantic metrics
        """
        self.embeddings = embeddings_generator

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: List[str],
        relevant_docs: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality.

        Args:
            query: Search query
            retrieved_docs: Retrieved document contents
            relevant_docs: Ground truth relevant documents

        Returns:
            Dict with precision, recall, f1
        """
        if not retrieved_docs or not relevant_docs:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Convert to sets for comparison (using content snippets)
        retrieved_set = set(doc[:200] for doc in retrieved_docs)
        relevant_set = set(doc[:200] for doc in relevant_docs)

        # Calculate metrics
        true_positives = len(retrieved_set & relevant_set)
        precision = true_positives / len(retrieved_set) if retrieved_set else 0
        recall = true_positives / len(relevant_set) if relevant_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def evaluate_answer_relevance(
        self,
        question: str,
        answer: str
    ) -> float:
        """
        Evaluate how relevant the answer is to the question.

        Uses keyword overlap as a simple heuristic.

        Args:
            question: Original question
            answer: Generated answer

        Returns:
            Relevance score (0-1)
        """
        if not answer or not question:
            return 0.0

        # Simple keyword overlap
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())

        # Remove common stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "what", "which", "who", "whom", "this", "that", "these",
            "those", "am", "it", "its", "i", "you", "he", "she", "we",
            "they", "me", "him", "her", "us", "them", "my", "your",
            "his", "our", "their", "and", "but", "or", "nor", "so",
            "yet", "both", "either", "neither", "not", "only", "own",
            "same", "than", "too", "very", "just", "how", "why", "when",
            "where", "?"
        }

        question_words -= stop_words
        answer_words -= stop_words

        if not question_words:
            return 0.5  # Neutral score if no meaningful words

        overlap = len(question_words & answer_words)
        relevance = overlap / len(question_words)

        return min(relevance, 1.0)

    def evaluate_faithfulness(
        self,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        Evaluate if the answer is faithful to the context.

        Checks if answer claims can be found in context.

        Args:
            answer: Generated answer
            contexts: Retrieved context documents

        Returns:
            Faithfulness score (0-1)
        """
        if not answer or not contexts:
            return 0.0

        # Combine contexts
        full_context = " ".join(contexts).lower()

        # Extract content words from answer
        answer_words = set(answer.lower().split())
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "i", "you", "it", "this", "that",
            "and", "or", "but", "in", "on", "at", "to", "for", "of",
            "with", "as", "by", "from", "not", "no", "yes", "."
        }
        answer_words -= stop_words

        if not answer_words:
            return 1.0  # Nothing to verify

        # Check how many answer words appear in context
        found = sum(1 for word in answer_words if word in full_context)
        faithfulness = found / len(answer_words)

        return faithfulness

    def evaluate_sample(self, sample: EvaluationSample) -> Dict[str, float]:
        """
        Evaluate a single sample.

        Args:
            sample: EvaluationSample to evaluate

        Returns:
            Dict with all metric scores
        """
        relevance = self.evaluate_answer_relevance(sample.question, sample.answer)
        faithfulness = self.evaluate_faithfulness(sample.answer, sample.contexts)

        return {
            "answer_relevancy": relevance,
            "faithfulness": faithfulness,
        }

    def evaluate_batch(
        self,
        samples: List[EvaluationSample]
    ) -> EvaluationResult:
        """
        Evaluate a batch of samples.

        Args:
            samples: List of EvaluationSample objects

        Returns:
            EvaluationResult with aggregated metrics
        """
        import time
        start_time = time.time()

        sample_results = []
        all_relevancy = []
        all_faithfulness = []

        for sample in samples:
            result = self.evaluate_sample(sample)
            sample_results.append({
                "question": sample.question,
                **result
            })
            all_relevancy.append(result["answer_relevancy"])
            all_faithfulness.append(result["faithfulness"])

        evaluation_time = time.time() - start_time

        return EvaluationResult(
            faithfulness=np.mean(all_faithfulness) if all_faithfulness else 0.0,
            answer_relevancy=np.mean(all_relevancy) if all_relevancy else 0.0,
            context_precision=0.0,  # Requires ground truth
            context_recall=0.0,     # Requires ground truth
            sample_results=sample_results,
            num_samples=len(samples),
            evaluation_time=evaluation_time,
        )


class RAGEvaluator:
    """
    Full RAG evaluator using RAGAS metrics.

    Requires RAGAS library for advanced metrics:
        - Faithfulness: Is the answer grounded in the context?
        - Answer Relevancy: Is the answer relevant to the question?
        - Context Precision: Are retrieved docs relevant?
        - Context Recall: Are all relevant docs retrieved?
    """

    def __init__(self, use_ragas: bool = True):
        """
        Initialize the evaluator.

        Args:
            use_ragas: Whether to use RAGAS library (falls back to simple if unavailable)
        """
        self.use_ragas = use_ragas
        self._ragas_available = self._check_ragas()

        if use_ragas and not self._ragas_available:
            logger.warning("RAGAS not available, falling back to simple evaluation")
            self.use_ragas = False

        self.simple_evaluator = SimpleEvaluator()

    def _check_ragas(self) -> bool:
        """Check if RAGAS library is available."""
        try:
            from ragas import evaluate
            return True
        except ImportError:
            return False

    def evaluate(
        self,
        samples: List[EvaluationSample]
    ) -> EvaluationResult:
        """
        Evaluate RAG system performance.

        Args:
            samples: List of evaluation samples

        Returns:
            EvaluationResult with all metrics
        """
        if self.use_ragas and self._ragas_available:
            return self._evaluate_with_ragas(samples)
        else:
            return self.simple_evaluator.evaluate_batch(samples)

    def _evaluate_with_ragas(
        self,
        samples: List[EvaluationSample]
    ) -> EvaluationResult:
        """Evaluate using RAGAS library."""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from datasets import Dataset

            # Prepare data for RAGAS
            data = {
                "question": [s.question for s in samples],
                "answer": [s.answer for s in samples],
                "contexts": [s.contexts for s in samples],
                "ground_truth": [s.ground_truth or "" for s in samples],
            }

            dataset = Dataset.from_dict(data)

            # Run evaluation
            result = evaluate(
                dataset=dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
            )

            return EvaluationResult(
                faithfulness=result["faithfulness"],
                answer_relevancy=result["answer_relevancy"],
                context_precision=result["context_precision"],
                context_recall=result["context_recall"],
                num_samples=len(samples),
            )

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            logger.info("Falling back to simple evaluation")
            return self.simple_evaluator.evaluate_batch(samples)


def create_evaluation_dataset(
    rag_chain,
    questions: List[str],
    ground_truths: Optional[List[str]] = None
) -> List[EvaluationSample]:
    """
    Create evaluation samples by running queries through RAG chain.

    Args:
        rag_chain: RAGChain instance
        questions: List of test questions
        ground_truths: Optional list of expected answers

    Returns:
        List of EvaluationSample objects
    """
    samples = []

    for i, question in enumerate(questions):
        result = rag_chain.query(question)

        sample = EvaluationSample(
            question=question,
            answer=result["answer"],
            contexts=[result["context"]] if result["context"] else [],
            ground_truth=ground_truths[i] if ground_truths else None,
        )
        samples.append(sample)

    return samples


def save_evaluation_results(
    results: EvaluationResult,
    output_path: str
) -> None:
    """Save evaluation results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)

    logger.info(f"Evaluation results saved to {output_path}")


def load_evaluation_results(input_path: str) -> EvaluationResult:
    """Load evaluation results from JSON file."""
    with open(input_path, "r") as f:
        data = json.load(f)

    return EvaluationResult(**data)


if __name__ == "__main__":
    print("=== RAG Evaluation Module ===\n")

    # Demo with simple evaluator
    evaluator = SimpleEvaluator()

    # Sample data
    sample = EvaluationSample(
        question="What is machine learning?",
        answer="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        contexts=[
            "Machine learning is a subset of AI that allows systems to learn from data without explicit programming.",
            "Deep learning is a type of machine learning using neural networks.",
        ],
    )

    result = evaluator.evaluate_sample(sample)
    print(f"Single sample evaluation:")
    print(f"  Answer Relevancy: {result['answer_relevancy']:.4f}")
    print(f"  Faithfulness: {result['faithfulness']:.4f}")

    # Batch evaluation
    samples = [
        sample,
        EvaluationSample(
            question="What is deep learning?",
            answer="Deep learning uses neural networks with multiple layers to learn from data.",
            contexts=["Deep learning is a machine learning technique using multi-layer neural networks."],
        ),
    ]

    batch_result = evaluator.evaluate_batch(samples)
    print(f"\nBatch evaluation:")
    print(batch_result)
