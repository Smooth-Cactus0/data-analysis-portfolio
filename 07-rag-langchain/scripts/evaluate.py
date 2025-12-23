"""
RAG Evaluation Script
=====================

Evaluates the RAG system using predefined test questions.
"""

import os
import sys
from pathlib import Path
import argparse
import logging
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Test questions for evaluation
TEST_QUESTIONS = [
    {
        "question": "What is machine learning?",
        "ground_truth": "Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed.",
        "topic": "machine learning"
    },
    {
        "question": "How do transformers work?",
        "ground_truth": "Transformers use self-attention mechanisms to process sequences in parallel, weighing the importance of different parts of the input.",
        "topic": "transformers"
    },
    {
        "question": "What is RAG?",
        "ground_truth": "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation, using retrieved documents as context for LLM responses.",
        "topic": "rag"
    },
    {
        "question": "What are word embeddings?",
        "ground_truth": "Word embeddings are dense vector representations of words that capture semantic meaning in vector space.",
        "topic": "embeddings"
    },
    {
        "question": "What is deep learning?",
        "ground_truth": "Deep learning is a subset of machine learning using neural networks with multiple hidden layers.",
        "topic": "deep learning"
    },
    {
        "question": "What is a vector database?",
        "ground_truth": "Vector databases are specialized systems for storing and querying high-dimensional vectors efficiently, enabling similarity search at scale.",
        "topic": "vector databases"
    },
    {
        "question": "What is natural language processing?",
        "ground_truth": "NLP is a field that enables computers to understand, interpret, and generate human language.",
        "topic": "nlp"
    },
    {
        "question": "What are large language models?",
        "ground_truth": "LLMs are neural networks trained on massive text corpora that can generate human-like text and perform various language tasks.",
        "topic": "llm"
    },
]


def run_evaluation(vectorstore_dir: str, output_file: str = None):
    """
    Run RAG system evaluation.

    Args:
        vectorstore_dir: Directory containing the vector store
        output_file: Optional path to save results
    """
    from src.embeddings import EmbeddingGenerator
    from src.vectorstore import VectorStoreManager, VectorStoreType
    from src.retriever import RetrievalPipeline, RetrieverConfig
    from src.evaluation import SimpleEvaluator, EvaluationSample, EvaluationResult

    vectorstore_path = Path(vectorstore_dir)

    logger.info("=" * 60)
    logger.info("RAG System Evaluation")
    logger.info("=" * 60)

    # Load metadata
    metadata_path = vectorstore_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        logger.info(f"\nVector store metadata:")
        logger.info(f"  Created: {metadata.get('created_at', 'Unknown')}")
        logger.info(f"  Chunks indexed: {metadata.get('chunks_created', 'Unknown')}")
        logger.info(f"  Embedding model: {metadata.get('embedding_model', 'Unknown')}")

    # Load vector store
    logger.info("\n[Step 1/3] Loading vector store...")
    embedder = EmbeddingGenerator(model="all-MiniLM-L6-v2")

    vectorstore_manager = VectorStoreManager(
        embeddings=embedder.langchain_embeddings,
        store_type=VectorStoreType.FAISS
    )
    vectorstore_manager.load(vectorstore_path)
    logger.info("  Vector store loaded successfully")

    # Create retrieval pipeline
    config = RetrieverConfig(top_k=4)
    retrieval_pipeline = RetrievalPipeline(vectorstore_manager, config=config)

    # Run queries and collect results
    logger.info("\n[Step 2/3] Running test queries...")
    evaluation_samples = []

    for i, test in enumerate(TEST_QUESTIONS, 1):
        question = test["question"]
        ground_truth = test["ground_truth"]

        logger.info(f"\n  Query {i}/{len(TEST_QUESTIONS)}: {question}")

        # Retrieve documents
        results = retrieval_pipeline.retrieve_with_scores(question, k=4)

        if results:
            # Get context and create simple answer
            contexts = [r["document"].page_content for r in results]
            top_score = results[0]["score"]

            # Create a simple answer from top context
            answer = contexts[0][:300] + "..." if len(contexts[0]) > 300 else contexts[0]

            logger.info(f"    Retrieved {len(results)} docs (top score: {top_score:.4f})")
        else:
            contexts = []
            answer = "No relevant information found."
            logger.info("    No relevant documents found")

        sample = EvaluationSample(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth
        )
        evaluation_samples.append(sample)

    # Evaluate results
    logger.info("\n[Step 3/3] Computing evaluation metrics...")
    evaluator = SimpleEvaluator()
    results = evaluator.evaluate_batch(evaluation_samples)

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"\n{results}")

    # Per-question breakdown
    logger.info("\nPer-Question Results:")
    logger.info("-" * 60)
    for i, sample_result in enumerate(results.sample_results):
        q = TEST_QUESTIONS[i]
        logger.info(f"\n  Q{i+1}: {q['question']}")
        logger.info(f"      Topic: {q['topic']}")
        logger.info(f"      Relevancy: {sample_result['answer_relevancy']:.4f}")
        logger.info(f"      Faithfulness: {sample_result['faithfulness']:.4f}")

    # Save results if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "evaluation_date": datetime.now().isoformat(),
            "vectorstore_path": str(vectorstore_path),
            "num_test_questions": len(TEST_QUESTIONS),
            "metrics": results.to_dict(),
            "per_question_results": results.sample_results,
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"\nResults saved to: {output_path}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG system performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-v", "--vectorstore-dir",
        type=str,
        default=None,
        help="Directory containing the vector store"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file for evaluation results (JSON)"
    )

    args = parser.parse_args()

    # Set default paths
    project_dir = Path(__file__).parent.parent
    vectorstore_dir = args.vectorstore_dir or str(project_dir / "data" / "vectorstore")

    # Check if vector store exists
    if not Path(vectorstore_dir).exists():
        logger.error(f"Vector store not found: {vectorstore_dir}")
        logger.info("Run 'python scripts/ingest.py' first to create the vector store.")
        sys.exit(1)

    try:
        run_evaluation(
            vectorstore_dir=vectorstore_dir,
            output_file=args.output
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
