"""
Embeddings Module
=================

Vector embedding generation for RAG systems.
Supports multiple embedding models and providers.
"""

import os
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModel(Enum):
    """Available embedding models."""

    # Sentence Transformers (local, free)
    MINILM = "all-MiniLM-L6-v2"              # 384 dims, fast
    MPNET = "all-mpnet-base-v2"              # 768 dims, better quality
    BGE_SMALL = "BAAI/bge-small-en-v1.5"    # 384 dims, good balance
    BGE_BASE = "BAAI/bge-base-en-v1.5"      # 768 dims, high quality
    E5_SMALL = "intfloat/e5-small-v2"       # 384 dims
    E5_BASE = "intfloat/e5-base-v2"         # 768 dims

    # Instructor models (task-specific)
    INSTRUCTOR = "hkunlp/instructor-base"    # 768 dims, instruction-tuned


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model_name: str = EmbeddingModel.MINILM.value
    device: str = "cpu"  # "cpu", "cuda", or "mps"
    normalize_embeddings: bool = True
    show_progress: bool = True
    batch_size: int = 32

    # Model-specific settings
    encode_kwargs: Optional[Dict[str, Any]] = None


class EmbeddingGenerator:
    """
    Generate vector embeddings from text.

    Supports local sentence-transformer models for cost-effective,
    private embedding generation.

    Example:
        >>> embedder = EmbeddingGenerator(model=EmbeddingModel.MINILM)
        >>> embeddings = embedder.embed_texts(["Hello", "World"])
        >>> embeddings = embedder.embed_documents(documents)
    """

    def __init__(
        self,
        model: Union[str, EmbeddingModel] = EmbeddingModel.MINILM,
        device: str = "cpu",
        config: Optional[EmbeddingConfig] = None
    ):
        """
        Initialize the embedding generator.

        Args:
            model: Model name or EmbeddingModel enum
            device: Device to use ("cpu", "cuda", "mps")
            config: Optional EmbeddingConfig for advanced settings
        """
        if config:
            self.config = config
        else:
            model_name = model.value if isinstance(model, EmbeddingModel) else model
            self.config = EmbeddingConfig(model_name=model_name, device=device)

        self._embeddings = self._create_embeddings()
        self._dimension = None

    def _create_embeddings(self) -> Embeddings:
        """Create the LangChain embeddings object."""
        encode_kwargs = self.config.encode_kwargs or {}

        if self.config.normalize_embeddings:
            encode_kwargs["normalize_embeddings"] = True

        model_kwargs = {"device": self.config.device}

        logger.info(f"Loading embedding model: {self.config.model_name}")

        return HuggingFaceEmbeddings(
            model_name=self.config.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            show_progress=self.config.show_progress,
        )

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        if self._dimension is None:
            # Generate a test embedding to get dimension
            test_embedding = self._embeddings.embed_query("test")
            self._dimension = len(test_embedding)
        return self._dimension

    @property
    def langchain_embeddings(self) -> Embeddings:
        """Get the underlying LangChain embeddings object."""
        return self._embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query.

        Args:
            text: Query text

        Returns:
            Embedding vector
        """
        return self._embeddings.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        logger.info(f"Generating embeddings for {len(texts)} texts")
        return self._embeddings.embed_documents(texts)

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Generate embeddings for Document objects.

        Args:
            documents: List of Document objects

        Returns:
            List of embedding vectors
        """
        texts = [doc.page_content for doc in documents]
        return self.embed_texts(texts)

    def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0-1 for normalized embeddings)
        """
        emb1 = np.array(self.embed_query(text1))
        emb2 = np.array(self.embed_query(text2))

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)

    def find_most_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find the most similar texts to a query.

        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of results to return

        Returns:
            List of dicts with text, score, and index
        """
        query_emb = np.array(self.embed_query(query))
        candidate_embs = np.array(self.embed_texts(candidates))

        # Compute similarities
        similarities = np.dot(candidate_embs, query_emb)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": candidates[idx],
                "score": float(similarities[idx]),
                "index": int(idx)
            })

        return results

    def get_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        return {
            "model_name": self.config.model_name,
            "dimension": self.dimension,
            "device": self.config.device,
            "normalize_embeddings": self.config.normalize_embeddings,
        }


class EmbeddingModelSelector:
    """Helper class to select the right embedding model for a use case."""

    RECOMMENDATIONS = {
        "speed": EmbeddingModel.MINILM,
        "quality": EmbeddingModel.BGE_BASE,
        "balanced": EmbeddingModel.BGE_SMALL,
        "multilingual": EmbeddingModel.E5_BASE,
    }

    @classmethod
    def recommend(cls, priority: str = "balanced") -> EmbeddingModel:
        """
        Get a model recommendation based on priority.

        Args:
            priority: One of "speed", "quality", "balanced", "multilingual"

        Returns:
            Recommended EmbeddingModel
        """
        return cls.RECOMMENDATIONS.get(priority, EmbeddingModel.BGE_SMALL)

    @classmethod
    def list_models(cls) -> Dict[str, Dict[str, Any]]:
        """List all available models with their properties."""
        return {
            EmbeddingModel.MINILM.value: {
                "dimension": 384,
                "size_mb": 90,
                "speed": "fast",
                "quality": "good",
            },
            EmbeddingModel.MPNET.value: {
                "dimension": 768,
                "size_mb": 420,
                "speed": "medium",
                "quality": "better",
            },
            EmbeddingModel.BGE_SMALL.value: {
                "dimension": 384,
                "size_mb": 130,
                "speed": "fast",
                "quality": "better",
            },
            EmbeddingModel.BGE_BASE.value: {
                "dimension": 768,
                "size_mb": 440,
                "speed": "medium",
                "quality": "best",
            },
        }


def get_embeddings(
    model: Union[str, EmbeddingModel] = EmbeddingModel.MINILM,
    device: str = "cpu"
) -> Embeddings:
    """
    Quick function to get a LangChain embeddings object.

    Args:
        model: Model name or EmbeddingModel enum
        device: Device to use

    Returns:
        LangChain Embeddings object
    """
    generator = EmbeddingGenerator(model=model, device=device)
    return generator.langchain_embeddings


if __name__ == "__main__":
    # Demo usage
    print("=== Embedding Generator Demo ===\n")

    # Initialize with a small, fast model
    embedder = EmbeddingGenerator(model=EmbeddingModel.MINILM)
    print(f"Model info: {embedder.get_info()}\n")

    # Test texts
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Natural language processing enables computers to understand text.",
        "The weather today is sunny and warm.",
    ]

    # Generate embeddings
    embeddings = embedder.embed_texts(texts)
    print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}\n")

    # Find similar texts
    query = "AI and machine learning technologies"
    print(f"Query: '{query}'\n")
    print("Most similar texts:")

    results = embedder.find_most_similar(query, texts, top_k=3)
    for i, result in enumerate(results, 1):
        print(f"  {i}. (score: {result['score']:.4f}) {result['text']}")
