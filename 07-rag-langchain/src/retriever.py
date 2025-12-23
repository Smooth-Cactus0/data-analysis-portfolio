"""
Retriever Module
================

Advanced retrieval pipeline for RAG systems.
Includes hybrid search, reranking, and query expansion.
"""

from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Available retrieval strategies."""

    SIMILARITY = "similarity"           # Basic similarity search
    MMR = "mmr"                         # Maximum Marginal Relevance
    SIMILARITY_THRESHOLD = "threshold"  # Filter by similarity score


@dataclass
class RetrieverConfig:
    """Configuration for the retriever."""

    # Basic settings
    top_k: int = 4
    strategy: RetrievalStrategy = RetrievalStrategy.SIMILARITY

    # MMR settings
    fetch_k: int = 20
    lambda_mult: float = 0.5

    # Threshold settings
    score_threshold: float = 0.5

    # Query settings
    include_metadata: bool = True


class EnhancedRetriever(BaseRetriever):
    """
    Enhanced retriever with multiple search strategies.

    Features:
        - Multiple retrieval strategies (similarity, MMR, threshold)
        - Query preprocessing
        - Result post-processing
        - Metadata filtering

    Example:
        >>> retriever = EnhancedRetriever(vectorstore_manager)
        >>> docs = retriever.get_relevant_documents("What is machine learning?")
    """

    vectorstore_manager: Any  # VectorStoreManager
    config: RetrieverConfig = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        vectorstore_manager,
        config: Optional[RetrieverConfig] = None,
        **kwargs
    ):
        """
        Initialize the enhanced retriever.

        Args:
            vectorstore_manager: VectorStoreManager instance
            config: Optional RetrieverConfig
        """
        config = config or RetrieverConfig()
        super().__init__(vectorstore_manager=vectorstore_manager, config=config, **kwargs)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            run_manager: Optional callback manager

        Returns:
            List of relevant Document objects
        """
        # Preprocess query
        processed_query = self._preprocess_query(query)

        # Get documents based on strategy
        if self.config.strategy == RetrievalStrategy.SIMILARITY:
            docs = self._similarity_search(processed_query)
        elif self.config.strategy == RetrievalStrategy.MMR:
            docs = self._mmr_search(processed_query)
        elif self.config.strategy == RetrievalStrategy.SIMILARITY_THRESHOLD:
            docs = self._threshold_search(processed_query)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

        # Post-process results
        docs = self._postprocess_results(docs, query)

        return docs

    def _preprocess_query(self, query: str) -> str:
        """Preprocess the query before search."""
        # Basic cleaning
        query = query.strip()
        return query

    def _similarity_search(self, query: str) -> List[Document]:
        """Basic similarity search."""
        return self.vectorstore_manager.similarity_search(
            query,
            k=self.config.top_k
        )

    def _mmr_search(self, query: str) -> List[Document]:
        """Maximum Marginal Relevance search for diversity."""
        return self.vectorstore_manager.max_marginal_relevance_search(
            query,
            k=self.config.top_k,
            fetch_k=self.config.fetch_k,
            lambda_mult=self.config.lambda_mult
        )

    def _threshold_search(self, query: str) -> List[Document]:
        """Similarity search with score threshold."""
        results = self.vectorstore_manager.similarity_search_with_score(
            query,
            k=self.config.fetch_k
        )

        # Filter by threshold
        filtered = [
            doc for doc, score in results
            if score >= self.config.score_threshold
        ]

        return filtered[:self.config.top_k]

    def _postprocess_results(
        self,
        docs: List[Document],
        original_query: str
    ) -> List[Document]:
        """Post-process retrieved documents."""
        # Add retrieval metadata
        for i, doc in enumerate(docs):
            doc.metadata["retrieval_rank"] = i + 1
            doc.metadata["query"] = original_query

        return docs


class ContextualCompressor:
    """
    Compress retrieved documents to extract only relevant portions.

    Useful for long documents where only parts are relevant to the query.
    """

    def __init__(
        self,
        max_context_length: int = 2000,
        overlap: int = 100
    ):
        """
        Initialize the compressor.

        Args:
            max_context_length: Maximum characters per document
            overlap: Overlap when splitting
        """
        self.max_context_length = max_context_length
        self.overlap = overlap

    def compress(
        self,
        documents: List[Document],
        query: str
    ) -> List[Document]:
        """
        Compress documents to relevant portions.

        Args:
            documents: Documents to compress
            query: Query for relevance scoring

        Returns:
            Compressed documents
        """
        compressed = []

        for doc in documents:
            content = doc.page_content

            # If document is short enough, keep as is
            if len(content) <= self.max_context_length:
                compressed.append(doc)
                continue

            # Find most relevant portion based on query terms
            query_terms = set(query.lower().split())
            best_start = 0
            best_score = 0

            # Sliding window to find best portion
            for start in range(0, len(content) - self.max_context_length, self.overlap):
                window = content[start:start + self.max_context_length].lower()
                score = sum(1 for term in query_terms if term in window)

                if score > best_score:
                    best_score = score
                    best_start = start

            # Extract best portion
            compressed_content = content[best_start:best_start + self.max_context_length]

            compressed.append(Document(
                page_content=compressed_content,
                metadata={
                    **doc.metadata,
                    "compressed": True,
                    "original_length": len(content),
                    "compression_start": best_start,
                }
            ))

        return compressed


class RetrievalPipeline:
    """
    Full retrieval pipeline combining multiple components.

    Pipeline stages:
        1. Query preprocessing
        2. Document retrieval
        3. Result compression (optional)
        4. Reranking (optional)
        5. Post-processing
    """

    def __init__(
        self,
        vectorstore_manager,
        config: Optional[RetrieverConfig] = None,
        use_compression: bool = False,
        max_context_length: int = 2000
    ):
        """
        Initialize the retrieval pipeline.

        Args:
            vectorstore_manager: VectorStoreManager instance
            config: Retriever configuration
            use_compression: Whether to compress long documents
            max_context_length: Max length for compression
        """
        self.config = config or RetrieverConfig()
        self.vectorstore_manager = vectorstore_manager
        self.use_compression = use_compression

        self.retriever = EnhancedRetriever(
            vectorstore_manager=vectorstore_manager,
            config=self.config
        )

        if use_compression:
            self.compressor = ContextualCompressor(max_context_length=max_context_length)
        else:
            self.compressor = None

    def retrieve(
        self,
        query: str,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Execute the full retrieval pipeline.

        Args:
            query: Search query
            filter: Optional metadata filter

        Returns:
            List of relevant documents
        """
        logger.info(f"Retrieving documents for query: '{query[:50]}...'")

        # Retrieve documents
        docs = self.retriever.get_relevant_documents(query)

        # Apply compression if enabled
        if self.compressor and docs:
            docs = self.compressor.compress(docs, query)

        logger.info(f"Retrieved {len(docs)} documents")
        return docs

    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with relevance scores.

        Args:
            query: Search query
            k: Number of results (uses config.top_k if not specified)

        Returns:
            List of dicts with document and score
        """
        k = k or self.config.top_k

        results = self.vectorstore_manager.similarity_search_with_score(query, k=k)

        return [
            {
                "document": doc,
                "score": float(score),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata,
            }
            for doc, score in results
        ]

    def format_context(
        self,
        documents: List[Document],
        separator: str = "\n\n---\n\n"
    ) -> str:
        """
        Format retrieved documents as context string.

        Args:
            documents: Documents to format
            separator: Separator between documents

        Returns:
            Formatted context string
        """
        if not documents:
            return ""

        context_parts = []

        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", doc.metadata.get("source_file", f"Source {i}"))
            content = doc.page_content

            context_parts.append(f"[{source}]\n{content}")

        return separator.join(context_parts)

    def get_langchain_retriever(self) -> BaseRetriever:
        """Get the underlying LangChain retriever."""
        return self.retriever


def create_retriever(
    vectorstore_manager,
    top_k: int = 4,
    strategy: RetrievalStrategy = RetrievalStrategy.SIMILARITY
) -> RetrievalPipeline:
    """
    Quick function to create a retrieval pipeline.

    Args:
        vectorstore_manager: VectorStoreManager instance
        top_k: Number of documents to retrieve
        strategy: Retrieval strategy

    Returns:
        Configured RetrievalPipeline
    """
    config = RetrieverConfig(top_k=top_k, strategy=strategy)
    return RetrievalPipeline(vectorstore_manager, config=config)


if __name__ == "__main__":
    print("=== Retriever Module ===")
    print("This module requires a VectorStoreManager to function.")
    print("See chain.py for full pipeline usage.")
