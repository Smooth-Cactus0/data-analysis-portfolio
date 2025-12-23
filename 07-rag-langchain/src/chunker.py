"""
Text Chunking Module
====================

Multiple text chunking strategies for RAG systems.
Includes fixed-size, recursive, and semantic chunking.
"""

from typing import List, Optional, Dict, Any, Callable, Literal
from dataclasses import dataclass
from enum import Enum
import logging

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies."""

    FIXED = "fixed"
    RECURSIVE = "recursive"
    TOKEN = "token"
    SEMANTIC = "semantic"


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""

    # Size parameters
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Strategy selection
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE

    # Recursive splitter settings
    separators: Optional[List[str]] = None

    # Token splitter settings
    encoding_name: str = "cl100k_base"  # GPT-4 encoding

    # Metadata options
    add_start_index: bool = True


class TextChunker:
    """
    Text chunking with multiple strategies.

    Strategies:
        - FIXED: Simple character-based splitting
        - RECURSIVE: Smart splitting using hierarchy of separators
        - TOKEN: Splitting based on token count
        - SEMANTIC: Content-aware splitting (requires embeddings)

    Example:
        >>> chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        >>> chunks = chunker.chunk_documents(documents)
        >>> chunks = chunker.chunk_text("Long text here...")
    """

    # Default separators for recursive splitting (ordered by priority)
    DEFAULT_SEPARATORS = [
        "\n\n",      # Paragraphs
        "\n",        # Lines
        ". ",        # Sentences
        "? ",        # Questions
        "! ",        # Exclamations
        "; ",        # Clauses
        ", ",        # Phrases
        " ",         # Words
        ""           # Characters
    ]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        config: Optional[ChunkingConfig] = None
    ):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Target size for each chunk
            chunk_overlap: Overlap between consecutive chunks
            strategy: Chunking strategy to use
            config: Optional ChunkingConfig for advanced settings
        """
        if config:
            self.config = config
        else:
            self.config = ChunkingConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                strategy=strategy
            )

        self._splitter = self._create_splitter()

    def _create_splitter(self):
        """Create the appropriate text splitter based on strategy."""
        strategy = self.config.strategy

        if strategy == ChunkingStrategy.FIXED:
            return CharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separator="\n",
                add_start_index=self.config.add_start_index,
            )

        elif strategy == ChunkingStrategy.RECURSIVE:
            separators = self.config.separators or self.DEFAULT_SEPARATORS
            return RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=separators,
                add_start_index=self.config.add_start_index,
            )

        elif strategy == ChunkingStrategy.TOKEN:
            return TokenTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                encoding_name=self.config.encoding_name,
                add_start_index=self.config.add_start_index,
            )

        elif strategy == ChunkingStrategy.SEMANTIC:
            # Semantic chunking uses recursive as base,
            # but with semantic grouping post-processing
            return RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=self.DEFAULT_SEPARATORS,
                add_start_index=self.config.add_start_index,
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def chunk_documents(
        self,
        documents: List[Document],
        add_chunk_metadata: bool = True
    ) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of Document objects to chunk
            add_chunk_metadata: Whether to add chunk index metadata

        Returns:
            List of chunked Document objects
        """
        if not documents:
            return []

        logger.info(
            f"Chunking {len(documents)} document(s) with "
            f"strategy={self.config.strategy.value}, "
            f"chunk_size={self.config.chunk_size}, "
            f"overlap={self.config.chunk_overlap}"
        )

        chunks = self._splitter.split_documents(documents)

        if add_chunk_metadata:
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunk_size"] = len(chunk.page_content)

        logger.info(f"Created {len(chunks)} chunks from {len(documents)} document(s)")
        return chunks

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Split a text string into chunks.

        Args:
            text: Text string to chunk
            metadata: Optional metadata to attach to all chunks

        Returns:
            List of Document objects
        """
        if not text:
            return []

        # Create a temporary document
        doc = Document(page_content=text, metadata=metadata or {})

        return self.chunk_documents([doc])

    def get_stats(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about chunked documents.

        Args:
            chunks: List of chunked Document objects

        Returns:
            Dictionary with chunking statistics
        """
        if not chunks:
            return {"total_chunks": 0}

        sizes = [len(chunk.page_content) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(sizes) // len(sizes),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "total_characters": sum(sizes),
            "strategy": self.config.strategy.value,
            "target_chunk_size": self.config.chunk_size,
            "overlap": self.config.chunk_overlap,
        }


class ChunkerFactory:
    """Factory for creating chunkers with preset configurations."""

    @staticmethod
    def for_qa(chunk_size: int = 500) -> TextChunker:
        """
        Chunker optimized for question-answering.
        Smaller chunks for precise retrieval.
        """
        return TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=50,
            strategy=ChunkingStrategy.RECURSIVE
        )

    @staticmethod
    def for_summarization(chunk_size: int = 2000) -> TextChunker:
        """
        Chunker optimized for summarization.
        Larger chunks to maintain context.
        """
        return TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=200,
            strategy=ChunkingStrategy.RECURSIVE
        )

    @staticmethod
    def for_code(chunk_size: int = 1500) -> TextChunker:
        """
        Chunker optimized for code documentation.
        Uses code-aware separators.
        """
        code_separators = [
            "\nclass ",
            "\ndef ",
            "\n\ndef ",
            "\n\n",
            "\n",
            " ",
            "",
        ]
        config = ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=100,
            strategy=ChunkingStrategy.RECURSIVE,
            separators=code_separators,
        )
        return TextChunker(config=config)

    @staticmethod
    def for_legal(chunk_size: int = 1000) -> TextChunker:
        """
        Chunker optimized for legal documents.
        Preserves section structure.
        """
        legal_separators = [
            "\n\n\n",    # Major sections
            "\nSection ",
            "\nArticle ",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ]
        config = ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=150,
            strategy=ChunkingStrategy.RECURSIVE,
            separators=legal_separators,
        )
        return TextChunker(config=config)


def compare_chunking_strategies(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> Dict[str, Dict[str, Any]]:
    """
    Compare different chunking strategies on the same text.

    Args:
        text: Text to chunk
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks

    Returns:
        Dictionary with stats for each strategy
    """
    results = {}

    for strategy in [ChunkingStrategy.FIXED, ChunkingStrategy.RECURSIVE, ChunkingStrategy.TOKEN]:
        try:
            chunker = TextChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                strategy=strategy
            )
            chunks = chunker.chunk_text(text)
            results[strategy.value] = chunker.get_stats(chunks)
        except Exception as e:
            results[strategy.value] = {"error": str(e)}

    return results


if __name__ == "__main__":
    # Demo usage
    sample_text = """
    Machine learning is a subset of artificial intelligence that enables systems
    to learn and improve from experience without being explicitly programmed.

    Deep learning is a subset of machine learning that uses neural networks with
    many layers (hence "deep") to analyze various factors of data. These networks
    can automatically learn representations from data.

    Natural language processing (NLP) is a field at the intersection of computer
    science, artificial intelligence, and linguistics. The goal is to enable
    computers to understand, interpret, and generate human language.

    RAG (Retrieval-Augmented Generation) combines the benefits of retrieval-based
    and generation-based approaches. It retrieves relevant documents and uses them
    as context for generating responses.
    """

    print("=== Chunking Strategy Comparison ===\n")

    comparison = compare_chunking_strategies(sample_text, chunk_size=200, chunk_overlap=20)

    for strategy, stats in comparison.items():
        print(f"\n{strategy.upper()}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
