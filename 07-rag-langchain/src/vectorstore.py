"""
Vector Store Module
===================

Vector database management for RAG systems.
Supports FAISS and ChromaDB backends.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import shutil

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreType(Enum):
    """Available vector store backends."""

    FAISS = "faiss"
    CHROMA = "chroma"


@dataclass
class VectorStoreConfig:
    """Configuration for vector stores."""

    store_type: VectorStoreType = VectorStoreType.FAISS
    persist_directory: Optional[str] = None
    collection_name: str = "documents"

    # FAISS-specific
    faiss_index_type: str = "Flat"  # "Flat", "IVF", "HNSW"

    # ChromaDB-specific
    chroma_distance_fn: str = "cosine"  # "cosine", "l2", "ip"


class VectorStoreManager:
    """
    Unified interface for vector store operations.

    Supports both FAISS (fast, local) and ChromaDB (feature-rich, persistent).

    Example:
        >>> manager = VectorStoreManager(embeddings, store_type=VectorStoreType.FAISS)
        >>> manager.add_documents(documents)
        >>> results = manager.similarity_search("query", k=5)
        >>> manager.save("./vectorstore")
    """

    def __init__(
        self,
        embeddings: Embeddings,
        store_type: VectorStoreType = VectorStoreType.FAISS,
        config: Optional[VectorStoreConfig] = None
    ):
        """
        Initialize the vector store manager.

        Args:
            embeddings: LangChain embeddings object
            store_type: Type of vector store to use
            config: Optional VectorStoreConfig for advanced settings
        """
        self.embeddings = embeddings

        if config:
            self.config = config
        else:
            self.config = VectorStoreConfig(store_type=store_type)

        self._store: Optional[VectorStore] = None
        self._documents: List[Document] = []

    @property
    def store(self) -> Optional[VectorStore]:
        """Get the underlying vector store."""
        return self._store

    @property
    def is_initialized(self) -> bool:
        """Check if the store has been initialized with documents."""
        return self._store is not None

    def create_from_documents(self, documents: List[Document]) -> VectorStore:
        """
        Create a new vector store from documents.

        Args:
            documents: List of Document objects to index

        Returns:
            The created VectorStore
        """
        if not documents:
            raise ValueError("Cannot create store from empty document list")

        logger.info(
            f"Creating {self.config.store_type.value} store with "
            f"{len(documents)} documents"
        )

        if self.config.store_type == VectorStoreType.FAISS:
            self._store = self._create_faiss(documents)
        elif self.config.store_type == VectorStoreType.CHROMA:
            self._store = self._create_chroma(documents)
        else:
            raise ValueError(f"Unknown store type: {self.config.store_type}")

        self._documents = documents
        logger.info(f"Vector store created successfully")
        return self._store

    def _create_faiss(self, documents: List[Document]) -> VectorStore:
        """Create a FAISS vector store."""
        from langchain_community.vectorstores import FAISS

        return FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )

    def _create_chroma(self, documents: List[Document]) -> VectorStore:
        """Create a ChromaDB vector store."""
        from langchain_community.vectorstores import Chroma

        return Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.config.collection_name,
            persist_directory=self.config.persist_directory,
            collection_metadata={"hnsw:space": self.config.chroma_distance_fn},
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to an existing store.

        Args:
            documents: Documents to add

        Returns:
            List of document IDs
        """
        if not self.is_initialized:
            return self.create_from_documents(documents)

        logger.info(f"Adding {len(documents)} documents to existing store")

        ids = self._store.add_documents(documents)
        self._documents.extend(documents)

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            **kwargs: Additional search parameters

        Returns:
            List of similar Document objects
        """
        if not self.is_initialized:
            raise RuntimeError("Vector store not initialized. Add documents first.")

        return self._store.similarity_search(query, k=k, filter=filter, **kwargs)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with relevance scores.

        Args:
            query: Search query
            k: Number of results to return
            **kwargs: Additional search parameters

        Returns:
            List of (Document, score) tuples
        """
        if not self.is_initialized:
            raise RuntimeError("Vector store not initialized. Add documents first.")

        return self._store.similarity_search_with_score(query, k=k, **kwargs)

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs
    ) -> List[Document]:
        """
        Search using Maximum Marginal Relevance for diversity.

        Args:
            query: Search query
            k: Number of results to return
            fetch_k: Number of candidates to fetch
            lambda_mult: Diversity parameter (0=max diversity, 1=max relevance)
            **kwargs: Additional parameters

        Returns:
            List of Document objects
        """
        if not self.is_initialized:
            raise RuntimeError("Vector store not initialized. Add documents first.")

        return self._store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )

    def save(self, directory: Union[str, Path]) -> None:
        """
        Save the vector store to disk.

        Args:
            directory: Directory to save to
        """
        if not self.is_initialized:
            raise RuntimeError("No store to save")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving vector store to {directory}")

        if self.config.store_type == VectorStoreType.FAISS:
            self._store.save_local(str(directory))
        elif self.config.store_type == VectorStoreType.CHROMA:
            # ChromaDB persists automatically if persist_directory is set
            if hasattr(self._store, "_client"):
                self._store._client.persist()

        logger.info("Vector store saved successfully")

    def load(self, directory: Union[str, Path]) -> VectorStore:
        """
        Load a vector store from disk.

        Args:
            directory: Directory to load from

        Returns:
            Loaded VectorStore
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        logger.info(f"Loading vector store from {directory}")

        if self.config.store_type == VectorStoreType.FAISS:
            from langchain_community.vectorstores import FAISS

            self._store = FAISS.load_local(
                str(directory),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
        elif self.config.store_type == VectorStoreType.CHROMA:
            from langchain_community.vectorstores import Chroma

            self._store = Chroma(
                collection_name=self.config.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(directory),
            )

        logger.info("Vector store loaded successfully")
        return self._store

    def delete(self, directory: Optional[Union[str, Path]] = None) -> None:
        """
        Delete the vector store.

        Args:
            directory: Optional directory to delete
        """
        if directory:
            directory = Path(directory)
            if directory.exists():
                shutil.rmtree(directory)
                logger.info(f"Deleted vector store at {directory}")

        self._store = None
        self._documents = []

    def get_retriever(self, search_type: str = "similarity", **kwargs):
        """
        Get a LangChain retriever from the store.

        Args:
            search_type: "similarity" or "mmr"
            **kwargs: Additional retriever parameters

        Returns:
            LangChain Retriever object
        """
        if not self.is_initialized:
            raise RuntimeError("Vector store not initialized")

        search_kwargs = kwargs.get("search_kwargs", {"k": 4})

        return self._store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        stats = {
            "store_type": self.config.store_type.value,
            "is_initialized": self.is_initialized,
            "document_count": len(self._documents),
        }

        if self.config.persist_directory:
            stats["persist_directory"] = self.config.persist_directory

        return stats


def create_vectorstore(
    documents: List[Document],
    embeddings: Embeddings,
    store_type: VectorStoreType = VectorStoreType.FAISS,
    persist_directory: Optional[str] = None
) -> VectorStoreManager:
    """
    Quick function to create a vector store.

    Args:
        documents: Documents to index
        embeddings: Embeddings object
        store_type: Type of store
        persist_directory: Optional directory to persist

    Returns:
        Initialized VectorStoreManager
    """
    config = VectorStoreConfig(
        store_type=store_type,
        persist_directory=persist_directory,
    )
    manager = VectorStoreManager(embeddings, config=config)
    manager.create_from_documents(documents)
    return manager


if __name__ == "__main__":
    from embeddings import EmbeddingGenerator, EmbeddingModel

    print("=== Vector Store Demo ===\n")

    # Create sample documents
    documents = [
        Document(
            page_content="Machine learning enables computers to learn from data.",
            metadata={"source": "intro.txt", "topic": "ml"}
        ),
        Document(
            page_content="Deep learning uses neural networks with multiple layers.",
            metadata={"source": "dl.txt", "topic": "dl"}
        ),
        Document(
            page_content="Natural language processing handles text and speech.",
            metadata={"source": "nlp.txt", "topic": "nlp"}
        ),
        Document(
            page_content="RAG combines retrieval with language model generation.",
            metadata={"source": "rag.txt", "topic": "rag"}
        ),
    ]

    # Initialize embeddings
    embedder = EmbeddingGenerator(model=EmbeddingModel.MINILM)
    embeddings = embedder.langchain_embeddings

    # Create FAISS store
    print("Creating FAISS vector store...")
    manager = VectorStoreManager(embeddings, store_type=VectorStoreType.FAISS)
    manager.create_from_documents(documents)

    print(f"Stats: {manager.get_stats()}\n")

    # Search
    query = "How do language models work?"
    print(f"Query: '{query}'\n")

    results = manager.similarity_search_with_score(query, k=2)
    print("Top results:")
    for doc, score in results:
        print(f"  Score: {score:.4f}")
        print(f"  Content: {doc.page_content}")
        print(f"  Metadata: {doc.metadata}\n")
