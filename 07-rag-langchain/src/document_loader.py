"""
Document Loader Module
======================

Multi-format document loading for RAG system.
Supports PDF, TXT, DOCX, and web content.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
import logging

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    WebBaseLoader,
    UnstructuredWordDocumentLoader,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoaderConfig:
    """Configuration for document loaders."""

    # PDF settings
    pdf_extract_images: bool = False

    # Web settings
    web_timeout: int = 30
    web_verify_ssl: bool = True

    # Directory settings
    recursive: bool = True
    show_progress: bool = True

    # Text settings
    encoding: str = "utf-8"

    # Metadata
    add_source_metadata: bool = True


class DocumentLoader:
    """
    Unified document loader supporting multiple formats.

    Supported formats:
        - PDF (.pdf)
        - Text (.txt, .md)
        - Word (.docx)
        - Web pages (URLs)

    Example:
        >>> loader = DocumentLoader()
        >>> docs = loader.load_pdf("document.pdf")
        >>> docs = loader.load_directory("./documents")
        >>> docs = loader.load_urls(["https://example.com"])
    """

    def __init__(self, config: Optional[LoaderConfig] = None):
        """
        Initialize the document loader.

        Args:
            config: Optional LoaderConfig for customization
        """
        self.config = config or LoaderConfig()
        self._supported_extensions = {
            ".pdf": self._load_pdf,
            ".txt": self._load_text,
            ".md": self._load_text,
            ".docx": self._load_docx,
        }

    def load(self, source: Union[str, Path, List[str]]) -> List[Document]:
        """
        Load documents from any supported source.

        Automatically detects source type (file, directory, or URLs).

        Args:
            source: File path, directory path, URL, or list of URLs

        Returns:
            List of Document objects
        """
        if isinstance(source, list):
            # Assume list of URLs
            return self.load_urls(source)

        source = Path(source) if isinstance(source, str) else source

        if source.is_dir():
            return self.load_directory(source)
        elif source.is_file():
            return self.load_file(source)
        elif str(source).startswith(("http://", "https://")):
            return self.load_urls([str(source)])
        else:
            raise ValueError(f"Unknown source type: {source}")

    def load_file(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load a single file.

        Args:
            file_path: Path to the file

        Returns:
            List of Document objects
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = file_path.suffix.lower()

        if extension not in self._supported_extensions:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported: {list(self._supported_extensions.keys())}"
            )

        logger.info(f"Loading file: {file_path}")
        documents = self._supported_extensions[extension](file_path)

        if self.config.add_source_metadata:
            for doc in documents:
                doc.metadata["source_file"] = str(file_path.name)
                doc.metadata["source_path"] = str(file_path.absolute())
                doc.metadata["file_type"] = extension

        logger.info(f"Loaded {len(documents)} document(s) from {file_path.name}")
        return documents

    def load_directory(
        self,
        directory: Union[str, Path],
        glob_pattern: str = "**/*",
        extensions: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Load all supported documents from a directory.

        Args:
            directory: Path to directory
            glob_pattern: Pattern for file matching
            extensions: Optional list of extensions to include

        Returns:
            List of Document objects
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        extensions = extensions or list(self._supported_extensions.keys())
        all_documents = []

        for ext in extensions:
            pattern = f"{glob_pattern}{ext}" if not glob_pattern.endswith(ext) else glob_pattern
            files = list(directory.glob(pattern))

            for file_path in files:
                try:
                    docs = self.load_file(file_path)
                    all_documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")

        logger.info(f"Loaded {len(all_documents)} document(s) from {directory}")
        return all_documents

    def load_urls(
        self,
        urls: List[str],
        parser: str = "html.parser"
    ) -> List[Document]:
        """
        Load documents from web URLs.

        Args:
            urls: List of URLs to load
            parser: BeautifulSoup parser to use

        Returns:
            List of Document objects
        """
        logger.info(f"Loading {len(urls)} URL(s)")

        try:
            loader = WebBaseLoader(
                web_paths=urls,
                bs_kwargs={"parse_only": None}
            )
            documents = loader.load()
        except Exception as e:
            logger.error(f"Failed to load URLs: {e}")
            raise

        if self.config.add_source_metadata:
            for doc, url in zip(documents, urls):
                doc.metadata["source_url"] = url
                doc.metadata["file_type"] = "web"

        logger.info(f"Loaded {len(documents)} document(s) from URLs")
        return documents

    def _load_pdf(self, file_path: Path) -> List[Document]:
        """Load PDF file."""
        loader = PyPDFLoader(str(file_path))
        return loader.load()

    def _load_text(self, file_path: Path) -> List[Document]:
        """Load text file."""
        loader = TextLoader(str(file_path), encoding=self.config.encoding)
        return loader.load()

    def _load_docx(self, file_path: Path) -> List[Document]:
        """Load Word document."""
        loader = UnstructuredWordDocumentLoader(str(file_path))
        return loader.load()

    def get_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about loaded documents.

        Args:
            documents: List of Document objects

        Returns:
            Dictionary with document statistics
        """
        if not documents:
            return {"total_documents": 0}

        total_chars = sum(len(doc.page_content) for doc in documents)
        total_words = sum(len(doc.page_content.split()) for doc in documents)

        # Group by file type
        by_type = {}
        for doc in documents:
            file_type = doc.metadata.get("file_type", "unknown")
            by_type[file_type] = by_type.get(file_type, 0) + 1

        return {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_chars_per_doc": total_chars // len(documents),
            "avg_words_per_doc": total_words // len(documents),
            "documents_by_type": by_type,
        }


# Convenience function for quick loading
def load_documents(source: Union[str, Path, List[str]]) -> List[Document]:
    """
    Quick function to load documents from any source.

    Args:
        source: File path, directory path, or list of URLs

    Returns:
        List of Document objects
    """
    loader = DocumentLoader()
    return loader.load(source)


if __name__ == "__main__":
    # Demo usage
    loader = DocumentLoader()

    # Example: Load from a directory (if it exists)
    sample_dir = Path(__file__).parent.parent / "data" / "sample_docs"

    if sample_dir.exists():
        docs = loader.load_directory(sample_dir)
        stats = loader.get_stats(docs)
        print(f"Loaded documents stats: {stats}")
    else:
        print(f"Sample directory not found: {sample_dir}")
        print("Run scripts/fetch_documents.py to download sample documents")
