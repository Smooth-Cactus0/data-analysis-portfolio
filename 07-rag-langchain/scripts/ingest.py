"""
Document Ingestion Script
=========================

Processes documents and builds the vector store for the RAG system.
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


def run_ingestion(
    documents_dir: str,
    output_dir: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_model: str = "all-MiniLM-L6-v2"
):
    """
    Run the full document ingestion pipeline.

    Args:
        documents_dir: Directory containing source documents
        output_dir: Directory to save the vector store
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        embedding_model: Name of the embedding model
    """
    from src.document_loader import DocumentLoader
    from src.chunker import TextChunker, ChunkingStrategy
    from src.embeddings import EmbeddingGenerator
    from src.vectorstore import VectorStoreManager, VectorStoreType

    documents_path = Path(documents_dir)
    output_path = Path(output_dir)

    logger.info("=" * 60)
    logger.info("RAG Document Ingestion Pipeline")
    logger.info("=" * 60)

    # Step 1: Load documents
    logger.info("\n[Step 1/4] Loading documents...")
    loader = DocumentLoader()

    if documents_path.is_dir():
        documents = loader.load_directory(documents_path)
    elif documents_path.is_file():
        documents = loader.load_file(documents_path)
    else:
        raise ValueError(f"Invalid path: {documents_path}")

    stats = loader.get_stats(documents)
    logger.info(f"  Loaded {stats['total_documents']} documents")
    logger.info(f"  Total characters: {stats['total_characters']:,}")
    logger.info(f"  Total words: {stats['total_words']:,}")

    if not documents:
        logger.error("No documents loaded. Exiting.")
        return

    # Step 2: Chunk documents
    logger.info("\n[Step 2/4] Chunking documents...")
    chunker = TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=ChunkingStrategy.RECURSIVE
    )
    chunks = chunker.chunk_documents(documents)

    chunk_stats = chunker.get_stats(chunks)
    logger.info(f"  Created {chunk_stats['total_chunks']} chunks")
    logger.info(f"  Average chunk size: {chunk_stats['avg_chunk_size']} chars")
    logger.info(f"  Min/Max chunk size: {chunk_stats['min_chunk_size']}/{chunk_stats['max_chunk_size']} chars")

    # Step 3: Generate embeddings and create vector store
    logger.info("\n[Step 3/4] Generating embeddings...")
    embedder = EmbeddingGenerator(model=embedding_model)

    embedding_info = embedder.get_info()
    logger.info(f"  Model: {embedding_info['model_name']}")
    logger.info(f"  Dimension: {embedding_info['dimension']}")
    logger.info(f"  Device: {embedding_info['device']}")

    # Create vector store
    logger.info("\n[Step 4/4] Building vector store...")
    vectorstore_manager = VectorStoreManager(
        embeddings=embedder.langchain_embeddings,
        store_type=VectorStoreType.FAISS
    )
    vectorstore_manager.create_from_documents(chunks)

    # Save vector store
    output_path.mkdir(parents=True, exist_ok=True)
    vectorstore_manager.save(output_path)

    # Save metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "source_directory": str(documents_path),
        "documents_loaded": stats["total_documents"],
        "chunks_created": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_info["dimension"],
        "store_type": "FAISS",
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("Ingestion Complete!")
    logger.info("=" * 60)
    logger.info(f"  Vector store saved to: {output_path}")
    logger.info(f"  Metadata saved to: {metadata_path}")
    logger.info(f"  Total chunks indexed: {len(chunks)}")

    return vectorstore_manager


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Document Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py                                    # Use defaults (sample_docs)
  python ingest.py -d ../data/documents              # Custom document directory
  python ingest.py --chunk-size 1000 --overlap 100   # Custom chunking
  python ingest.py -m all-mpnet-base-v2              # Different embedding model
        """
    )

    parser.add_argument(
        "-d", "--documents-dir",
        type=str,
        default=None,
        help="Directory containing documents to ingest"
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Directory to save the vector store"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Size of text chunks (default: 500)"
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap between chunks (default: 50)"
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model name (default: all-MiniLM-L6-v2)"
    )

    args = parser.parse_args()

    # Set default paths relative to project
    project_dir = Path(__file__).parent.parent

    documents_dir = args.documents_dir or str(project_dir / "data" / "sample_docs")
    output_dir = args.output_dir or str(project_dir / "data" / "vectorstore")

    # Check if documents exist
    if not Path(documents_dir).exists():
        logger.error(f"Documents directory not found: {documents_dir}")
        logger.info("Run 'python scripts/fetch_documents.py' first to create sample documents.")
        sys.exit(1)

    try:
        run_ingestion(
            documents_dir=documents_dir,
            output_dir=output_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            embedding_model=args.model
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()
