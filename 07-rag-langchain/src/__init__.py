"""
RAG System with LangChain
=========================

A modular RAG (Retrieval-Augmented Generation) system for document Q&A.

Modules:
    - document_loader: Multi-format document loading (PDF, TXT, Web)
    - chunker: Text chunking strategies (fixed, recursive, semantic)
    - embeddings: Vector embedding generation
    - vectorstore: FAISS and ChromaDB management
    - retriever: Query and retrieval pipeline
    - chain: RAG chain assembly with LLM
    - evaluation: RAGAS quality metrics
"""

__version__ = "1.0.0"
__author__ = "Alexy Louis"
