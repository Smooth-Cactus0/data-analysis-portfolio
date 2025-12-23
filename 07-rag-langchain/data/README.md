# Data Directory

This directory contains data for the RAG system.

## Structure

```
data/
├── documents/      # Full document collection (Wikipedia articles)
├── sample_docs/    # Sample documents for quick demos
├── vectorstore/    # Persisted FAISS vector index
└── README.md       # This file
```

## Data Sources

### Sample Documents (`sample_docs/`)
Pre-written documents covering AI/ML topics:
- `machine_learning.txt` - ML fundamentals
- `deep_learning.txt` - Neural networks
- `nlp_overview.txt` - Natural Language Processing
- `rag_systems.txt` - RAG architecture
- `vector_databases.txt` - Vector stores
- `transformers.txt` - Transformer architecture
- `embeddings_explained.txt` - Text embeddings
- `llm_basics.txt` - Large Language Models

These documents are created by `scripts/fetch_documents.py` and provide a consistent dataset for demos.

### Wikipedia Documents (`documents/`)
Automatically fetched Wikipedia articles about:
- Machine Learning
- Deep Learning
- Natural Language Processing
- Neural Networks
- Transformers
- Large Language Models
- And more...

Run `python scripts/fetch_documents.py` to download.

### Vector Store (`vectorstore/`)
Persisted FAISS index created by `scripts/ingest.py`. Contains:
- `index.faiss` - Vector index file
- `index.pkl` - Document store pickle
- `metadata.json` - Ingestion metadata

## Generating Data

```bash
# Generate sample documents
python scripts/fetch_documents.py

# Build vector store
python scripts/ingest.py

# Evaluate system
python scripts/evaluate.py
```

## Notes

- Large files (vector stores, Wikipedia dumps) are `.gitignore`d
- Sample documents are version controlled for reproducibility
- Vector store can be regenerated with `ingest.py`
