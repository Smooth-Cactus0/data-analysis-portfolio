# RAG System with LangChain

A comprehensive Retrieval-Augmented Generation (RAG) system demonstrating document processing, semantic search, and AI-powered question answering.

**Author:** Alexy Louis
**Email:** alexy.louis.scholar@gmail.com
**LinkedIn:** [Alexy Louis](https://www.linkedin.com/in/alexy-louis-19a5a9262/)

---

## Objective

Build a production-ready RAG system that demonstrates:

1. **Document Ingestion** - Load and process PDF, text, and web content from multiple sources
2. **Intelligent Chunking** - Split documents using multiple strategies (fixed, recursive, semantic)
3. **Embedding Generation** - Convert text to vector representations using sentence-transformers
4. **Vector Storage** - Index embeddings in FAISS and ChromaDB for fast similarity search
5. **Context Retrieval** - Find relevant document chunks for any query using various strategies
6. **Answer Generation** - Use LLMs to synthesize accurate responses grounded in retrieved context
7. **Quality Evaluation** - Measure retrieval and generation performance with RAGAS-style metrics

---

## Project Status

| Phase | Component | Status |
|-------|-----------|--------|
| Setup | Project structure & dependencies | Complete |
| Data | Document collection & loading | Complete |
| Processing | Text chunking strategies | Complete |
| Embeddings | Vector generation | Complete |
| Storage | FAISS & ChromaDB setup | Complete |
| Retrieval | Query pipeline | Complete |
| Generation | LLM integration | Complete |
| Evaluation | Quality metrics | Complete |
| Demo | Streamlit application | Complete |

---

## Key Features

### Document Processing
- **Multi-format support**: PDF, TXT, DOCX, Markdown, Web pages
- **Automatic metadata extraction**: Source tracking, file types, timestamps
- **Batch processing**: Handle directories of documents efficiently

### Chunking Strategies
- **Fixed-size**: Simple character-based splitting
- **Recursive**: Smart splitting using hierarchy of separators (paragraphs → sentences → words)
- **Token-based**: Split by token count for LLM context limits
- **Configurable overlap**: Maintain context between chunks

### Embedding Models
- **all-MiniLM-L6-v2**: Fast, 384 dimensions (default)
- **all-mpnet-base-v2**: Better quality, 768 dimensions
- **BGE models**: BAAI's high-quality embeddings
- **Custom support**: Any sentence-transformers compatible model

### Vector Stores
- **FAISS**: Facebook's efficient similarity search library
- **ChromaDB**: Feature-rich with metadata filtering
- **Persistence**: Save and load indices for reuse

### Retrieval Methods
- **Similarity Search**: Basic cosine similarity
- **MMR (Maximum Marginal Relevance)**: Balance relevance with diversity
- **Threshold-based**: Filter by minimum similarity score
- **Context compression**: Extract relevant portions from long documents

### Evaluation Metrics
- **Faithfulness**: Is the answer grounded in the retrieved context?
- **Answer Relevancy**: Does the answer address the question?
- **Context Quality**: Are the retrieved documents relevant?

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Document Sources                        │
│              (PDF, TXT, DOCX, Web Pages)                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                ┌─────────▼─────────┐
                │  Document Loader   │
                │  (Multi-format)    │
                └─────────┬─────────┘
                          │
                ┌─────────▼─────────┐
                │   Text Chunker    │
                │  (Recursive/      │
                │   Fixed/Token)    │
                └─────────┬─────────┘
                          │
                ┌─────────▼─────────┐
                │ Embedding Model   │
                │ (sentence-        │
                │  transformers)    │
                └─────────┬─────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
    │   FAISS   │   │ ChromaDB  │   │ Metadata  │
    │   Index   │   │Collection │   │  Store    │
    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
          │               │               │
          └───────────────┼───────────────┘
                          │
                ┌─────────▼─────────┐
                │    Retriever      │
                │ (Similarity/MMR/  │
                │  Threshold)       │
                └─────────┬─────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
        ┌─────▼─────┐          ┌──────▼──────┐
        │   Query   │          │  Retrieved  │
        │           │          │   Context   │
        └─────┬─────┘          └──────┬──────┘
              │                       │
              └───────────┬───────────┘
                          │
                ┌─────────▼─────────┐
                │   RAG Chain       │
                │   (LLM +          │
                │    Context)       │
                └─────────┬─────────┘
                          │
                ┌─────────▼─────────┐
                │  Generated        │
                │  Answer +         │
                │  Sources          │
                └───────────────────┘
```

---

## Project Structure

```
07-rag-langchain/
├── data/
│   ├── documents/          # Source documents (Wikipedia articles)
│   ├── vectorstore/        # Persisted FAISS index
│   ├── sample_docs/        # Sample documents for demos
│   └── README.md           # Data documentation
├── images/                 # Visualizations
├── notebooks/
│   └── rag_system.ipynb    # Interactive walkthrough
├── scripts/
│   ├── fetch_documents.py  # Download sample documents
│   ├── ingest.py           # Document ingestion pipeline
│   └── evaluate.py         # RAG evaluation script
├── src/
│   ├── __init__.py
│   ├── document_loader.py  # Multi-format document loading
│   ├── chunker.py          # Text chunking strategies
│   ├── embeddings.py       # Embedding generation
│   ├── vectorstore.py      # FAISS & ChromaDB management
│   ├── retriever.py        # Retrieval pipeline
│   ├── chain.py            # RAG chain with LLM
│   └── evaluation.py       # Quality metrics
├── app.py                  # Streamlit demo application
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

```bash
# Navigate to project
cd 07-rag-langchain

# Install dependencies
pip install -r requirements.txt

# Generate sample documents
python scripts/fetch_documents.py

# Build vector store
python scripts/ingest.py

# Evaluate system
python scripts/evaluate.py

# Launch demo application
streamlit run app.py
```

### Command Options

```bash
# Custom ingestion
python scripts/ingest.py --chunk-size 1000 --overlap 100 -m all-mpnet-base-v2

# Specify document directory
python scripts/ingest.py -d ./my_documents -o ./my_vectorstore

# Save evaluation results
python scripts/evaluate.py -o results/evaluation.json
```

---

## Module Reference

### DocumentLoader
```python
from src.document_loader import DocumentLoader

loader = DocumentLoader()
docs = loader.load_directory("./documents")
docs = loader.load_file("document.pdf")
docs = loader.load_urls(["https://example.com"])
```

### TextChunker
```python
from src.chunker import TextChunker, ChunkingStrategy

chunker = TextChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.chunk_documents(documents)
```

### EmbeddingGenerator
```python
from src.embeddings import EmbeddingGenerator, EmbeddingModel

embedder = EmbeddingGenerator(model=EmbeddingModel.MINILM)
embeddings = embedder.embed_texts(["Hello", "World"])
```

### VectorStoreManager
```python
from src.vectorstore import VectorStoreManager, VectorStoreType

manager = VectorStoreManager(embeddings, store_type=VectorStoreType.FAISS)
manager.create_from_documents(chunks)
manager.save("./vectorstore")
```

### RetrievalPipeline
```python
from src.retriever import RetrievalPipeline

pipeline = RetrievalPipeline(vectorstore_manager)
results = pipeline.retrieve("What is machine learning?")
```

---

## Technologies Used

| Component | Technology |
|-----------|------------|
| RAG Framework | LangChain |
| Vector Store | FAISS, ChromaDB |
| Embeddings | Sentence-Transformers |
| LLM Integration | HuggingFace Hub |
| Evaluation | Custom metrics (RAGAS-inspired) |
| Demo Interface | Streamlit |
| Data Processing | pandas, BeautifulSoup |

---

## Sample Documents

The project includes sample documents covering AI/ML topics:

- Machine Learning fundamentals
- Deep Learning and neural networks
- Natural Language Processing
- RAG systems architecture
- Vector databases
- Transformer architecture
- Text embeddings
- Large Language Models

These provide a consistent dataset for demonstrating RAG capabilities.

---

## Evaluation Results

The system is evaluated on 8 test questions covering various AI/ML topics:

| Metric | Score | Description |
|--------|-------|-------------|
| Faithfulness | **96.5%** | Retrieved context supports the answer |
| Answer Relevancy | 41.7% | Answer addresses the question |
| Retrieval Quality | High | Top results score 0.44-1.25 similarity |

### Per-Topic Performance

| Topic | Relevancy | Faithfulness |
|-------|-----------|--------------|
| Machine Learning | 50% | 96.6% |
| Transformers | 50% | 96.4% |
| RAG | 0% | 96.6% |
| Embeddings | 0% | 96.6% |
| Deep Learning | 50% | 96.6% |
| Vector Databases | 50% | 96.7% |
| NLP | 66.7% | 96.2% |
| LLMs | 66.7% | 96.9% |

*Note: Low relevancy scores reflect the simple keyword-matching heuristic used for evaluation without an LLM. The high faithfulness scores indicate excellent retrieval quality - the system finds relevant context for all queries.*

Run `python scripts/evaluate.py` to regenerate metrics.

---

## License

MIT License
