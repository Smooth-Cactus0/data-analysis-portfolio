"""
Document Fetching Script
========================

Fetches sample documents for the RAG system demo.
Downloads AI/ML articles from Wikipedia for a realistic dataset.
"""

import os
import sys
from pathlib import Path
import logging
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Wikipedia articles to fetch (AI/ML focused)
WIKIPEDIA_ARTICLES = [
    "Machine_learning",
    "Deep_learning",
    "Natural_language_processing",
    "Artificial_neural_network",
    "Transformer_(machine_learning_model)",
    "Large_language_model",
    "Word_embedding",
    "Recurrent_neural_network",
    "Convolutional_neural_network",
    "Reinforcement_learning",
]

# Sample document templates (for offline use)
SAMPLE_DOCUMENTS = {
    "machine_learning.txt": """
Machine Learning: A Comprehensive Overview

Machine learning is a subset of artificial intelligence (AI) that focuses on building systems
that can learn from and make decisions based on data. Rather than being explicitly programmed
to perform a specific task, these systems use algorithms and statistical models to identify
patterns and make inferences from data.

Types of Machine Learning:

1. Supervised Learning: The algorithm learns from labeled training data, making predictions
   based on that data. Examples include classification and regression tasks. Common algorithms
   include linear regression, logistic regression, decision trees, and support vector machines.

2. Unsupervised Learning: The algorithm works with unlabeled data, trying to find hidden
   patterns or intrinsic structures. Examples include clustering (K-means, hierarchical) and
   dimensionality reduction (PCA, t-SNE).

3. Reinforcement Learning: The algorithm learns by interacting with an environment, receiving
   rewards or penalties for actions. It's used in robotics, game playing, and autonomous systems.

Key Concepts:
- Training data: The dataset used to train the model
- Features: Input variables used for predictions
- Labels: Target variables in supervised learning
- Model: The mathematical representation learned from data
- Overfitting: When a model performs well on training data but poorly on new data
- Underfitting: When a model is too simple to capture patterns in the data

Machine learning has applications in image recognition, speech processing, recommendation systems,
fraud detection, medical diagnosis, and many other fields.
""",

    "deep_learning.txt": """
Deep Learning: Neural Networks at Scale

Deep learning is a subset of machine learning based on artificial neural networks with multiple
layers. These networks are called "deep" because they have many hidden layers between input and
output, allowing them to learn increasingly abstract representations of data.

Architecture Components:

1. Input Layer: Receives the raw data
2. Hidden Layers: Process and transform data through weighted connections
3. Output Layer: Produces the final prediction or classification
4. Activation Functions: Non-linear functions (ReLU, sigmoid, tanh) that enable learning complex patterns

Types of Deep Neural Networks:

- Feedforward Neural Networks (FNN): Basic architecture where information flows in one direction
- Convolutional Neural Networks (CNN): Specialized for image and spatial data processing
- Recurrent Neural Networks (RNN): Handle sequential data like time series and text
- Transformers: Use attention mechanisms for parallel processing of sequences
- Generative Adversarial Networks (GAN): Two networks competing to generate realistic data

Training Deep Networks:
- Backpropagation: Algorithm for computing gradients
- Optimization: Gradient descent variants (SGD, Adam, RMSprop)
- Regularization: Dropout, batch normalization, weight decay
- Data augmentation: Artificially expanding training data

Deep learning has achieved state-of-the-art results in computer vision, natural language
processing, speech recognition, and game playing.
""",

    "nlp_overview.txt": """
Natural Language Processing: Understanding Human Language

Natural Language Processing (NLP) is a field at the intersection of computer science,
artificial intelligence, and linguistics. Its goal is to enable computers to understand,
interpret, and generate human language in useful ways.

Core NLP Tasks:

1. Text Classification: Categorizing text into predefined categories
   - Sentiment analysis (positive/negative/neutral)
   - Topic classification
   - Spam detection

2. Named Entity Recognition (NER): Identifying entities in text
   - People, organizations, locations
   - Dates, quantities, monetary values

3. Machine Translation: Converting text between languages
   - Neural machine translation (NMT)
   - Sequence-to-sequence models

4. Question Answering: Extracting answers from text
   - Reading comprehension
   - Knowledge-based QA

5. Text Generation: Creating coherent text
   - Language modeling
   - Summarization
   - Dialogue systems

Key Technologies:

- Word Embeddings: Word2Vec, GloVe, FastText
- Transformers: BERT, GPT, T5, RoBERTa
- Attention Mechanisms: Self-attention, cross-attention
- Pre-training: Learning representations from large corpora
- Fine-tuning: Adapting pre-trained models for specific tasks

NLP applications include chatbots, virtual assistants, content recommendation,
automated customer service, and information extraction.
""",

    "rag_systems.txt": """
Retrieval-Augmented Generation: Enhancing LLMs with External Knowledge

Retrieval-Augmented Generation (RAG) is a technique that combines the generative capabilities
of large language models with information retrieval from external knowledge bases. This approach
addresses key limitations of pure LLMs, such as hallucinations and outdated knowledge.

RAG Pipeline Components:

1. Document Ingestion:
   - Load documents from various sources (PDF, web, databases)
   - Parse and clean text content
   - Extract metadata

2. Chunking:
   - Split documents into smaller segments
   - Balance context preservation with retrieval precision
   - Strategies: fixed-size, recursive, semantic

3. Embedding Generation:
   - Convert text chunks to vector representations
   - Models: sentence-transformers, OpenAI embeddings
   - Capture semantic meaning in vector space

4. Vector Storage:
   - Index embeddings for efficient retrieval
   - Databases: FAISS, Chroma, Pinecone, Weaviate
   - Enable similarity search at scale

5. Retrieval:
   - Find relevant documents for a query
   - Methods: similarity search, MMR, hybrid
   - Reranking for improved precision

6. Generation:
   - Provide retrieved context to LLM
   - Generate grounded responses
   - Include citations/sources

Benefits of RAG:
- Reduced hallucinations through grounding
- Access to current information
- Domain-specific knowledge
- Transparency with source citations

RAG is used in enterprise search, customer support, research assistants,
and knowledge management systems.
""",

    "vector_databases.txt": """
Vector Databases: Powering Semantic Search

Vector databases are specialized database systems designed to store, manage, and query
high-dimensional vectors efficiently. They are essential for applications requiring
similarity search, including RAG systems, recommendation engines, and image search.

Key Concepts:

1. Vector Embeddings:
   - Dense numerical representations of data
   - Capture semantic meaning in vector space
   - Dimensionality typically ranges from 384 to 1536

2. Similarity Metrics:
   - Cosine Similarity: Measures angle between vectors
   - Euclidean Distance: Straight-line distance
   - Dot Product: Related to cosine for normalized vectors

3. Indexing Algorithms:
   - Flat Index: Exact search, O(n) complexity
   - IVF (Inverted File): Clustering-based approximate search
   - HNSW (Hierarchical Navigable Small World): Graph-based approximate search

Popular Vector Databases:

- FAISS: Facebook's library for efficient similarity search
- Chroma: Open-source, easy-to-use vector store
- Pinecone: Managed vector database service
- Weaviate: Open-source with GraphQL interface
- Milvus: Highly scalable open-source solution
- Qdrant: Rust-based with filtering support

Best Practices:
- Choose appropriate embedding model for your domain
- Consider index type based on dataset size
- Implement metadata filtering
- Monitor and optimize query performance
- Handle updates and deletions properly

Vector databases enable semantic search beyond keyword matching,
finding conceptually similar content even with different words.
""",

    "transformers.txt": """
Transformers: The Architecture Revolutionizing AI

Transformers are a neural network architecture introduced in the paper "Attention Is All You Need"
(2017). They have become the foundation for modern NLP and are increasingly used in computer vision
and other domains.

Core Mechanism - Self-Attention:

The transformer's key innovation is the self-attention mechanism, which allows the model to weigh
the importance of different parts of the input when processing each element.

Q, K, V (Query, Key, Value):
- Query: What am I looking for?
- Key: What do I contain?
- Value: What information do I provide?

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

Architecture Components:

1. Encoder:
   - Processes input sequence
   - Self-attention + feed-forward layers
   - Used in BERT-style models

2. Decoder:
   - Generates output sequence
   - Masked self-attention + cross-attention
   - Used in GPT-style models

3. Positional Encoding:
   - Adds sequence position information
   - Sine/cosine functions or learned embeddings

Notable Transformer Models:

- BERT: Bidirectional encoder for understanding
- GPT: Decoder-only for generation
- T5: Encoder-decoder for text-to-text
- ViT: Vision Transformer for images
- DALL-E: Multimodal generation

Transformers scale well with data and compute, enabling the development
of large language models with billions of parameters.
""",

    "embeddings_explained.txt": """
Text Embeddings: Representing Meaning as Numbers

Text embeddings are dense vector representations that capture the semantic meaning of text.
They transform words, sentences, or documents into numerical vectors where similar meanings
are positioned closer together in the vector space.

Evolution of Embeddings:

1. One-Hot Encoding (Traditional):
   - Sparse vectors with vocabulary size
   - No semantic information
   - High dimensionality

2. Word2Vec (2013):
   - Dense vectors (typically 300 dimensions)
   - Learned from context (Skip-gram, CBOW)
   - Captures semantic relationships

3. GloVe (2014):
   - Global Vectors for Word Representation
   - Combines global statistics with local context
   - Pre-trained on large corpora

4. Contextual Embeddings (2018+):
   - Different representations based on context
   - BERT, ELMo, GPT embeddings
   - Handle polysemy (words with multiple meanings)

5. Sentence Embeddings:
   - sentence-transformers library
   - Models like all-MiniLM-L6-v2, all-mpnet-base-v2
   - Optimized for semantic similarity

Properties of Good Embeddings:
- Similar items have high cosine similarity
- Analogies preserved (king - man + woman ≈ queen)
- Clustering of related concepts
- Transfer to downstream tasks

Applications:
- Semantic search
- Document similarity
- Clustering and classification
- RAG retrieval
- Recommendation systems

Choosing embedding models involves trade-offs between
speed, quality, and dimensionality.
""",

    "llm_basics.txt": """
Large Language Models: Foundation of Modern NLP

Large Language Models (LLMs) are neural networks trained on massive text corpora
that can generate human-like text and perform various language tasks. They represent
the current state of the art in natural language processing.

How LLMs Work:

1. Architecture:
   - Transformer-based (usually decoder-only)
   - Billions of parameters
   - Attention mechanisms for context

2. Training:
   - Pre-training: Self-supervised on web text
   - Next token prediction objective
   - Massive compute requirements

3. Capabilities:
   - Text generation
   - Question answering
   - Summarization
   - Translation
   - Code generation
   - Reasoning (to some extent)

Major LLM Families:

- GPT (OpenAI): GPT-3.5, GPT-4
- Claude (Anthropic): Claude 2, Claude 3
- LLaMA (Meta): Open-weights models
- Mistral: Efficient open models
- Gemini (Google): Multimodal capabilities

Using LLMs Effectively:

- Prompt Engineering: Crafting effective inputs
- Few-shot Learning: Providing examples in prompt
- Chain-of-Thought: Encouraging step-by-step reasoning
- RAG: Grounding with retrieved context
- Fine-tuning: Adapting for specific tasks

Limitations:
- Hallucinations (generating false information)
- Knowledge cutoff (outdated information)
- Context length limits
- Computational cost
- Bias from training data

LLMs are powerful tools when used appropriately,
especially when combined with retrieval systems.
"""
}


def fetch_wikipedia_article(title: str) -> dict:
    """
    Fetch a Wikipedia article via the API.

    Args:
        title: Wikipedia article title

    Returns:
        Dict with title and content
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id != "-1":
                return {
                    "title": page_data.get("title", title),
                    "content": page_data.get("extract", ""),
                }

    except Exception as e:
        logger.warning(f"Failed to fetch {title}: {e}")

    return None


def save_documents_to_files(output_dir: Path) -> int:
    """
    Save sample documents to files.

    Args:
        output_dir: Directory to save files

    Returns:
        Number of documents saved
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for filename, content in SAMPLE_DOCUMENTS.items():
        filepath = output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content.strip())
        logger.info(f"Saved: {filename}")
        count += 1

    return count


def fetch_and_save_wikipedia(output_dir: Path) -> int:
    """
    Fetch Wikipedia articles and save to files.

    Args:
        output_dir: Directory to save files

    Returns:
        Number of articles fetched
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for title in WIKIPEDIA_ARTICLES:
        article = fetch_wikipedia_article(title)
        if article and article["content"]:
            filename = f"wikipedia_{title.lower()}.txt"
            filepath = output_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# {article['title']}\n\n")
                f.write(article["content"])

            logger.info(f"Fetched: {title}")
            count += 1
        else:
            logger.warning(f"Skipped: {title}")

    return count


def main():
    """Main function to fetch and save documents."""
    # Determine output directories
    project_dir = Path(__file__).parent.parent
    sample_dir = project_dir / "data" / "sample_docs"
    documents_dir = project_dir / "data" / "documents"

    logger.info("=== Document Fetching Script ===")

    # Save sample documents (always works, no network required)
    logger.info("\n--- Saving Sample Documents ---")
    sample_count = save_documents_to_files(sample_dir)
    logger.info(f"Saved {sample_count} sample documents to {sample_dir}")

    # Optionally fetch from Wikipedia
    logger.info("\n--- Fetching Wikipedia Articles ---")
    try:
        wiki_count = fetch_and_save_wikipedia(documents_dir)
        logger.info(f"Fetched {wiki_count} Wikipedia articles to {documents_dir}")
    except Exception as e:
        logger.warning(f"Wikipedia fetching failed: {e}")
        logger.info("Using sample documents only")

    logger.info("\n=== Done ===")
    logger.info(f"Sample documents: {sample_dir}")
    logger.info(f"Wikipedia documents: {documents_dir}")


if __name__ == "__main__":
    main()
