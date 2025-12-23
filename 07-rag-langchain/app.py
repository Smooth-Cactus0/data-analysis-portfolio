"""
RAG System Demo Application
===========================

Interactive Streamlit application for exploring the RAG system.
"""

import streamlit as st
from pathlib import Path
import time

# Page configuration
st.set_page_config(
    page_title="RAG System Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .source-tag {
        background-color: #e1e4e8;
        border-radius: 5px;
        padding: 0.2rem 0.5rem;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    .context-box {
        background-color: #f8f9fa;
        border-left: 3px solid #4a90d9;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "rag_initialized" not in st.session_state:
        st.session_state.rag_initialized = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "retrieval_pipeline" not in st.session_state:
        st.session_state.retrieval_pipeline = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None


def load_sample_documents():
    """Load sample documents for demo."""
    from langchain_core.documents import Document

    # Sample documents about AI/ML topics
    documents = [
        Document(
            page_content="""Machine learning is a subset of artificial intelligence (AI) that enables
            systems to automatically learn and improve from experience without being explicitly programmed.
            Machine learning focuses on developing computer programs that can access data and use it to
            learn for themselves. The process begins with observations or data, such as examples, direct
            experience, or instruction, in order to look for patterns in data and make better decisions
            in the future based on the examples provided.""",
            metadata={"source": "ml_intro.txt", "topic": "machine learning"}
        ),
        Document(
            page_content="""Deep learning is a subset of machine learning that uses artificial neural
            networks with multiple layers (hence "deep") to progressively extract higher-level features
            from raw input. For example, in image processing, lower layers may identify edges, while
            higher layers may identify human-recognizable concepts such as digits, letters, or faces.
            Deep learning has revolutionized computer vision, natural language processing, and speech
            recognition.""",
            metadata={"source": "deep_learning.txt", "topic": "deep learning"}
        ),
        Document(
            page_content="""Natural Language Processing (NLP) is a field of artificial intelligence that
            gives machines the ability to read, understand, and derive meaning from human languages. NLP
            combines computational linguistics with statistical, machine learning, and deep learning models.
            Applications include sentiment analysis, machine translation, chatbots, and text summarization.
            Modern NLP heavily relies on transformer architectures like BERT and GPT.""",
            metadata={"source": "nlp_overview.txt", "topic": "nlp"}
        ),
        Document(
            page_content="""Retrieval-Augmented Generation (RAG) is a technique that combines information
            retrieval with text generation. Instead of relying solely on the knowledge encoded in model
            parameters, RAG systems retrieve relevant documents from an external knowledge base and use
            them as context for generating responses. This approach reduces hallucinations and enables
            access to up-to-date information. Key components include document chunking, embedding generation,
            vector storage, and retrieval pipelines.""",
            metadata={"source": "rag_explained.txt", "topic": "rag"}
        ),
        Document(
            page_content="""Vector databases are specialized databases designed to store and query high-dimensional
            vectors efficiently. They enable similarity search at scale, which is crucial for RAG systems,
            recommendation engines, and semantic search applications. Popular vector databases include FAISS
            (Facebook AI Similarity Search), Pinecone, Weaviate, Chroma, and Milvus. These systems use
            approximate nearest neighbor (ANN) algorithms for fast retrieval.""",
            metadata={"source": "vector_db.txt", "topic": "vector databases"}
        ),
        Document(
            page_content="""Embeddings are dense vector representations of data (text, images, etc.) that
            capture semantic meaning. Text embeddings convert words, sentences, or documents into numerical
            vectors where similar meanings are closer together in vector space. Popular embedding models
            include Word2Vec, GloVe, and modern transformer-based models like BERT embeddings and
            sentence-transformers. The quality of embeddings significantly impacts retrieval performance.""",
            metadata={"source": "embeddings.txt", "topic": "embeddings"}
        ),
        Document(
            page_content="""Large Language Models (LLMs) are neural networks trained on massive text corpora
            that can generate human-like text, answer questions, and perform various language tasks.
            Examples include GPT-4, Claude, LLaMA, and Mistral. LLMs use the transformer architecture and
            are trained using self-supervised learning on billions of tokens. Fine-tuning and prompt
            engineering are common techniques to adapt LLMs for specific tasks.""",
            metadata={"source": "llm_basics.txt", "topic": "llm"}
        ),
        Document(
            page_content="""Chunking is the process of splitting documents into smaller pieces for processing
            and retrieval. Effective chunking strategies balance context preservation with retrieval precision.
            Common approaches include fixed-size chunking, recursive chunking (splitting by paragraphs,
            sentences, then words), and semantic chunking (grouping related content). Chunk size and overlap
            are key parameters that affect RAG system performance.""",
            metadata={"source": "chunking.txt", "topic": "chunking"}
        ),
    ]

    return documents


@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system components."""
    from src.document_loader import DocumentLoader
    from src.chunker import TextChunker, ChunkingStrategy
    from src.embeddings import EmbeddingGenerator, EmbeddingModel
    from src.vectorstore import VectorStoreManager, VectorStoreType
    from src.retriever import RetrievalPipeline, RetrieverConfig
    from src.chain import RAGChain, RAGConfig

    try:
        # Load sample documents
        documents = load_sample_documents()

        # Chunk documents
        chunker = TextChunker(
            chunk_size=500,
            chunk_overlap=50,
            strategy=ChunkingStrategy.RECURSIVE
        )
        chunks = chunker.chunk_documents(documents)

        # Create embeddings
        embedder = EmbeddingGenerator(model=EmbeddingModel.MINILM)
        embeddings = embedder.langchain_embeddings

        # Create vector store
        vectorstore_manager = VectorStoreManager(embeddings, store_type=VectorStoreType.FAISS)
        vectorstore_manager.create_from_documents(chunks)

        # Create retrieval pipeline
        retrieval_pipeline = RetrievalPipeline(vectorstore_manager)

        return {
            "success": True,
            "documents": documents,
            "chunks": chunks,
            "vectorstore_manager": vectorstore_manager,
            "retrieval_pipeline": retrieval_pipeline,
            "embedder": embedder,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def display_sidebar():
    """Display sidebar with system info and controls."""
    st.sidebar.markdown("## System Configuration")

    # Model selection
    st.sidebar.selectbox(
        "Embedding Model",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "BAAI/bge-small-en-v1.5"],
        index=0,
        disabled=True,
        help="Embedding model for document vectorization"
    )

    # Retrieval settings
    st.sidebar.markdown("### Retrieval Settings")
    top_k = st.sidebar.slider("Number of documents to retrieve", 1, 10, 4)
    search_type = st.sidebar.selectbox(
        "Search Strategy",
        ["Similarity", "MMR (Diverse)", "Threshold"],
        index=0
    )

    st.sidebar.markdown("---")

    # System status
    st.sidebar.markdown("### System Status")
    if st.session_state.rag_initialized:
        st.sidebar.success("RAG System: Active")
        if hasattr(st.session_state, "num_chunks"):
            st.sidebar.info(f"Documents indexed: {st.session_state.num_chunks}")
    else:
        st.sidebar.warning("RAG System: Initializing...")

    st.sidebar.markdown("---")

    # Info
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    This demo showcases a RAG (Retrieval-Augmented Generation) system built with LangChain.

    **Components:**
    - Document Loader
    - Text Chunker
    - Embedding Generator
    - Vector Store (FAISS)
    - Retrieval Pipeline

    **Author:** Alexy Louis
    """)

    return {"top_k": top_k, "search_type": search_type}


def display_chat_interface(rag_components, settings):
    """Display the main chat interface."""
    st.markdown('<p class="main-header">RAG System Demo</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Ask questions about AI, Machine Learning, NLP, and RAG systems</p>',
        unsafe_allow_html=True
    )

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                if "sources" in message and message["sources"]:
                    st.markdown("**Sources:** " + ", ".join(message["sources"]))

    # Chat input
    if question := st.chat_input("Ask a question about the documents..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating response..."):
                start_time = time.time()

                # Retrieve relevant documents
                results = rag_components["retrieval_pipeline"].retrieve_with_scores(
                    question,
                    k=settings["top_k"]
                )

                # Format context
                context = rag_components["retrieval_pipeline"].format_context(
                    [r["document"] for r in results]
                )

                # Generate answer (using simple response for demo without LLM)
                if results:
                    # Create a summary response based on retrieved content
                    answer = generate_simple_response(question, results)
                    sources = list(set(r["document"].metadata.get("source", "Unknown") for r in results))
                else:
                    answer = "I couldn't find relevant information to answer your question."
                    sources = []

                elapsed = time.time() - start_time

                # Display response
                st.write(answer)

                if sources:
                    st.markdown("**Sources:** " + ", ".join(sources))

                st.caption(f"Retrieved {len(results)} documents in {elapsed:.2f}s")

        # Add assistant message to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })


def generate_simple_response(question: str, results: list) -> str:
    """Generate a simple response without LLM (for demo purposes)."""
    if not results:
        return "I couldn't find relevant information in the knowledge base."

    # Get the most relevant document
    top_doc = results[0]["document"]
    score = results[0]["score"]

    # Create response based on retrieved content
    content = top_doc.page_content.strip()

    # Simple response formatting
    if score > 0.5:
        response = f"Based on the documents:\n\n{content[:500]}..."
    else:
        response = f"I found some potentially relevant information:\n\n{content[:300]}...\n\nNote: This may not directly answer your question."

    return response


def display_document_explorer(rag_components):
    """Display document exploration interface."""
    st.markdown("### Document Explorer")

    docs = load_sample_documents()

    # Document selector
    doc_names = [doc.metadata.get("source", f"Document {i}") for i, doc in enumerate(docs)]
    selected_doc = st.selectbox("Select a document", doc_names)

    # Find and display selected document
    for doc in docs:
        if doc.metadata.get("source") == selected_doc:
            st.markdown(f"**Topic:** {doc.metadata.get('topic', 'N/A')}")
            st.markdown("**Content:**")
            st.markdown(f'<div class="context-box">{doc.page_content}</div>', unsafe_allow_html=True)
            break


def display_search_demo(rag_components, settings):
    """Display semantic search demonstration."""
    st.markdown("### Semantic Search Demo")

    query = st.text_input("Enter a search query", placeholder="e.g., How do neural networks learn?")

    if query:
        with st.spinner("Searching..."):
            results = rag_components["retrieval_pipeline"].retrieve_with_scores(query, k=settings["top_k"])

        st.markdown(f"**Found {len(results)} relevant documents:**")

        for i, result in enumerate(results, 1):
            doc = result["document"]
            score = result["score"]

            with st.expander(f"Result {i}: {doc.metadata.get('source', 'Unknown')} (Score: {score:.4f})"):
                st.markdown(f"**Topic:** {doc.metadata.get('topic', 'N/A')}")
                st.markdown(f"**Content:**")
                st.write(doc.page_content)


def main():
    """Main application entry point."""
    init_session_state()

    # Display sidebar and get settings
    settings = display_sidebar()

    # Initialize RAG system
    if not st.session_state.rag_initialized:
        with st.spinner("Initializing RAG system..."):
            result = initialize_rag_system()

            if result["success"]:
                st.session_state.rag_initialized = True
                st.session_state.rag_components = result
                st.session_state.num_chunks = len(result["chunks"])
                st.rerun()
            else:
                st.error(f"Failed to initialize RAG system: {result['error']}")
                st.stop()

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["Chat", "Search Demo", "Document Explorer"])

    with tab1:
        display_chat_interface(st.session_state.rag_components, settings)

    with tab2:
        display_search_demo(st.session_state.rag_components, settings)

    with tab3:
        display_document_explorer(st.session_state.rag_components)

    # Footer
    st.markdown("---")
    st.markdown(
        "*RAG System Demo | Built with LangChain, FAISS, and Streamlit*",
        help="Project 7 of the Data Analysis Portfolio"
    )


if __name__ == "__main__":
    main()
