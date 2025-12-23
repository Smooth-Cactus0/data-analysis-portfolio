"""
RAG Chain Module
================

Complete RAG chain assembly with LLM integration.
Combines retrieval with language model generation.
"""

import os
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.llms import HuggingFaceHub

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Available LLM providers."""

    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    LOCAL = "local"


@dataclass
class RAGConfig:
    """Configuration for RAG chain."""

    # LLM settings
    llm_provider: LLMProvider = LLMProvider.HUGGINGFACE
    model_name: str = "google/flan-t5-base"
    temperature: float = 0.1
    max_tokens: int = 512

    # Retrieval settings
    top_k: int = 4

    # Response settings
    include_sources: bool = True
    verbose: bool = False


# Prompt templates
QA_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, say "I don't have enough information to answer this question."
Don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

CHAT_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.
Be concise and accurate. If the context doesn't contain relevant information, acknowledge that.

Context:
{context}

User Question: {question}

Assistant Response:"""


class RAGChain:
    """
    Complete RAG chain for question answering.

    Combines:
        - Document retrieval from vector store
        - Context formatting
        - LLM-based answer generation
        - Source tracking

    Example:
        >>> rag = RAGChain(retrieval_pipeline)
        >>> answer = rag.query("What is machine learning?")
        >>> print(answer["answer"])
    """

    def __init__(
        self,
        retrieval_pipeline,
        config: Optional[RAGConfig] = None,
        llm=None
    ):
        """
        Initialize the RAG chain.

        Args:
            retrieval_pipeline: RetrievalPipeline instance
            config: Optional RAGConfig
            llm: Optional pre-configured LLM (overrides config)
        """
        self.config = config or RAGConfig()
        self.retrieval_pipeline = retrieval_pipeline

        if llm:
            self._llm = llm
        else:
            self._llm = self._create_llm()

        self._prompt = self._create_prompt()
        self._chain = self._build_chain()

    def _create_llm(self):
        """Create the LLM based on configuration."""
        if self.config.llm_provider == LLMProvider.HUGGINGFACE:
            # Using HuggingFace Hub (requires HF_TOKEN env var for some models)
            return HuggingFaceHub(
                repo_id=self.config.model_name,
                model_kwargs={
                    "temperature": self.config.temperature,
                    "max_length": self.config.max_tokens,
                }
            )
        elif self.config.llm_provider == LLMProvider.OPENAI:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")

    def _create_prompt(self) -> PromptTemplate:
        """Create the prompt template."""
        return PromptTemplate(
            template=QA_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

    def _format_docs(self, docs: List[Document]) -> str:
        """Format documents into context string."""
        return self.retrieval_pipeline.format_context(docs)

    def _build_chain(self):
        """Build the LangChain chain."""
        # Simple chain: retrieve -> format -> generate
        return (
            self._prompt
            | self._llm
            | StrOutputParser()
        )

    def query(
        self,
        question: str,
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a RAG query.

        Args:
            question: User question
            filter: Optional metadata filter for retrieval

        Returns:
            Dict with answer, sources, and metadata
        """
        logger.info(f"Processing query: '{question[:50]}...'")

        # Retrieve relevant documents
        docs = self.retrieval_pipeline.retrieve(question, filter=filter)

        if not docs:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "context": "",
                "question": question,
            }

        # Format context
        context = self._format_docs(docs)

        # Generate answer
        answer = self._chain.invoke({
            "context": context,
            "question": question
        })

        # Extract sources
        sources = []
        if self.config.include_sources:
            for doc in docs:
                source = doc.metadata.get(
                    "source",
                    doc.metadata.get("source_file", "Unknown")
                )
                if source not in sources:
                    sources.append(source)

        return {
            "answer": answer.strip(),
            "sources": sources,
            "context": context,
            "question": question,
            "num_docs_retrieved": len(docs),
        }

    def stream_query(
        self,
        question: str,
        filter: Optional[Dict[str, Any]] = None
    ):
        """
        Execute a streaming RAG query.

        Args:
            question: User question
            filter: Optional metadata filter

        Yields:
            Answer tokens as they are generated
        """
        # Retrieve documents
        docs = self.retrieval_pipeline.retrieve(question, filter=filter)

        if not docs:
            yield "I couldn't find any relevant information to answer your question."
            return

        # Format context
        context = self._format_docs(docs)

        # Stream answer
        for chunk in self._chain.stream({
            "context": context,
            "question": question
        }):
            yield chunk

    def batch_query(
        self,
        questions: List[str],
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple questions.

        Args:
            questions: List of questions
            filter: Optional metadata filter

        Returns:
            List of answer dicts
        """
        return [self.query(q, filter=filter) for q in questions]


class ConversationalRAG:
    """
    RAG chain with conversation history support.

    Maintains chat history and can use it for context.
    """

    def __init__(
        self,
        retrieval_pipeline,
        config: Optional[RAGConfig] = None,
        max_history: int = 10
    ):
        """
        Initialize conversational RAG.

        Args:
            retrieval_pipeline: RetrievalPipeline instance
            config: Optional RAGConfig
            max_history: Maximum conversation turns to remember
        """
        self.rag = RAGChain(retrieval_pipeline, config=config)
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []

    def chat(self, message: str) -> Dict[str, Any]:
        """
        Send a chat message and get a response.

        Args:
            message: User message

        Returns:
            Response dict with answer and metadata
        """
        # Get response
        response = self.rag.query(message)

        # Update history
        self.history.append({
            "role": "user",
            "content": message
        })
        self.history.append({
            "role": "assistant",
            "content": response["answer"]
        })

        # Trim history if needed
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]

        return response

    def clear_history(self):
        """Clear conversation history."""
        self.history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.history.copy()


class RAGChainBuilder:
    """Builder pattern for constructing RAG chains."""

    def __init__(self):
        self._retrieval_pipeline = None
        self._config = RAGConfig()
        self._llm = None
        self._custom_prompt = None

    def with_retrieval(self, retrieval_pipeline) -> "RAGChainBuilder":
        """Set the retrieval pipeline."""
        self._retrieval_pipeline = retrieval_pipeline
        return self

    def with_llm(self, llm) -> "RAGChainBuilder":
        """Set a custom LLM."""
        self._llm = llm
        return self

    def with_huggingface(
        self,
        model_name: str = "google/flan-t5-base",
        temperature: float = 0.1
    ) -> "RAGChainBuilder":
        """Configure HuggingFace LLM."""
        self._config.llm_provider = LLMProvider.HUGGINGFACE
        self._config.model_name = model_name
        self._config.temperature = temperature
        return self

    def with_openai(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.1
    ) -> "RAGChainBuilder":
        """Configure OpenAI LLM."""
        self._config.llm_provider = LLMProvider.OPENAI
        self._config.model_name = model_name
        self._config.temperature = temperature
        return self

    def with_top_k(self, k: int) -> "RAGChainBuilder":
        """Set number of documents to retrieve."""
        self._config.top_k = k
        return self

    def build(self) -> RAGChain:
        """Build the RAG chain."""
        if not self._retrieval_pipeline:
            raise ValueError("Retrieval pipeline is required")

        return RAGChain(
            retrieval_pipeline=self._retrieval_pipeline,
            config=self._config,
            llm=self._llm
        )


def create_rag_chain(
    retrieval_pipeline,
    model_name: str = "google/flan-t5-base",
    top_k: int = 4
) -> RAGChain:
    """
    Quick function to create a RAG chain.

    Args:
        retrieval_pipeline: RetrievalPipeline instance
        model_name: HuggingFace model name
        top_k: Number of documents to retrieve

    Returns:
        Configured RAGChain
    """
    config = RAGConfig(
        model_name=model_name,
        top_k=top_k,
    )
    return RAGChain(retrieval_pipeline, config=config)


if __name__ == "__main__":
    print("=== RAG Chain Module ===")
    print("This module requires a RetrievalPipeline to function.")
    print("See app.py or notebooks/rag_system.ipynb for full usage.")
