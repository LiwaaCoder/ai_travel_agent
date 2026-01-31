"""
RAG Pipeline: Document ingestion and vector store management.

This module handles:
1. Loading travel knowledge documents from markdown files
2. Splitting documents into retrievable chunks
3. Creating and persisting a ChromaDB vector store
4. Providing a retriever for the LangGraph agent
"""

import os
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Constants
KNOWLEDGE_DIR = Path(__file__).parent.parent.parent / "data" / "knowledge"
VECTOR_STORE_DIR = Path(__file__).parent.parent.parent / "vector_db" / "chroma_store"
COLLECTION_NAME = "travel_knowledge"


def load_documents(knowledge_dir: Optional[Path] = None) -> list:
    """
    Load all markdown documents from the knowledge directory.
    
    Args:
        knowledge_dir: Optional path to knowledge directory. Defaults to data/knowledge.
    
    Returns:
        List of LangChain Document objects.
    """
    dir_path = knowledge_dir or KNOWLEDGE_DIR
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Knowledge directory not found: {dir_path}")
    
    loader = DirectoryLoader(
        str(dir_path),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    
    documents = loader.load()
    
    # Add source metadata for citation
    for doc in documents:
        doc.metadata["source_type"] = "travel_knowledge"
        doc.metadata["filename"] = Path(doc.metadata.get("source", "")).name
    
    return documents


def split_documents(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Split documents into smaller chunks for better retrieval.
    
    Args:
        documents: List of LangChain Document objects.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Overlap between chunks for context continuity.
    
    Returns:
        List of split Document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n---", "\n\n", "\n", " ", ""],
        length_function=len,
    )
    
    return splitter.split_documents(documents)


def create_vector_store(
    documents: list,
    persist_dir: Optional[Path] = None,
    force_recreate: bool = False,
) -> Chroma:
    """
    Create or load a ChromaDB vector store with travel knowledge.
    
    Args:
        documents: List of Document chunks to embed.
        persist_dir: Directory to persist the vector store.
        force_recreate: If True, recreate even if exists.
    
    Returns:
        ChromaDB vector store instance.
    """
    store_path = persist_dir or VECTOR_STORE_DIR
    store_path.parent.mkdir(parents=True, exist_ok=True)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Check if store exists and we don't need to recreate
    if store_path.exists() and not force_recreate:
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(store_path),
        )
    
    # Create new vector store
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(store_path),
    )
    
    return vector_store


def get_retriever(k: int = 5, score_threshold: float = 0.7):
    """
    Get a retriever for the travel knowledge base.
    
    Args:
        k: Number of documents to retrieve.
        score_threshold: Minimum similarity score for results.
    
    Returns:
        LangChain retriever configured for travel knowledge.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_STORE_DIR),
    )
    
    return vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold},
    )


def build_knowledge_base(force_recreate: bool = False) -> Chroma:
    """
    Full pipeline: load docs, split, embed, and persist.
    
    Args:
        force_recreate: If True, recreate vector store even if exists.
    
    Returns:
        ChromaDB vector store instance.
    """
    print("📚 Loading travel knowledge documents...")
    docs = load_documents()
    print(f"   Loaded {len(docs)} documents")
    
    print("✂️  Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"   Created {len(chunks)} chunks")
    
    print("🔮 Creating vector embeddings...")
    vector_store = create_vector_store(chunks, force_recreate=force_recreate)
    print(f"   Vector store ready at {VECTOR_STORE_DIR}")
    
    return vector_store


if __name__ == "__main__":
    # CLI utility to rebuild the knowledge base
    import sys
    
    force = "--force" in sys.argv
    build_knowledge_base(force_recreate=force)
    print("✅ Knowledge base built successfully!")
