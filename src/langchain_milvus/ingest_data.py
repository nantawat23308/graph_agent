from langchain_milvus import Milvus
from langchain_aws import BedrockEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from typing import List
from src.langchain_milvus.utility import get_bedrock_embeddings
from langchain_core.vectorstores import VectorStore

def ingest_from_texts(texts: List[str], collection_name: str, uri: str):
    """Ingest texts into Milvus vector store."""
    vector_store = Milvus.from_texts(
        texts,
        get_bedrock_embeddings(),
        collection_name=collection_name,
        connection_args={"uri": uri},
        drop_old=True,
    )
    return vector_store


def ingest_from_documents(documents: List[Document], collection_name: str, uri: str):
    """Ingest documents into Milvus vector store."""
    vector_store = Milvus.from_documents(
        documents,
        get_bedrock_embeddings(),
        collection_name=collection_name,
        connection_args={"uri": uri},
        drop_old=True,
    )
    return vector_store


def ingest_documents(vector_store: VectorStore, documents: List[Document]):
    """Ingest documents into Milvus vector store."""
    vector_store.add_documents(documents)
    return vector_store


def ingest_texts(vector_store: VectorStore, texts: List[str]):
    """Ingest texts into Milvus vector store."""
    vector_store.add_texts(texts)
    return vector_store
