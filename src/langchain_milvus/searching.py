from faiss import HNSW

from src.langchain_milvus.db import get_vector_store
from src.langchain_milvus.utility import get_bedrock_embeddings
from langchain_core.documents import Document
from typing import List


def search_vectors(
    collection_name: str,
    uri: str,
    query_text: str,
    top_k: int = 5
) -> List[Document]:
    """Search for vectors in Milvus vector store.
    Standard similarity search using the query text to find K most similar documents.
    """
    vector_store = get_vector_store(collection_name, uri)
    results = vector_store.similarity_search(
        query_text,
        k=top_k
    )
    return results


def search_similar_vectors(
    collection_name: str,
    uri: str,
    query_text: str,
    top_k: int = 5
) -> List[Document]:
    """Search for similar vectors in Milvus vector store.
    Standard similarity search using the query text embedding to find K most similar documents.
    """
    vector_store = get_vector_store(collection_name, uri)
    query_embedding = get_bedrock_embeddings().embed_query(query_text)
    results = vector_store.similarity_search_by_vector(
        query_embedding,
        k=top_k
    )
    return results

def search_similar_vectors_with_metadata(
    collection_name: str,
    uri: str,
    query_text: str,
    metadata_filter: dict,
    top_k: int = 5
) -> List[Document]:
    """Search for similar vectors in Milvus vector store with metadata filtering.
    Standard similarity search using the query text embedding to find K most similar documents
    that match the metadata filter.
    """
    vector_store = get_vector_store(collection_name, uri)
    query_embedding = get_bedrock_embeddings().embed_query(query_text)
    results = vector_store.similarity_search_by_vector(
        query_embedding,
        k=top_k,
        filter=metadata_filter
    )
    return results


def search_similar_vectors_with_scores(
    collection_name: str,
    uri: str,
    query_text: str,
    top_k: int = 5
) -> List[tuple[Document, float]]:
    """Search for similar vectors in Milvus vector store with scores.
    Search using the query text to find K most similar documents along with their similarity scores.
    COSINE similarity scores range from -1 to 1, where 1 indicates identical vectors.
    L2 distance scores range from 0 to infinity, where 0 indicates identical vectors.
    """
    vector_store = get_vector_store(collection_name, uri)
    results = vector_store.similarity_search_with_score(
        query_text,
        k=top_k
    )
    return results


def search_marginal_relevance(
    collection_name: str,
    uri: str,
    query_text: str,
    top_k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5
) -> List[Document]:
    """Search for vectors in Milvus vector store using Maximal Marginal Relevance (MMR).
    MMR is a technique that balances relevance and diversity in the search results.
    It selects documents that are both relevant to the query and diverse from each other.
    """
    vector_store = get_vector_store(collection_name, uri)
    results = vector_store.max_marginal_relevance_search(
        query_text,
        k=top_k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult
    )
    return results

def search_param_hnsw(
    collection_name: str,
    uri: str,
    query_text: str,
    top_k: int = 5,
    ef: int = 64
) -> List[Document]:
    """Search for vectors in Milvus vector store using HNSW with custom ef parameter.
    """
    search_param = {
        "ef": ef # 'ef' is the search range. Higher = more accurate but slower.
    }
    vector_store = get_vector_store(collection_name, uri)
    results = vector_store.similarity_search(
        query_text,
        k=top_k,
        param=search_param
    )
    return results

def search_retrieve(
    collection_name: str,
    uri: str,
    top_k: int = 5
):
    """Search for vectors in Milvus vector store using hybrid retrieval.
    Combines vector similarity search with traditional keyword-based retrieval.
    """
    vector_store = get_vector_store(collection_name, uri)
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k}
    )
    return retriever