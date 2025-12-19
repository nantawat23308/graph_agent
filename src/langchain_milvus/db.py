from langchain_milvus import Milvus
from src.langchain_milvus.utility import get_bedrock_embeddings

def get_vector_store(collection_name: str, uri: str) -> Milvus:
    """Create Milvus vector store instance.
    index_type: how fast the search
        options: FLAT, IVF_FLAT, IVF_SQ8, HNSW, ANNOY
        FLAT: brute-force search, accurate but slow
        IVF_FLAT: Inverted File with Flat quantization, faster but less accurate
        IVF_SQ8: Inverted File with Scalar Quantization, faster but less accurate
        HNSW: Hierarchical Navigable Small World, Extremely fast and accurate but uses more memory
        ANNOY: Approximate Nearest Neighbors Oh Yeah, fast but less accurate

    metric_type: how to measure the distance between vectors (Usually Depend on embedding model)
        options: L2, IP, COSINE
        L2: Euclidean Distance Measure the straight-line distance between two points in Euclidean space
            (Image search or vector length matters)
        IP: Inner Product Measure the similarity between two vectors based on the angle between them
            (Recommendation system or vector length matters)
        COSINE: Cosine Similarity (Most Commonly used) Measure the cosine of the angle between two vectors
            (Text search or vector direction matters)
            (Amazon Bedrock Embedding use COSINE as default)
    FLAT

    """
    vector_store = Milvus(
        embedding_function=get_bedrock_embeddings(),
        connection_args={
            "uri": uri,
        },
        collection_name=collection_name,
        index_params={
            "index_type": "HNSW", # Use HNSW for fast and accurate search
            "metric_type": "COSINE", # Use COSINE for text search
        },
    )
    return vector_store