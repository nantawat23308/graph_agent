from pymilvus import Collection, MilvusException, connections, db, utility

from langchain_milvus import Milvus
from langchain_core.vectorstores import VectorStore
from src.langchain_milvus.utility import get_bedrock_embeddings
from src.langchain_milvus import constant

def get_vector_store(collection_name: str, uri: str = constant.URI, data_base:str = constant.DB_NAME) -> VectorStore:
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
            "token": "root:Milvus",
            "db_name": data_base

        },
        collection_name=collection_name,
        index_params={
            "index_type": "HNSW",  # Use HNSW for fast and accurate search
            "metric_type": "COSINE",  # Use COSINE for text search
        },
        primary_field="pk",
        auto_id=True,
        consistency_level="Strong",
        drop_old=False
    )
    return vector_store


def clear_collection(collection_name: str, db_name: str = constant.DB_NAME):
    """Clear Milvus vector store collection."""

    conn = connections.connect(host="127.0.0.1", port=19530)

    # Check if the database exists
    try:
        existing_databases = db.list_database()
        if db_name in existing_databases:

            # Use the database context
            db.using_database(db_name)
            print(f"Using '{db_name}'.")

            # Drop all collections in the database
            collections = utility.list_collections()
            if collection_name in collections:
                collection = Collection(name=collection_name)
                collection.drop()
                print(f"Collection '{collection_name}' has been dropped.")
    except MilvusException as e:
        print(f"An error occurred: {str(e)}")


if __name__ == '__main__':
    clear_collection(constant.COLLECTION_NAME)

