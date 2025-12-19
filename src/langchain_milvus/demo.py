from langchain_aws import BedrockEmbeddings
from langchain_milvus import Milvus
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from uuid import uuid4
URI = "./milvus_example.db"




def get_bedrock_embeddings() -> Embeddings:
    """Get Bedrock embeddings instance."""
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name="us-west-2",
    )
    return embeddings


vector_store = Milvus(
        embedding_function=get_bedrock_embeddings(),
        connection_args={
            "uri": URI,
        },
        index_params={
            "index_type": "FLAT",
            "metric_type": "L2",

        }
    )
def ingest_from_documents():
    vector_store_saved = Milvus.from_documents(
        [Document(page_content="foo!")],
        get_bedrock_embeddings(),
        collection_name="langchain_example",
        connection_args={"uri": URI},
    )
    return vector_store_saved

def ingest_documents():
    """Ingest documents into Milvus vector store."""
    document_1 = Document(
        page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
        metadata={"source": "tweet"},
    )

    document_2 = Document(
        page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
        metadata={"source": "news"},
    )

    document_3 = Document(
        page_content="Building an exciting new project with LangChain - come check it out!",
        metadata={"source": "tweet"},
    )

    document_4 = Document(
        page_content="Robbers broke into the city bank and stole $1 million in cash.",
        metadata={"source": "news"},
    )

    document_5 = Document(
        page_content="Wow! That was an amazing movie. I can't wait to see it again.",
        metadata={"source": "tweet"},
    )

    document_6 = Document(
        page_content="Is the new iPhone worth the price? Read this review to find out.",
        metadata={"source": "website"},
    )

    document_7 = Document(
        page_content="The top 10 soccer players in the world right now.",
        metadata={"source": "website"},
    )

    document_8 = Document(
        page_content="LangGraph is the best framework for building stateful, agentic applications!",
        metadata={"source": "tweet"},
    )

    document_9 = Document(
        page_content="The stock market is down 500 points today due to fears of a recession.",
        metadata={"source": "news"},
    )

    document_10 = Document(
        page_content="I have a bad feeling I am going to get deleted :(",
        metadata={"source": "tweet"},
    )

    documents = [
        document_1,
        document_2,
        document_3,
        document_4,
        document_5,
        document_6,
        document_7,
        document_8,
        document_9,
        document_10,
    ]
    uuids = [str(uuid4()) for _ in range(len(documents))]

    vector_store.add_documents(documents=documents, ids=uuids)
    print(f"Ingested {len(documents)} documents into Milvus vector store.")

def delete_items_from_vector_store(id: str):
    """Delete all items from the Milvus vector store."""
    vector_store.delete(ids=[id])
    print("Deleted all items from Milvus vector store.")

def query_direct():
    """Query the Milvus vector store."""
    query_result = vector_store.similarity_search(
        "exciting project with LangChain",
        k=5,
    )
    for i, doc in enumerate(query_result):
        print(f"Result {i + 1}: {doc.page_content} (Source: {doc.metadata['source']})")
    return query_result

if __name__ == "__main__":
    ingest_documents()
    query_direct()
    # delete_items_from_vector_store()