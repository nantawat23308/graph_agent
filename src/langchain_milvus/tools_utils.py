from langchain_core.tools import tool
from src.langchain_milvus.searching import search_vectors
from src.langchain_milvus import constant
from src.langchain_milvus.rag import rag_search_model
from langchain.chat_models import init_chat_model


@tool("VectorStoreSearch", return_direct=True)
def milvus_search(query: str) -> str:
    """Searches a Milvus vector store for relevant documents.

    Args:
        query: The search query string.
    Returns:
        Formatted string of search results.
    """

    results = search_vectors(
        collection_name=constant.COLLECTION_NAME,
        uri=constant.URI,
        query_text=query,
        top_k=5)

    if not results:
        return "No relevant documents found in the vector store."

    formatted_results = "Relevant documents found:\n\n"
    for i, doc in enumerate(results, 1):
        formatted_results += f"Document {i}:\n{doc.page_content}\n\n"

    return formatted_results

@tool("RAG_Search", return_direct=True)
def rag_milvus(query: str) -> str:
    """Performs RAG search using Milvus vector store and a language model.
    Args:
        query: The search query string.
    Returns:
        Formatted string of search results with model-generated responses.
    """
    model = init_chat_model("bedrock_converse:us.meta.llama4-maverick-17b-instruct-v1:0")
    formatted_results = rag_search_model(
        model=model,
        query=query,
    )
    return formatted_results
if __name__ == '__main__':
    # Example usage
    search_example = "exciting project with LangChain"
    print(rag_milvus.invoke(search_example))