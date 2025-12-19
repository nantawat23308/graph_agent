from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from src.langchain_milvus.searching import search_retrieve
from src.langchain_milvus import constant
from dotenv import load_dotenv
load_dotenv()


# 1. Define the Prompt
template = """
You are an expert assistant. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 

Context:
{context}

Question: 
{question}

Answer:
"""

# 2. Helper function to format documents for the prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def rag_search_model(model: BaseChatModel, query: str) -> str:
    """Perform RAG search using Milvus vector store and LLM.

    Args:
        model: The language model to use for generating responses.
        query: The search query string.
    """
    prompt = ChatPromptTemplate.from_template(template)
    retriever = search_retrieve(collection_name=constant.COLLECTION_NAME, uri=constant.URI, top_k=3)
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )
    response = rag_chain.invoke(query)
    return response

if __name__ == '__main__':
    # example_query = "What exciting projects can be built with LangChain?"
    llm = init_chat_model("bedrock_converse:us.meta.llama4-maverick-17b-instruct-v1:0")
    # result = rag_search_model(llm, example_query)
    # print(result)
    result = llm("Hello, world!")  # Test to ensure model is working
    print(result)