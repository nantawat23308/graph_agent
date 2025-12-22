from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from src.langchain_milvus.searching import search_retrieve, search_retrieve_rerank
from src.langchain_milvus import constant
from src.langchain_milvus.prompt import SYSTEM_INSTRUCTION
from dotenv import load_dotenv

load_dotenv()


template = """
You are an expert assistant. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 

Context:
{context}

Question: 
{question}

Answer:
"""


template_1 = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""


template_strict = """
<system_instruction>
You are a High-Precision Extraction Engine. Your goal is to answer questions based strictly on the provided <context_documents>.

<constraints>
1. FAITHFULNESS: Answer ONLY using the provided documents. If the information is missing, output: "I do not have sufficient information."
2. NO CHAT: Do not say "Here is the answer" or "According to the text." Do not use any introductory or concluding sentences.
3. CITATIONS: Every factual claim must be followed by a source identifier in brackets, e.g., [Source: file_name.pdf].
4. FORMAT: Use clean Markdown. Start the response immediately with the factual data.
5. REASONING: Before providing the answer, analyze the documents in your internal thought process to ensure accuracy, but DO NOT include this analysis in the final output.
</constraints>

<context_documents>
{context}
</context_documents>

<output_format_example>
The maximum weight capacity is 500kg [Source: specs.pdf]. The safety factor is rated at 2.0 [Source: safety_manual.pdf].
SOURCES:
- specs.pdf
- safety_manual.pdf
</output_format_example>
</system_instruction>

<user_query>
{question}
</user_query>

<final_result_start_here>
"""


# 2. Helper function to format documents for the prompt
def format_docs(docs):
    print(len(docs))
    return "\n\n".join(doc.page_content for doc in docs)


def format_docs_with_sources(docs):
    formatted_chunks = []
    for doc in docs:
        # Pull the filename from metadata (adjust key name based on your Milvus setup)
        source_name = doc.metadata.get("source", "Unknown Source")

        # Create a block that clearly links text to source for the LLM
        chunk_text = f"CONTENT: {doc.page_content}\nMETADATA_SOURCE: {source_name}"
        formatted_chunks.append(chunk_text)

    return "\n\n---\n\n".join(formatted_chunks)


def rag_search_model(model: BaseChatModel, query: str) -> str:
    """Perform RAG search using Milvus vector store and LLM.

    Args:
        model: The language model to use for generating responses.
        query: The search query string.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_INSTRUCTION), ("human", "Context:\n{context}\n\nQuestion:\n{question}")]
    )
    retriever = search_retrieve(collection_name=constant.COLLECTION_NAME, uri=constant.URI, top_k=5)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | model | StrOutputParser()
    )
    response = rag_chain.invoke(query)
    return response


def rag_search_model_rerank(model: BaseChatModel, query: str) -> str:
    """Perform RAG search using Milvus vector store and LLM.

    Args:
        model: The language model to use for generating responses.
        query: The search query string.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_INSTRUCTION), ("human", "Context:\n{context}\n\nQuestion:\n{question}")]
    )
    # prompt = ChatPromptTemplate.from_template(template)
    retriever = search_retrieve_rerank(collection_name=constant.COLLECTION_NAME, uri=constant.URI, top_k=5)
    rag_chain = (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    response = rag_chain.invoke(query)
    return response


if __name__ == '__main__':
    example_query = "What document about?"
    llm = init_chat_model("bedrock_converse:us.meta.llama4-maverick-17b-instruct-v1:0")
    result = rag_search_model(llm, example_query)
    print(result)
    # result = llm("Hello, world!")  # Test to ensure model is working
    # print(result)
