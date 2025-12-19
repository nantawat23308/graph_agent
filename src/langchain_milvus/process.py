from torch.utils.hipify.hipify_python import meta_data

from src.langchain_milvus.chunking import chunk_docling
from src.langchain_milvus.ingest_data import ingest_documents
from src.langchain_milvus.db import get_vector_store
from src.langchain_milvus import constant
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from typing import List

def process_and_ingest_file(file_path: List[str], vector_store: VectorStore, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Process a text file by chunking and ingesting into Milvus vector store.

    Args:
        file_path (str): The path to the text file.
        vector_store (Milvus): The Milvus vector store instance.
        chunk_size (int, optional): The size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): The overlap between chunks. Defaults to 200.
    """
    # Chunk the text file
    document_collection = []
    for file in file_path:
        chunk_texts = chunk_docling(file, chunk_size, chunk_overlap)
        # Create Document objects from chunked texts
        documents = [Document(page_content=text, metadata={"source": file}) for text in chunk_texts]
        document_collection.extend(documents)

    # Ingest documents into the vector store
    ingest_documents(vector_store, document_collection)
    return vector_store

if __name__ == '__main__':


    vector_store = get_vector_store(constant.COLLECTION_NAME, constant.URI)
    file_path = r"D:\AI\portf\src\langchain_milvus\2512.16917v1.pdf"
    process_and_ingest_file([file_path], vector_store)