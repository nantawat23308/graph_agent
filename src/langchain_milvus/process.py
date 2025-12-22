import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from src.langchain_milvus import constant
from src.langchain_milvus.chunking import chunk_docling
from src.langchain_milvus.db import get_vector_store
from src.langchain_milvus.ingest_data import ingest_documents


def process_and_ingest_file(file_path: str, vector_store: VectorStore, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Process a text file by chunking and ingesting into Milvus vector store.

    Args:
        file_path (str): The path to the text file.
        vector_store (Milvus): The Milvus vector store instance.
        chunk_size (int, optional): The size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): The overlap between chunks. Defaults to 200.
    """
    # Chunk the text file
    chunk_texts = chunk_docling(file_path, chunk_size, chunk_overlap)
    # Create Document objects from chunked texts
    documents = [Document(page_content=text, metadata={"source": file_path}) for text in chunk_texts]

    # Ingest documents into the vector store
    ingest_documents(vector_store, documents)
    return vector_store

def process_and_ingest_directory(directory_path: str, vector_store: VectorStore, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Process all text files in a directory by chunking and ingesting into Milvus vector store.

    Args:
        directory_path (str): The path to the directory containing text files.
        vector_store (Milvus): The Milvus vector store instance.
        chunk_size (int, optional): The size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): The overlap between chunks. Defaults to 200.
    """


    document_collection = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            chunk_texts = chunk_docling(file_path, chunk_size, chunk_overlap)
            # Create Document objects from chunked texts
            documents = [Document(page_content=text, metadata={"source": file_path}) for text in chunk_texts]
            document_collection.extend(documents)

    # Ingest documents into the vector store
    ingest_documents(vector_store, document_collection)
    return vector_store

if __name__ == '__main__':
    path_doc = Path().cwd().parent.parent / "doc"
    print(path_doc.as_posix())
    vector_store = get_vector_store(constant.COLLECTION_NAME, constant.URI)
    process_and_ingest_directory(path_doc.as_posix(), vector_store)
    print("Ingestion completed.")