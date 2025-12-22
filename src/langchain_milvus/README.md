# LangChain Milvus Integration

This directory contains a collection of Python scripts for integrating the LangChain framework with a Milvus vector database. It provides functionalities for data ingestion, chunking, searching, and Retrieval-Augmented Generation (RAG).

## Key Features

- **Data Ingestion**: Scripts to ingest text data and documents into a Milvus collection.
- **Chunking**: Utilities to split large documents into smaller chunks for efficient processing.
- **Vector Search**: Various search methods to retrieve relevant documents from the Milvus database based on vector similarity.
- **RAG Implementation**: A complete RAG pipeline that uses Milvus as the vector store and a language model to generate answers.
- **Tooling**: Ready-to-use tools for searching the vector store and performing RAG searches.

## File Descriptions

- **`constant.py`**: Defines constants used across the project, such as the Milvus URI, database name, and collection name.
- **`db.py`**: Contains functions for creating and managing Milvus vector store instances.
- **`chunking.py`**: Provides functions for chunking text files and documents using different strategies.
- **`ingest_data.py`**: Includes functions for ingesting data (texts and documents) into the Milvus vector store.
- **`process.py`**: A script that processes and ingests a file into the Milvus database.
- **`searching.py`**: Implements various search functionalities, including similarity search, MMR, and metadata filtering.
- **`rag.py`**: Contains the implementation of the RAG pipeline, combining a retriever with a language model.
- **`tools_utils.py`**: Defines high-level tools for performing vector store searches and RAG searches.
- **`utility.py`**: Provides utility functions, such as initializing the Bedrock embeddings model.
- **`demo.py`**: A demonstration script showcasing how to use the different functionalities.
- **`server.py`**: A script for managing the Milvus database, including creating and deleting databases and collections.

## Setup and Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Set up Environment Variables**:
   Create a `.env` file in the root directory and add the necessary environment variables (e.g., AWS credentials for Bedrock).
3. **Use the Tools**:
   The `tools_utils.py` script provides tools that can be easily integrated into other applications.

## Dependencies

- `langchain`
- `langchain-aws`
- `langchain-milvus`
- `pymilvus`
- `python-dotenv`
- `torch`
- `langchain-docling`

