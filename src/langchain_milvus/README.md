# LangChain Milvus Integration

This directory contains a comprehensive collection of Python modules for integrating the LangChain framework with Milvus vector database. It provides production-ready functionalities for document processing, vector storage, advanced search capabilities, and Retrieval-Augmented Generation (RAG) with reranking support.

## Overview

The Milvus integration provides a complete pipeline for building RAG applications, from document ingestion to intelligent search and answer generation. It uses AWS Bedrock embeddings (Amazon Titan v2) and supports multiple reranking models for improved search relevance.

## Key Features

- **Document Processing**: 
  - PDF and document parsing using Docling
  - Text file loading with UTF-8 support
  - Markdown processing with header-based splitting
  
- **Intelligent Chunking**: 
  - Recursive character text splitting with configurable overlap
  - Document-aware chunking for PDFs via Docling
  - Metadata preservation during chunking
  
- **Vector Storage**: 
  - Milvus vector store with HNSW indexing for fast similarity search
  - COSINE similarity metric optimized for text embeddings
  - AWS Bedrock embeddings (Amazon Titan v2)
  - Strong consistency guarantees
  
- **Advanced Search Methods**: 
  - Standard similarity search with score thresholds
  - Maximal Marginal Relevance (MMR) for diversity
  - Metadata filtering for targeted searches
  - Reranking with CrossEncoder models (HuggingFace)
  - Custom HNSW parameters for speed/accuracy trade-offs
  
- **RAG Implementation**: 
  - Complete RAG pipeline with context formatting
  - Source attribution and citation support
  - Reranking integration for improved relevance
  - Strict fact-based answer generation
  
- **Tool Integration**: 
  - LangChain-compatible tools for agent systems
  - `milvus_search`: Direct vector store search
  - `rag_milvus`: Full RAG with language model
  - Ready for integration with LangGraph agents

## File Descriptions

### Core Modules

- **`constant.py`**: 
  - Defines system-wide constants for Milvus connection
  - Database configuration: URI (`http://localhost:19530`), database name, collection name
  - Reranker model configurations:
    - `RERANK_MODEL_BGE`: BAAI/bge-reranker-v2-m3
    - `RERANK_MODEL_LIGHT`: cross-encoder/ms-marco-MiniLM-L-6-v2 (default)
    - `RERANK_MODEL_MXBAI`: mixedbread-ai/mxbai-rerank-large-v2
    - `RERANK_MODEL_MXBAI_BASE`: mxbai-rerank-base-v2

- **`db.py`**: 
  - Vector store factory with `get_vector_store()` function
  - Configures HNSW indexing for optimal performance
  - COSINE similarity metric for text embeddings
  - Strong consistency level for reliable reads
  - Collection management: `clear_collection()`
  - Supports custom index types: FLAT, IVF_FLAT, IVF_SQ8, HNSW, ANNOY
  - Metric types: L2 (Euclidean), IP (Inner Product), COSINE

- **`utility.py`**: 
  - AWS Bedrock embeddings initialization
  - Uses Amazon Titan Embed Text v2 model
  - Region: us-west-2
  - Returns Embeddings interface for LangChain compatibility

### Document Processing

- **`chunking.py`**: 
  - `chunk_text_file()`: Splits text files using RecursiveCharacterTextSplitter
    - Default chunk size: 1000 characters
    - Default overlap: 200 characters
  - `chunk_docling()`: Advanced PDF/document parsing with Docling
    - Exports to Markdown format for better structure preservation
    - Supports complex document layouts
  - Both functions return list of chunked text with metadata

- **`ingest_data.py`**: 
  - `ingest_from_texts()`: Create new vector store from text list
  - `ingest_from_documents()`: Create new vector store from Document objects
  - `ingest_texts()`: Add texts to existing vector store
  - `ingest_documents()`: Add Document objects to existing vector store
  - All functions handle embeddings and vector store operations automatically

- **`process.py`**: 
  - `process_and_ingest_file()`: Complete pipeline for single file
    - Chunks document using Docling
    - Creates Document objects with source metadata
    - Ingests into Milvus vector store
  - `process_and_ingest_directory()`: Batch processing for entire directories
    - Iterates through all files in directory
    - Maintains source attribution per file

### Search and Retrieval

- **`searching.py`**: 
  - **Basic Search**:
    - `search_vectors()`: Standard similarity search with top-K results
    - `search_similar_vectors()`: Search using pre-computed embeddings
    - `search_similar_vectors_with_scores()`: Returns (Document, score) tuples
  
  - **Advanced Search**:
    - `search_similar_vectors_with_metadata()`: Filtered search by metadata
    - `search_marginal_relevance()`: MMR for diversity (configurable λ)
    - `search_param_hnsw()`: Custom HNSW parameters (ef parameter)
  
  - **Retriever Configurations**:
    - `search_retrieve()`: MMR-based retriever for LangChain
    - `search_retrieve_rerank()`: Similarity search with CrossEncoder reranking
      - Fetches top 50 candidates
      - Reranks to top-K using cross-encoder model
      - Uses ContextualCompressionRetriever pattern

### RAG Pipeline

- **`prompt.py`**: 
  - System instruction for strict fact-based extraction
  - Rules: No introductory phrases, source citations required
  - Fallback response: "I don't know" when context insufficient

- **`rag.py`**: 
  - Multiple prompt templates:
    - `template`: Basic RAG prompt
    - `template_1`: Friendly response format
    - `template_strict`: High-precision extraction with citations
  
  - Helper functions:
    - `format_docs()`: Concatenates document content
    - `format_docs_with_sources()`: Adds source metadata to each chunk
  
  - RAG implementations:
    - `rag_search_model()`: Standard RAG with retriever
    - `rag_search_model_rerank()`: RAG with reranked results
    - Both use LCEL (LangChain Expression Language) chains

### Tool Integration

- **`tools_utils.py`**: 
  - **`milvus_search`** tool:
    - Decorated with `@tool` for LangChain compatibility
    - Performs vector similarity search
    - Returns formatted string of top-K documents
    - `return_direct=True` for immediate response
  
  - **`rag_milvus`** tool (2 variants):
    - Standard RAG search with language model
    - Reranked RAG search with CrossEncoder
    - Initializes LLM from configuration
    - Returns model-generated answers with context

### Database Management

- **`server.py`**: 
  - Database lifecycle management script
  - Checks for existing database
  - Creates new database if not exists
  - Drops all collections in database
  - Deletes database (cleanup utility)
  - Connection to Milvus: localhost:19530

### Testing

- **`test.py`**: 
  - Test scripts for validating Milvus functionality
  - Integration tests for search and retrieval

## Setup and Usage

### Prerequisites

- Python 3.12 or higher
- Milvus server (local or remote)
- AWS account with Bedrock access
- Dependencies listed in project root `pyproject.toml`

### Installation

1. **Install Dependencies** (from project root):
   ```bash
   # Using UV (recommended)
   uv sync
   
   # Or using pip
   pip install langchain langchain-aws langchain-milvus pymilvus langchain-docling sentence-transformers
   ```

2. **Start Milvus Server**:
   ```bash
   # Using Docker
   docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
   
   # Or using Milvus Lite (embedded)
   # Automatically started when using connection URI
   ```

3. **Set up Environment Variables**:
   Create a `.env` file:
   ```env
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_DEFAULT_REGION=us-west-2
   ```

4. **Initialize Database**:
   ```python
   from src.langchain_milvus.server import *
   # Creates database and collection automatically
   ```

### Basic Usage Examples

#### 1. Ingest Documents

```python
from src.langchain_milvus.process import process_and_ingest_file
from src.langchain_milvus.db import get_vector_store
from src.langchain_milvus import constant

# Initialize vector store
vector_store = get_vector_store(constant.COLLECTION_NAME, constant.URI)

# Process and ingest a single file
process_and_ingest_file(
    file_path="path/to/document.pdf",
    vector_store=vector_store,
    chunk_size=1000,
    chunk_overlap=200
)
```

#### 2. Search Documents

```python
from src.langchain_milvus.searching import (
    search_vectors,
    search_similar_vectors_with_scores,
    search_marginal_relevance
)
from src.langchain_milvus import constant

# Basic similarity search
results = search_vectors(
    collection_name=constant.COLLECTION_NAME,
    uri=constant.URI,
    query_text="What is LangChain?",
    top_k=5
)

# Search with scores
results_with_scores = search_similar_vectors_with_scores(
    collection_name=constant.COLLECTION_NAME,
    uri=constant.URI,
    query_text="Machine learning applications",
    top_k=5
)

# MMR search for diversity
mmr_results = search_marginal_relevance(
    collection_name=constant.COLLECTION_NAME,
    uri=constant.URI,
    query_text="AI technologies",
    top_k=5,
    fetch_k=20,
    lambda_mult=0.5  # Balance relevance and diversity
)
```

#### 3. RAG Pipeline

```python
from src.langchain_milvus.rag import rag_search_model, rag_search_model_rerank
from langchain.chat_models import init_chat_model

# Initialize model
model = init_chat_model("bedrock_converse:us.meta.llama4-maverick-17b-instruct-v1:0")

# Standard RAG
answer = rag_search_model(
    model=model,
    query="Explain retrieval augmented generation"
)
print(answer)

# RAG with reranking (better relevance)
reranked_answer = rag_search_model_rerank(
    model=model,
    query="What are the benefits of RAG?"
)
print(reranked_answer)
```

#### 4. Using as LangChain Tools

```python
from src.langchain_milvus.tools_utils import milvus_search, rag_milvus

# Use in agent
tools = [milvus_search, rag_milvus]

# Direct invocation
search_result = milvus_search.invoke("LangChain tutorials")
rag_result = rag_milvus.invoke("How to build agents?")
```

### Advanced Configuration

#### Custom Reranking Model

```python
from src.langchain_milvus import constant

# Change reranker model in constant.py
constant.RERANK_MODEL_LIGHT = "cross-encoder/ms-marco-MiniLM-L-12-v2"
```

#### Custom HNSW Parameters

```python
from src.langchain_milvus.searching import search_param_hnsw

# Higher ef = more accurate but slower
results = search_param_hnsw(
    collection_name=constant.COLLECTION_NAME,
    uri=constant.URI,
    query_text="query",
    top_k=5,
    ef=128  # Default is 64
)
```

#### Metadata Filtering

```python
from src.langchain_milvus.searching import search_similar_vectors_with_metadata

# Filter by source
results = search_similar_vectors_with_metadata(
    collection_name=constant.COLLECTION_NAME,
    uri=constant.URI,
    query_text="AI research",
    metadata_filter={"source": "research_paper.pdf"},
    top_k=5
)
```

### Database Management

```python
from src.langchain_milvus.db import clear_collection
from src.langchain_milvus import constant

# Clear all data from collection
clear_collection(constant.COLLECTION_NAME, constant.DB_NAME)
```

## Architecture

### Vector Store Configuration

```
┌─────────────────────────────────────────────────────────────┐
│                    Document Processing                       │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐    │
│  │   PDFs   │→ │ Docling  │→ │  Chunking (1000/200)  │    │
│  │  Files   │  │  Parser  │  │  RecursiveTextSplitter │    │
│  └──────────┘  └──────────┘  └────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Embedding & Storage                       │
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐    │
│  │   AWS Titan  │→ │   Vectors   │→ │  Milvus HNSW   │    │
│  │  Embed v2    │  │ (COSINE)    │  │    Index       │    │
│  └──────────────┘  └─────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Search & Retrieval                        │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │  Similarity │  │     MMR      │  │   Reranking    │    │
│  │   Search    │  │   (Diversity) │  │ (CrossEncoder) │    │
│  └─────────────┘  └──────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    RAG Generation                            │
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐    │
│  │   Context    │→ │     LLM     │→ │ Cited Answer   │    │
│  │  Formation   │  │  (Bedrock)  │  │  with Sources  │    │
│  └──────────────┘  └─────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Index Types Comparison

| Index Type | Speed | Accuracy | Memory | Use Case |
|------------|-------|----------|--------|----------|
| FLAT | Slow | 100% | Low | Small datasets (<10K) |
| IVF_FLAT | Medium | ~95% | Medium | Medium datasets |
| IVF_SQ8 | Fast | ~90% | Low | Large datasets, limited memory |
| HNSW | Very Fast | ~99% | High | Production (recommended) |
| ANNOY | Fast | ~95% | Medium | Large-scale approximate search |

### Similarity Metrics

| Metric | Range | Best For | Characteristics |
|--------|-------|----------|-----------------|
| COSINE | [-1, 1] | Text embeddings | Direction-based, length-invariant |
| L2 | [0, ∞] | Image search | Distance-based, considers magnitude |
| IP | (-∞, ∞) | Recommendations | Dot product, considers both direction & magnitude |

## Best Practices

### 1. Chunking Strategy

```python
# For technical documents
chunk_size = 1000
chunk_overlap = 200  # 20% overlap

# For narrative text
chunk_size = 1500
chunk_overlap = 300  # 20% overlap

# For code or structured data
chunk_size = 500
chunk_overlap = 50  # 10% overlap
```

### 2. Search Configuration

```python
# Fast search with good accuracy (production)
index_type = "HNSW"
ef = 64  # Standard search range

# High accuracy search (critical queries)
index_type = "HNSW"
ef = 256  # Wider search range

# Memory-constrained environment
index_type = "IVF_SQ8"
```

### 3. Reranking Strategy

```python
# Always use reranking for:
# - Question answering systems
# - High-stakes information retrieval
# - When you need top-K < 10

# Process:
# 1. Retrieve 50-100 candidates (fast vector search)
# 2. Rerank to top-K using cross-encoder (accurate but slow)
# 3. Use reranked results for RAG
```

### 4. Citation Management

```python
# Always preserve source metadata
metadata = {
    "source": file_path,
    "page": page_number,
    "section": section_title,
    "timestamp": datetime.now().isoformat()
}

# Format answers with citations
answer = f"{content} [Source: {source}]"
```

### 5. Error Handling

```python
try:
    results = search_vectors(...)
    if not results:
        return "No relevant information found."
except Exception as e:
    logger.error(f"Search failed: {e}")
    return "Unable to complete search."
```

## Performance Optimization

### Vector Store Settings

```python
# High throughput
vector_store = get_vector_store(
    collection_name=COLLECTION_NAME,
    uri=URI
)
# Uses HNSW with COSINE by default

# Tune for your use case:
# - Increase ef for better accuracy
# - Use IVF for faster insertion
# - Use COSINE for text (default)
```

### Batch Operations

```python
# Batch ingest for large datasets
from src.langchain_milvus.process import process_and_ingest_directory

process_and_ingest_directory(
    directory_path="./documents",
    vector_store=vector_store,
    chunk_size=1000,
    chunk_overlap=200
)
```

## Dependencies

Core dependencies (from project `pyproject.toml`):

- **langchain** (>=1.1.3): LangChain framework
- **langchain-aws** (>=1.1.0): AWS Bedrock integration
- **langchain-milvus** (>=0.3.2): Milvus vector store
- **pymilvus[milvus-lite]** (>=2.6.5): Milvus client library
- **langchain-docling**: Document processing
- **sentence-transformers** (>=5.2.0): Embedding models
- **boto3** (>=1.42.9): AWS SDK
- **python-dotenv**: Environment variables
- **torch**: PyTorch for ML models

## Troubleshooting

### Common Issues

1. **Connection Refused (port 19530)**
   ```bash
   # Check if Milvus is running
   docker ps | grep milvus
   # Start Milvus if not running
   docker start milvus-standalone
   ```

2. **AWS Credentials Error**
   ```bash
   # Verify credentials
   aws configure list
   # Check .env file has correct keys
   ```

3. **Out of Memory**
   ```python
   # Use IVF_SQ8 instead of HNSW
   # Reduce chunk_size
   # Reduce top_k in searches
   ```

4. **Slow Search Performance**
   ```python
   # Increase ef parameter (HNSW)
   # Use MMR instead of similarity search
   # Enable reranking only when needed
   ```

## Integration with Main Project

This module integrates with the main Deep Research Agent:

```python
from src.langchain_milvus.tools_utils import milvus_search, rag_milvus
from src.utility import tavily_search, think_tool

# Add to researcher tools
tools = [tavily_search, think_tool, milvus_search, rag_milvus]

config = {
    "configurable": {
        "researcher_tools": tools,
        # ... other config
    }
}
```

## Contributing

When contributing to this module:

1. Maintain consistent error handling patterns
2. Add comprehensive docstrings to all functions
3. Test with various document formats
4. Update this README with new features
5. Follow existing code structure and naming conventions

## License

[Specify your license here]

