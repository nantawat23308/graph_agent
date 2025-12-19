from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType



def chunk_text_file(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Chunk a text file into smaller pieces.

    Args:
        file_path (str): The path to the text file.
        chunk_size (int, optional): The size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): The overlap between chunks. Defaults to 200.

    Returns:
        list[str]: A list of text chunks.
    """
    # Load the text file
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Split the documents into chunks
    chunks = text_splitter.split_documents(documents)

    # Extract the text from each chunk
    chunk_texts = [chunk.page_content for chunk in chunks]

    return chunk_texts

def chunk_docling(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Chunk a Docling file into smaller pieces.

    Args:
        file_path (str): The path to the Docling file.
        chunk_size (int, optional): The size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): The overlap between chunks. Defaults to 200.

    Returns:
        list[str]: A list of text chunks.
    """
    # Load the Docling file
    loader = DoclingLoader(file_path, export_type=ExportType.MARKDOWN)
    documents = loader.load()

    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Split the documents into chunks
    chunks = text_splitter.split_documents(documents)

    # Extract the text from each chunk
    chunk_texts = [chunk.page_content for chunk in chunks]

    return chunk_texts

def header_splitter(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Chunk a file into smaller pieces using header-based splitting.

    Args:
        file_path (str): The path to the file.
        chunk_size (int, optional): The size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): The overlap between chunks. Defaults to 200.

    Returns:
        list[str]: A list of text chunks.
    """
    # Load the text file
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # Create a header-based text splitter
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    docs_split_by_header = []
    for doc in documents:
        split_docs = markdown_splitter.split_text(doc.page_content)
        docs_split_by_header.extend(split_docs)

    # C. Apply a Recursive Splitter for very long sections
    # This handles cases where one section is 5,000 words long.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    final_chunks = text_splitter.split_documents(docs_split_by_header)

    return final_chunks


def manual_chunking(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        # Define the end of the chunk
        end = start + chunk_size

        # Grab the chunk
        chunk = text[start:end]

        # Add to list
        chunks.append(chunk)

        # Move the start pointer forward (size minus overlap)
        start += (chunk_size - overlap)

    return chunks


def chunk_by_paragraphs(text, max_chars=1000):
    # Split the text into individual paragraphs
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # If adding this paragraph exceeds the limit...
        if len(current_chunk) + len(para) > max_chars and current_chunk:
            # Save the current chunk and start a new one
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            # Otherwise, add paragraph to the current chunk
            current_chunk += "\n\n" + para

    # Add the last remaining piece
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks