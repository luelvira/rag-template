"""
files.py

This module handles file operations for the chatbot, including reading, processing, and uploading
files to the vector database. It provides functionality for:

- Reading and parsing PDF documents
- Splitting documents into chunks for processing
- Uploading processed documents to Weaviate vector store

Key components:
- read_file: Reads and extracts text from PDF files using PyMuPDFLoader
- split_documents: Splits documents into chunks using RecursiveCharacterTextSplitter
- upload_file: Handles the complete file upload process including validation, reading, splitting,
  and uploading to Weaviate

The module integrates with:
- WeaviateManager for vector store operations
- Configuration settings from chatbot.config
- Custom exceptions for error handling

Typical workflow:
1. File is received and validated
2. PDF content is extracted using PyMuPDFLoader
3. Content is split into chunks using RecursiveCharacterTextSplitter
4. Chunks are uploaded to Weaviate vector store using WeaviateManager

Exceptions:
- UploadFileException: Raised during file upload/processing errors
- InvalidFileExtensionError: Raised for non-PDF files
- FileNotFoundError: Raised when file cannot be found
"""

import os
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from chatbot.config import DEFAULT_CONFIGURATION
from chatbot.Exceptions import UploadFileException, InvalidFileExtensionError
from chatbot.WeaviateManager import WeaviateManager
from chatbot.config import WeaviateConfig as WEAVIATE_CONFIG, configurations_availables, DEFAULT_CONFIGURATION

def read_file(file_path):
    """
    Reads and extracts text content from a PDF file.

    This method uses PyMuPDFLoader to load and extract text from a PDF document. It handles
    the file reading process and returns the extracted content as a list of Document objects.

    Args:
        file_path (str): The path to the PDF file to be read

    Returns:
        List[Document]: A list of Document objects containing the extracted text content

    Raises:
        FileNotFoundError: If the specified file does not exist
        InvalidFileExtensionError: If the file is not a PDF document
        Exception: For any other errors during the file reading process
    """
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents


def split_documents(documents, chunk_size=1001, chunk_overlap=200) -> List[Document]:
    """
    Splits documents into chunks of specified size with overlap.

    This method uses RecursiveCharacterTextSplitter to divide documents into smaller chunks
    with a specified size and overlap. This is useful for processing large documents in
    manageable pieces, particularly for vector storage and retrieval.

    Args:
        documents (List[Document]): List of Document objects to be split
        chunk_size (int): Maximum size of each chunk in characters (default: 1001)
        chunk_overlap (int): Number of overlapping characters between chunks (default: 200)

    Returns:
        List[Document]: List of Document objects representing the split chunks

    Note:
        The chunk_size and chunk_overlap parameters should be chosen carefully based on
        the specific requirements of the text processing and the capabilities of the
        embedding model being used.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    return docs


def upload_file(file_path) -> bool:
    """
    Uploads and processes a file for the chatbot's knowledge base.

    This method handles the complete file upload process including:
    - Validating the file's existence and format
    - Reading and splitting the document content
    - Loading the content into the vector store

    Args:
        file_path (str): Path to the file to be uploaded

    Returns:
        bool: True if the file was successfully processed and uploaded

    Raises:
        FileNotFoundError: If the specified file does not exist
        InvalidFileExtensionError: If the file is not a PDF document
        UploadFileException: If there are errors during the vector store upload process
        Exception: For any other unexpected errors during the upload process

    Note:
        The method currently only supports PDF files. The file is processed using
        a specific configuration (index DEFAULT_CONFIGURATION) from the available configurations.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File named {file_path} does not exist")
    if not file_path.lower().endswith('.pdf'):
        raise InvalidFileExtensionError(f"File {file_path} is not a PDF document")

    document = split_documents(read_file(file_path))
    selected_config = configurations_availables[DEFAULT_CONFIGURATION]

    try:
        weaviate_manager = WeaviateManager(
            WEAVIATE_CONFIG.DB_HOST.value,
            WEAVIATE_CONFIG.DB_PORT.value,
            WEAVIATE_CONFIG.VECTORIZER_HOST.value,
            WEAVIATE_CONFIG.VECTORIZER_PORT.value
        )
        weaviate_manager.load_vectorstore(
            embed_model = selected_config.embed_model,
            collection_name= selected_config.collection_name
        )
        weaviate_manager.add_documents(document)
    except Exception as e:
        raise UploadFileException(
            f"Error while processing the file {file_path}: {str(e)}"
        ) from e
    finally:
        weaviate_manager.close()

    return True
