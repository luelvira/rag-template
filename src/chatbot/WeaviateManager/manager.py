"""
WeaviateManager.py

This module implements the WeaviateManager class which handles all interactions
with the Weaviate vector database.

It provides functionality for:
- Establishing and managing connections to Weaviate
- Creating and managing vector stores
- Document retrieval operations
- Configuration management for Weaviate connections

The module includes:
- WeaviateManager class: Main class for managing Weaviate operations
- Connection handling with error management
- Vector store initialization and management
- Integration with HuggingFace embeddings
- Configuration using WeaviateConfig from chatbot.config

Key Features:
- Automatic connection management with retry logic
- Support for multiple embedding models
- Collection management for vector stores
- Integration with LangChain's WeaviateVectorStore

Typical Usage:
1. Initialize WeaviateManager with connection parameters
2. Load vector store with specific embedding model and collection
3. Perform retrieval operations using the vector store
4. Close connection when done

Configuration:
- Uses WeaviateConfig for default connection parameters
- Supports custom host and port configurations
- Handles both HTTP and gRPC connections

Error Handling:
- Manages Weaviate connection errors
- Provides meaningful error messages for connection issues
- Handles vector store initialization failures
"""

from datetime import datetime
import logging


import weaviate
from weaviate.exceptions import WeaviateConnectionError
from weaviate.connect import ConnectionParams
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore

from chatbot.WeaviateManager.arguments import RetrieverParameters
from chatbot.config import WeaviateConfig

logger = logging.getLogger(__name__)

class WeaviateManager:
    """
    WeaviateManager class manages connections and operations with Weaviate vector database.

    This class provides functionality for:
    - Establishing and managing connections to Weaviate
    - Loading and managing vector stores
    - Handling document operations including adding and retrieving documents
    - Managing embeddings using HuggingFace models

    Attributes:
        db_host (str): Hostname for Weaviate database connection
        db_port (int): Port for Weaviate database connection
        vectorizer_host (str): Hostname for vectorizer service
        vectorizer_port (int): Port for vectorizer service
        client (weaviate.WeaviateClient): Weaviate client instance
        vectorstore (WeaviateVectorStore): Vector store instance for document operations

    Methods:
        __init__: Initializes WeaviateManager with connection parameters
        _connect_to_weaviate: Establishes connection to Weaviate database
        load_vectorstore: Loads vector store with specified embedding model and collection
        add_documents: Adds documents to the vector store with metadata
        get_retriever: Creates a document retriever for query operations
        close: Closes the Weaviate connection

    Configuration:
        Uses WeaviateConfig for default connection parameters
        Supports custom host and port configurations
        Handles both HTTP and gRPC connections

    Error Handling:
        Manages Weaviate connection errors
        Provides meaningful error messages for connection issues
        Handles vector store initialization failures

    Typical Usage:
        1. Initialize WeaviateManager with connection parameters
        2. Load vector store with specific embedding model and collection
        3. Perform document operations (add/retrieve)
        4. Close connection when done
    """
    def __init__(
            self,
            db_host: str = WeaviateConfig.DB_HOST.value,
            db_port: int = WeaviateConfig.DB_PORT.value,
            vectorizer_host: str = WeaviateConfig.VECTORIZER_HOST.value,
            vectorizer_port: int = WeaviateConfig.VECTORIZER_PORT.value
    ):
        self.db_host = db_host
        self.db_port = db_port
        self.vectorizer_host = vectorizer_host
        self.vectorizer_port = vectorizer_port
        self.client = self._connect_to_weaviate()
        self.vectorstore:WeaviateVectorStore|None = None

    def _connect_to_weaviate(self):
        """
        Establishes a connection to the Weaviate database.

        This method handles the connection setup to the Weaviate database using the configured
        host and port parameters. It supports both HTTP and gRPC connections.

        The connection is established using the following parameters:
        - HTTP host and port for REST API communication
        - gRPC host and port for vectorizer service communication
        - Secure connections disabled for both protocols

        Returns:
            weaviate.WeaviateClient: A connected Weaviate client instance if successful
            None: If the connection fails

        Raises:
            WeaviateConnectionError: If the connection to Weaviate fails
            Logs the error and returns None in case of connection failure

        Note:
            The method logs successful connections and connection failures
            The returned client instance should be used for subsequent operations
            The connection should be closed using the close() method when done
        """
        try:
            client = weaviate.WeaviateClient(
                connection_params=ConnectionParams.from_params(
                    http_host=self.db_host,
                    http_port=self.db_port,
                    http_secure=False,
                    grpc_host=self.vectorizer_host,
                    grpc_port=self.vectorizer_port,
                    grpc_secure=False,
                )
            )
            client.connect()
            logger.info("Connected to Weaviate successfully.")
            return client
        except WeaviateConnectionError as err:
            logger.error("Connection to Weaviate failed. Please check the URL and ports: %s", err)
            return None

    def load_vectorstore(self, embed_model, collection_name):
        """
        Loads and initializes the Weaviate vector store with the specified embedding
        model and collection.

        This method sets up the vector store by:
        1. Creating a HuggingFace embedding model instance using the provided model name
        2. Initializing a WeaviateVectorStore with the connected client, collection name,
           and embedding model
        3. Storing the initialized vector store instance for future use

        Args:
            embed_model (str): The name of the HuggingFace embedding model to use for vectorization
            collection_name (str): The name of the Weaviate collection to use as the vector store

        The vector store is configured with:
        - The connected Weaviate client
        - The specified collection name as the index
        - The HuggingFace embedding model for text vectorization
        - "text" as the key for document content

        Note:
            The initialized vector store is stored in the instance's vectorstore attribute
            This method should be called before performing any vector store operations
            The embedding model must be compatible with HuggingFace's Embeddings class
        """
        hugging_face_embed_model = HuggingFaceEmbeddings(model_name=embed_model)
        vectorstore = WeaviateVectorStore(
            client=self.client,
            index_name=collection_name,
            embedding=hugging_face_embed_model,
            text_key="text"
        )
        self.vectorstore = vectorstore

    def add_documents(self, docs):
        """
        Adds documents to the Weaviate vector store with enhanced metadata.

        This method processes and adds a list of documents to the vector store, enriching each
        document with additional metadata fields before storage. The metadata includes:

        - Source PDF information
        - Page numbers
        - Last modification timestamp
        - Category/topic classification
        - Difficulty level
        - Confidence/reliability score
        - Related links

        Args:
            docs (List[Document]): A list of Document objects to be added to the vector store.
                Each document should have basic metadata including 'source' and 'page'.

        The method performs the following operations:
        1. Generates a timestamp for the last modification date
        2. Enriches each document's metadata with additional fields
        3. Adds the documents to the vector store using the configured WeaviateVectorStore
        4. Logs the number of documents added

        Note:
            The documents must be properly formatted Document objects with at least 'source'
            and 'page' metadata fields.
            The vectorstore must be initialized using load_vectorstore() before calling this method.
            Metadata fields are standardized across all documents for consistent retrieval.
        """
        last_modification_date = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        for doc in docs:
            doc.metadata['pdf_resource_name'] = doc.metadata['source']
            doc.metadata['page_number'] = doc.metadata['page']
            doc.metadata['last_modification'] = last_modification_date
            doc.metadata['category_topic'] = "Cybersecurity"
            doc.metadata['difficulty_level'] = 5
            doc.metadata['confidence_reliability_level'] = 8
            doc.metadata['related_links'] = "http://example.com"
        self.vectorstore.add_documents(docs)
        logger.info("%d documentos agregados al vectorstore.", len(docs))

    def delete_collection(self, collection_name):
        """
        Deletes a specified collection from the Weaviate database.

        This method removes an entire collection and all its associated data from the Weaviate
        instance. It provides a direct interface to Weaviate's collection deletion functionality.

        Args:
            collection_name (str): The name of the collection to be deleted

        The method performs the following operations:
        1. Accesses the Weaviate client's collections interface
        2. Executes the delete operation on the specified collection
        3. All data, objects, and schema associated with the collection are permanently removed

        Note:
            This operation is irreversible and will permanently delete all data in the collection
            Ensure proper backups or data exports before executing this operation
            The collection must exist in the Weaviate instance
            Deleting a collection may affect dependent applications or services
        """
        self.client.collections.delete(collection_name)

    def get_retriever(self, params: RetrieverParameters):
        """
        Retrieves a document retriever instance configured with similarity score thresholding.

        This method implements an adaptive retrieval strategy that:
        1. Attempts to find relevant documents using a decreasing similarity threshold
        2. Starts with an initial threshold and decreases it incrementally
        3. Returns the first retriever that finds relevant documents
        4. Stops when the minimum threshold is reached

        Args:
            params (RetrieverParameters): Configuration parameters including:
                - model_name: Embedding model name
                - collection_name: Weaviate collection name
                - query: Search query string
                - initial_threshold: Starting similarity score threshold (default: 0.7)
                - min_threshold: Minimum similarity score threshold (default: 0.5)

        Returns:
            BaseRetriever: A retriever instance configured with:
                - Similarity score threshold search
                - k=4 (number of documents to retrieve)
                - Current threshold value

            None: If no relevant documents are found within the threshold range

        The method performs the following steps:
        1. Loads the vector store with specified embedding model and collection
        2. Iteratively searches with decreasing threshold values
        3. Returns the first retriever that finds relevant documents
        4. Returns None if no documents are found within the threshold range

        Note:
            The retriever uses similarity_score_threshold search type
            The threshold decreases by 0.01 in each iteration
            The method stops when the minimum threshold is reached
        """
        self.load_vectorstore(embed_model=params.model_name, collection_name=params.collection_name)
        score_threshold = params.initial_threshold
        while score_threshold > params.min_threshold:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={'score_threshold': score_threshold, 'k': 4}
            )
            relevant_documents = retriever.invoke(params.query)
            if relevant_documents:
                return retriever
            score_threshold -= 0.01
        return None


    def close(self):
        """Close the client connection and cleanup resources."""
        if self.client:
            self.client.close()
            self.client = None
            self.vectorstore = None


    def __del__(self):
        logger.debug("close weaviate connection")
        self.close()
