- [Conexión entre componentes](#org182ddac)
  - [Weaviate Manager](#orgdb815e2)
    - [Recuperación de información](#orgc46b48e)
  - [Integración del chat con el LLM](#org5040c6d)
  - [prompt service](#orgb0e7862)



<a id="org182ddac"></a>

# Conexión entre componentes

Al definir que es un RAG, mencionamos que está compueto por varios componentes externos. Por un lado, se encuentra la base de datos y por otro el LLM con el que trabajemos. Normalmente, estos modelos disponen de una API con la que nos debemos comunicar.

En este módulo crearemos dos componentes que permitan interactuar tanto con la API del LLM como con la base de datos.


<a id="orgdb815e2"></a>

## Weaviate Manager

El componente de `WeaviateManager` nos permite configurar el acceso a la base de datos de *Weaviate* y realizar operaciones básicas de inserción y eliminación.

El primer paso consiste en importar todas las contantes de configuración, junto con los módulos necesarios. Este contenido se añadirá al archivo `src/chatbot/WeaviateManager/manager.py`:

```python
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
```

Cada vez que un usuario interactúe con el chat, se iniciará una nueva conexión con la base de datos. Se define una clase clase encargada de manejar dichas conexiones, creando una nueva instancia cada vez que sea necesario interactuar con la base de datos.

El constructor de la clase acepta como parámetros los datos necesarios para establecer la configuración, almacenados en el archivo de configuración.

```python
class WeaviateManager:
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
```

También es necesario contar con un método que cierre la conexión entre la base de datos y nuestro sistema. Para ello, aprovechamos la funcionalidad de *Python* que permite implementar una función que se ejecuta automáticamente cuando el recolector de basura elimina la instancia.

```python
def close(self):
    """Close the client connection and cleanup resources."""
    if self.client:
        self.client.close()
        self.client = None
        self.vectorstore = None


def __del__(self):
    logger.debug("close weaviate connection")
    self.close()
```

El siguiente método que vamos a crear nos permite establecer la conexión con *Weaviate* empleando su cliente nativo.

```python
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
```

Una vez establecida la conexión, el siguiente paso es utilizar una estructura de datos compatible con el formato de vectores que maneja *Weaviate*. Para ello emplearemos la clase `WeaviateVectorStore`.

```python
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
```

Cada vez que el usuario suba un documento, este será almacenado en una tabla de la base de datos junto con un conjunto de metadados que facilitarán la generación de contexto.

```python
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
```

Finalmente, incorporamos un método que nos permita eliminar documentos almacenados en la base de datos.

```python
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
```


<a id="orgc46b48e"></a>

### Recuperación de información

Una vez que disponemos de las funciones básicas de escritura, es necesario definir un mecanismo que permita recuperar los documentos almacenados. Para ello, implementaremos un método que busque aquellos documentos más similares a la consulta del usuario. Por ese motivo, usamos un *threshold* que limite los documentos devueltos.

```python
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
```

El archivo `src/chatbot/WeaviateManager/manager.py` final queda estructurado de la siguiente manera

```python
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
```

1.  Consejo adicional

    Cuando un método o una función acepta "demasiados" parámetros, es conveniente agruparlos en una estructura propia. De esta forma, se simplifica tanto las llamadas a la función como el acceso a estos argumentos. En nuestro caso, vamos a crear un nuevo fichero con el siguiente contenido:
    
    ```python
    """
    arguments.py
    
    This module defines data structures and parameters used for interacting with Weaviate,
    a vector database. It provides a structured way to handle retrieval parameters for
    querying Weaviate collections.
    
    The module contains:
     - RetrieverParameters: A dataclass that encapsulates parameters needed for document
       retrieval from Weaviate
    
    Key components:
    - model_name: Specifies the embedding model to use for vector operations
    - collection_name: Identifies the Weaviate collection to query
    - query: Contains the search text for the retrieval operation
    - threshold parameters: Control the similarity matching behavior
    
    The RetrieverParameters class is used throughout the WeaviateManager module to
    standardize and manage retrieval operations, ensuring consistent parameter handling
    across different query types.
    
    Typical usage:
    1. Create a RetrieverParameters instance with required parameters
    2. Pass the instance to WeaviateManager methods for document retrieval
    3. Adjust threshold parameters as needed for different query scenarios
    """
    
    from dataclasses import dataclass
    
    
    @dataclass
    class RetrieverParameters:
        """
        A class to hold parameters for retrieving documents from Weaviate.
    
        Attributes:
            model_name (str): Name of the embedding model to use
            collection_name (str): Name of the Weaviate collection to query
            query (str): The search query text
            initial_threshold (float): Initial similarity threshold for matches (-1.0-1.0).
            min_threshold (float): Minimum similarity threshold to consider (-1.0-1.0).
        """
        model_name:        str
        collection_name:   str
        query:             str
        initial_threshold: float = 0.7
        min_threshold:     float = 0.6
    ```
    
    Dado que esta estructura es privada en el módulo `WeaviateManager` y no queremos que sea usada por el exterior, utilizaremos el fichero `src/chatbot/WeaviateManager/__init__.py` para exportar el contenido público del módulo.
    
    ```python
    """
    WeaviateManager Module
    
    This module provides the core functionality for managing interactions
    with the Weaviate vector database.
    
    The module includes:
    - WeaviateManager class: Main class for managing Weaviate operations
    - Connection handling with error management
    - Vector store initialization and management
    - Integration with HuggingFace embeddings
    - Configuration management using WeaviateConfig
    
    Key Features:
    - Automatic connection management with retry logic
    - Support for multiple embedding models
    - Collection management for vector stores
    - Integration with LangChain's WeaviateVectorStore
    - Document retrieval with similarity score thresholding
    - Document addition with enhanced metadata
    
    Typical Usage:
    1. Import WeaviateManager from this module
    2. Initialize with connection parameters
    3. Load vector store with specific embedding model and collection
    4. Perform document operations (add, retrieve, delete)
    5. Close connection when done
    
    Configuration:
    - Uses WeaviateConfig for default connection parameters
    - Supports custom host and port configurations
    - Handles both HTTP and gRPC connections
    
    Error Handling:
    - Manages Weaviate connection errors
    - Provides meaningful error messages for connection issues
    - Handles vector store initialization failures
    """
    from .manager import WeaviateManager
    ```


<a id="org5040c6d"></a>

## Integración del chat con el LLM

El LLM que vamos a usar es [ollama](https://ollama.com/), de *Meta*, ya que es Open Source y relativamente sencillo de usar. Lo primero que vamos a hacer es actualizar el *docker-compose* para incluir un contenedor con el modelo que queramos.

```yml
version: '3.4'
services:
  weaviate:
    command:
    - --host
    - 0.0.0.0
    - --port
    - '8090'  # Cambiado a 8090
    - --scheme
    - http
    image: cr.weaviate.io/semitechnologies/weaviate:1.27.1
    ports:
    - 8090:8090  # Mapeo de puertos actualizado
    - 50051:50051
    volumes:
    - weaviate_data:/var/lib/weaviate
    restart: on-failure:0
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      ENABLE_API_BASED_MODULES: 'true'
      CLUSTER_HOSTNAME: 'node1'

  ollama:
    image: ollama/ollama
    container_name: ollama-cont
    ports:
    - 11434:11434
    volumes:
    - ./ollama-data:/root/.ollama:Z


volumes:
    ollama:
    weaviate_data:
```

Una vez que el contenedor esté en ejecución, descargaremos el conjunto de modelos definidos en la configuración del programa, siendo estos llama3.2 y phi3.5.

Para descargar los modelos dentro del docker ejecutamos:

```
# Abrir una terminal dentro del contenedor
docker exec -it ollama-cont /bin/bash

# Dentro del contenedor
ollama pull phi3.5
ollama pull llama3.2
```

Ahora que el LLM está listo para su uso, implementaremos un servicio intermediario entre el chat y el LLM. Este servicio se encuentra en el fichero `src/chatbot/services/chat_services/chat_services.py` y proporciona una única función pública, `ask_memory`, encargada de interactuar con el LLM.

```python
def ask_memory(conversation: Conversation):
    """
    Processes a conversation with memory retrieval to generate a response.

    This method handles the complete conversation processing workflow including:
    - Setting up the chatbot configuration based on conversation settings
    - Creating the document retriever using Weaviate vector store
    - Building the RAG (Retrieval-Augmented Generation) chain
    - Generating a response using the LLM with retrieved context
    - Formatting the response with source document information

    Args:
        conversation (Conversation): The conversation object containing:
            - chat_history: List of previous messages in the conversation
            - conf: Configuration index for selecting chat settings

    Returns:
        dict: A dictionary containing:
            - "respuesta": The generated response text
            - "documentos_fuente": List of source documents with metadata including:
                * "documento": PDF resource name
                * "página": Page number
                * "dificultad": Difficulty level

        str: Error message if an exception occurs during processing

    Raises:
        Handles exceptions internally and returns error messages
        - Connection errors with Weaviate vector store
        - Errors during RAG chain invocation
        - Issues with LLM response generation
    """
    options = ChatbotOptions(
        endpoint=f"http://localhost:{os.getenv('OLLAMA_API_PORT', str(OLLAMA_API_PORT))}",
        configuration=configurations_availables[conversation.conf],
        propts=Prompts(),
    )
    query = conversation.chat_history[-1].content
    llm, retriever, weaviate_manager = _get_retriever(query, options)
    if retriever is None:
        return {
            "respuesta": "No dispongo de datos suficientes como para responder esa pregunta",
            "documentos_fuente": [],
        }

    rag_chain = _get_rag_chain(
        options.propts.default_prompt,
        llm,
        retriever
    )
    chat_history = [
        HumanMessage(content=x.content) if x.role == "user"
        else AIMessage(content=x.content)
        for x in conversation.chat_history
    ]
    try:
        response = rag_chain.invoke({"input":query, "chat_history": chat_history})
    except (ConnectionError, ValueError, RuntimeError) as e:
        weaviate_manager.close()
        return str(e)
    finally:
        weaviate_manager.close()
    return {
        "respuesta": response['answer'],
        "documentos_fuente": [
            {
                "documento":  doc.metadata['pdf_resource_name'],
                "página":     doc.metadata['page_number'],
                "dificultad": doc.metadata['difficulty_level']
            }
            for doc in response['context']
        ],
    }


def _get_retriever(query, options: ChatbotOptions):
```

Para complementar esta función, es necesario implementar dos funciones auxiliares. La primera se encarga de proporcionar el *retriever*, y la segunda de proporcionar un mecanismo para hacer las búsquedas de los vectores.

```python
    Retrieves the necessary components for RAG (Retrieval-Augmented Generation) chain.

    This method initializes and returns three key components needed for the RAG process:
    1. LLM (Language Model) instance configured with Ollama
    2. WeaviateManager instance for vector store operations
    3. Retriever instance for document retrieval

    Args:
        query (str): The user query to be used for document retrieval
        options (ChatbotOptions): Configuration options including:
            - endpoint: Ollama API endpoint
            - configuration: Model and collection configuration
            - propts: Prompt settings

    Returns:
        tuple: A tuple containing:
            - llm: Initialized OllamaLLM instance
            - retriever: Document retriever instance
            - weaviate_manager: WeaviateManager instance. This instance needs to be closed

    Note:
        The retriever is configured with parameters from the options, including:
        - Embedding model name
        - Collection name
        - Query string
    """
    llm = OllamaLLM(
        base_url=options.endpoint,
        model=options.configuration.llm_model,
        temperature=0.1
    )
    weaviate_manager = WeaviateManager()
    retriever = weaviate_manager.get_retriever(
        RetrieverParameters(
            model_name=options.configuration.embed_model,
            collection_name=options.configuration.collection_name,
            query=query
        )
    )
    return llm, retriever, weaviate_manager

def _get_rag_chain(prompt, llm, retriever):
    """
    Creates a RAG (Retrieval-Augmented Generation) chain for question answering.

    This method constructs a RAG chain by combining:
    1. A chat prompt template for formatting the question and context
    2. A document processing chain for handling retrieved documents
    3. A retrieval chain that integrates document retrieval with question answering

    Args:
        prompt (str): The prompt template to use for formatting the question and context
        llm (OllamaLLM): The language model instance to use for generating answers
        retriever (BaseRetriever): The document retriever instance for fetching relevant documents

    Returns:
        Runnable: A RAG chain that can be invoked with a question to generate an answer

    The RAG chain performs the following steps:
    1. Retrieves relevant documents based on the question
    2. Formats the question and documents using the provided prompt template
    3. Generates an answer using the language model

    Note:
        The returned RAG chain should be invoked with a dictionary containing:
        - "input": The user's question
        - "context": The retrieved documents (automatically handled by the chain)
    """
    chat_prompt = create_chat_prompt(prompt)
    question_answering_chain=create_stuff_documents_chain(llm, chat_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answering_chain)
    return rag_chain
```

El fichero junto con los imports necesarios, se estructura como se muestra a continuación

```python
"""
chat_service.py

This module implements the core chat service functionality for the chatbot application.
It handles conversation management, memory retrieval, and response generation using
a RAG (Retrieval-Augmented Generation) architecture.

The module provides:
- Conversation processing with memory retrieval
- Integration with Weaviate vector store for document retrieval
- Response generation using Ollama LLM
- Configuration management for different chat settings

Key components:
- ask_memory: Main function for processing conversations with memory retrieval
- _get_retriever: Helper function for setting up document retriever
- _get_rag_chain: Helper function for creating RAG chain
- Integration with WeaviateManager for vector store operations
- Configuration handling using BaseConfiguration and Prompts

The module uses:
- LangChain for chain creation and document processing
- Ollama for LLM integration
- Weaviate for vector store operations
- Custom types for conversation and chatbot options

Typical workflow:
1. Conversation object is received with chat history
2. Configuration is selected based on conversation settings
3. Document retriever is set up using Weaviate
4. RAG chain is created with appropriate prompt
5. Response is generated using LLM with retrieved context

Exceptions:
- Handles errors during RAG chain invocation
- Manages Weaviate connection cleanup
- Returns error messages for insufficient data cases
"""

import os

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaLLM
from chatbot.config import (
    configurations_availables,
    Prompts,
    OLLAMA_API_PORT,
)
from chatbot.WeaviateManager.manager import WeaviateManager
from chatbot.WeaviateManager.arguments import RetrieverParameters
from chatbot.services.chat_services.types import Conversation, ChatbotOptions
from chatbot.services.prompt_services import create_chat_prompt


def ask_memory(conversation: Conversation):
    """
    Processes a conversation with memory retrieval to generate a response.

    This method handles the complete conversation processing workflow including:
    - Setting up the chatbot configuration based on conversation settings
    - Creating the document retriever using Weaviate vector store
    - Building the RAG (Retrieval-Augmented Generation) chain
    - Generating a response using the LLM with retrieved context
    - Formatting the response with source document information

    Args:
        conversation (Conversation): The conversation object containing:
            - chat_history: List of previous messages in the conversation
            - conf: Configuration index for selecting chat settings

    Returns:
        dict: A dictionary containing:
            - "respuesta": The generated response text
            - "documentos_fuente": List of source documents with metadata including:
                * "documento": PDF resource name
                * "página": Page number
                * "dificultad": Difficulty level

        str: Error message if an exception occurs during processing

    Raises:
        Handles exceptions internally and returns error messages
        - Connection errors with Weaviate vector store
        - Errors during RAG chain invocation
        - Issues with LLM response generation
    """
    options = ChatbotOptions(
        endpoint=f"http://localhost:{os.getenv('OLLAMA_API_PORT', str(OLLAMA_API_PORT))}",
        configuration=configurations_availables[conversation.conf],
        propts=Prompts(),
    )
    query = conversation.chat_history[-1].content
    llm, retriever, weaviate_manager = _get_retriever(query, options)
    if retriever is None:
        return {
            "respuesta": "No dispongo de datos suficientes como para responder esa pregunta",
            "documentos_fuente": [],
        }

    rag_chain = _get_rag_chain(
        options.propts.default_prompt,
        llm,
        retriever
    )
    chat_history = [
        HumanMessage(content=x.content) if x.role == "user"
        else AIMessage(content=x.content)
        for x in conversation.chat_history
    ]
    try:
        response = rag_chain.invoke({"input":query, "chat_history": chat_history})
    except (ConnectionError, ValueError, RuntimeError) as e:
        weaviate_manager.close()
        return str(e)
    finally:
        weaviate_manager.close()
    return {
        "respuesta": response['answer'],
        "documentos_fuente": [
            {
                "documento":  doc.metadata['pdf_resource_name'],
                "página":     doc.metadata['page_number'],
                "dificultad": doc.metadata['difficulty_level']
            }
            for doc in response['context']
        ],
    }


def _get_retriever(query, options: ChatbotOptions):
    """
    Retrieves the necessary components for RAG (Retrieval-Augmented Generation) chain.

    This method initializes and returns three key components needed for the RAG process:
    1. LLM (Language Model) instance configured with Ollama
    2. WeaviateManager instance for vector store operations
    3. Retriever instance for document retrieval

    Args:
        query (str): The user query to be used for document retrieval
        options (ChatbotOptions): Configuration options including:
            - endpoint: Ollama API endpoint
            - configuration: Model and collection configuration
            - propts: Prompt settings

    Returns:
        tuple: A tuple containing:
            - llm: Initialized OllamaLLM instance
            - retriever: Document retriever instance
            - weaviate_manager: WeaviateManager instance. This instance needs to be closed

    Note:
        The retriever is configured with parameters from the options, including:
        - Embedding model name
        - Collection name
        - Query string
    """
    llm = OllamaLLM(
        base_url=options.endpoint,
        model=options.configuration.llm_model,
        temperature=0.1
    )
    weaviate_manager = WeaviateManager()
    retriever = weaviate_manager.get_retriever(
        RetrieverParameters(
            model_name=options.configuration.embed_model,
            collection_name=options.configuration.collection_name,
            query=query
        )
    )
    return llm, retriever, weaviate_manager

def _get_rag_chain(prompt, llm, retriever):
    """
    Creates a RAG (Retrieval-Augmented Generation) chain for question answering.

    This method constructs a RAG chain by combining:
    1. A chat prompt template for formatting the question and context
    2. A document processing chain for handling retrieved documents
    3. A retrieval chain that integrates document retrieval with question answering

    Args:
        prompt (str): The prompt template to use for formatting the question and context
        llm (OllamaLLM): The language model instance to use for generating answers
        retriever (BaseRetriever): The document retriever instance for fetching relevant documents

    Returns:
        Runnable: A RAG chain that can be invoked with a question to generate an answer

    The RAG chain performs the following steps:
    1. Retrieves relevant documents based on the question
    2. Formats the question and documents using the provided prompt template
    3. Generates an answer using the language model

    Note:
        The returned RAG chain should be invoked with a dictionary containing:
        - "input": The user's question
        - "context": The retrieved documents (automatically handled by the chain)
    """
    chat_prompt = create_chat_prompt(prompt)
    question_answering_chain=create_stuff_documents_chain(llm, chat_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answering_chain)
    return rag_chain
```

El siguiente paso consiste en emplear nuestro servicio, actualizando el fichero `src/chatbot/chat.py`, tanto el método `answer` como la lista de imports.

```diff
@@ -2,6 +2,7 @@
import logging
from gradio import ChatInterface, Error

from chatbot.services import files
+ from chatbot.services.chat_services import chat_service

from chatbot.Exceptions import UploadFileException
from chatbot.services.chat_services.types import Conversation, Message

logger = logging.getLogger(__name__)

@@ -61,14 +61,14 @@
    def answer(self, msg, history):
        if msg["files"]:
            for file in msg["files"]:
                uploaded = upload_file(file)
                if not isinstance(uploaded, bool):
                    return []
            return "Processed files"
        if len(self.history) != len(history):
            self.history = [Message.from_chat_message(msg) for msg in history]
        self.history.append(Message(text=msg["text"], own=True))
-        # simulate the response from the service
+       response = chat_service.ask_memory(Conversation(chat_history=self.history))
        response = {"respuesta": "Aún no está implementado"}
        self.history.append(Message(text=response["respuesta"], own=False))
        return response["respuesta"]
```

Finalmente, al igual que hicimos con `WeaviateManager`, vamos a definir una estructura de datos que nos permita organizar los argumentos que reciben las funciones y métodos cuando consideremos que son demasiados. Para ello, vamos a añadir una nueva `@dataclass` al fichero `src/chatbot/services/chat_services/types.py`

```python
@dataclass
class ChatbotOptions:
    """
    Dataclass used to store and agroup the option used to define a chat.

    Attributes:
        endpoint (str): The url where the LLM is accesible.
        configuration (BaseConfiguration): An instance of the BaseConfiguration used to
                                           setup the rest of parameters
        propts (Prompts): An instance of the Prompts class used to generate the responses.
    """
    endpoint: str
    configuration: BaseConfiguration
    propts: Prompts
```

Aunque aún no está definido, vamos a importar una clase `Prompts` del fichero config que implementaremos a continuación.

```python
from dataclasses import dataclass
from typing import List
from gradio import ChatMessage

from chatbot.config import BaseConfiguration, DEFAULT_CONFIGURATION, Prompts
```


<a id="orgb0e7862"></a>

## prompt service

Para ayudar a que el RAG nos proporcione mejores respuestas, podemos hacer *prompt engineering*. Weaviate proporciona una clase para preparar los prompts. Para ello, añadimos el fichero `src/chatbot/services/prompt_services.py`

```python
"""
prompt_services.py

This module provides prompt creation and management services for the chatbot application.
It handles the generation of prompt templates used in the RAG (Retrieval-Augmented Generation)
architecture and conversation management.

The module provides:

- Custom prompt template creation for question answering
- Chat prompt template generation with system messages and chat history
- Integration with LangChain's prompt template system
- Support for both simple and chat-based prompt structures

Key components:

- create_prompt: Creates a basic prompt template for question answering
- create_chat_prompt: Generates a chat prompt template with system message and history
- Integration with LangChain's PromptTemplate and ChatPromptTemplate
- Support for context-based question answering
"""

from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder


def create_prompt():
    """
    Creates a basic prompt template for question answering.

    This function generates a predefined prompt template designed for context-based
    question answering. The template instructs the chatbot to:

    - Use provided context information to answer the question
    - Admit when it doesn't know the answer
    - Avoid making up answers
    - Respond concisely in Spanish

    The template includes placeholders for:

    - Context: The relevant information to use for answering
    - Question: The user's query to be answered

    Returns:
        PromptTemplate: A LangChain PromptTemplate instance configured with:
            - The predefined Spanish question-answering template
            - Input variables for context and question
    """
    custom_prompt_template = """\
    Usa la siguiente información para responder a la pregunta del usuario.
    Si no sabes la respuesta, simplemente di que no lo sabes, no intentes inventar una respuesta.

    Contexto: {context}
    Pregunta: {question}

    Solo devuelve la respuesta útil a continuación y nada más y responde siempre en español.
    Respuesta útil:
    """
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])


def create_chat_prompt(system_prompt : str):
    """
    Creates a chat prompt template with system message and chat history support.

    This function generates a ChatPromptTemplate configured for conversational AI interactions.
    The template includes:
    - A system message that defines the chatbot's behavior and response guidelines
    - A placeholder for maintaining chat history context
    - A human message slot for the user's input

    The template is designed to:
    - Maintain conversational context through chat history
    - Allow dynamic system prompt configuration
    - Support multi-turn conversations
    - Integrate with LangChain's chat message handling

    Args:
        system_prompt (str): The system message defining the chatbot's behavior and guidelines.
                            This should be a complete instruction set for the chatbot's responses.

    Returns:
        ChatPromptTemplate: A LangChain ChatPromptTemplate instance configured with:
            - System message from the provided system_prompt
            - Chat history placeholder for maintaining conversation context
            - Human message slot for user input
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
```

En el fichero de configuración, vamos a añadir una clase que nos permita definir un conjunto de prompts y el puerto en el que escucha el contenedor de ollama

```python
@dataclass
class Prompts:
    """
    Prompts Dataclass

    This dataclass manages the different prompt templates used by the chatbot for
    generating responses. It provides multiple prompt variations with different
    levels of strictness in context usage.

    Attributes:
        PROMPT1: Basic prompt template allowing some flexibility in response generation
        PROMPT2: More strict prompt template requiring explicit context usage
        PROMPT3: Most strict prompt template prohibiting any external knowledge
        _default: Internal index tracking the currently selected default prompt

    Methods:
        default_prompt: Property that returns the currently selected default prompt
        default_prompt.setter: Allows changing the default prompt by index

    The prompts are designed to:
    - Guide the chatbot's response generation
    - Control the strictness of context usage
    - Ensure concise and contextually appropriate responses
    - Handle cases where context is insufficient

    Typical Usage:
    1. Access specific prompts directly (PROMPT1, PROMPT2, PROMPT3)
    2. Use default_prompt property for the currently selected default
    3. Change default prompt using the setter when needed
    """

    PROMPT1 = ("Eres un asistente para tareas de preguntas y respuestas. Usa los "
               "siguientes fragmentos de contexto recuperado para responder la pregunta. "
               "Si no sabes la respuesta, di que no lo sabes. Usa un máximo de tres "
               "frases y mantén la respuesta concisa.\n\n{context}")
    PROMPT2 = ("Eres un asistente de preguntas y respuestas. Usa únicamente los "
               "fragmentos de contexto proporcionado para responder la pregunta. Si el "
               "contexto no tiene suficiente información, responde explícitamente: "
               "'No lo sé'. No respondas basándote en conocimientos previos o sin "
               "contexto suficiente. Usa un máximo de tres frases y mantén la respuesta "
               "concisa.\n\n{context}")
    PROMPT3 = ("Eres un asistente para tareas de preguntas y respuestas. Usa "
               "exclusivamente los fragmentos de contexto a continuación para responder "
               "la pregunta. Si no sabes la respuesta o el contexto es insuficiente, "
               "di: 'No lo sé' y no intentes adivinar. No uses tus conocimientos "
               "previos. Usa un máximo de tres frases y mantén la respuesta "
               "concisa.\n\n{context}")
    _default = 2

    @property
    def default_prompt(self):
        """
        Returns the default prompt from the available prompts.

        This property provides access to the most commonly used prompt configuration
        for the chatbot. It returns PROMPT3 as the default prompt, which is designed
        for strict context-based question answering.

        Returns:
            str: The default prompt string (PROMPT3)
        """
        if self._default == 0:
            return self.PROMPT1
        if self._default == 1:
            return self.PROMPT2
        if self._default == 2:
            return self.PROMPT3
        raise ValueError("Default prompt index out of range. Must be between 0 and 2")

    @default_prompt.setter
    def default_prompt(self, value: int):
        """
        Sets the default prompt index.

        This setter allows changing the default prompt by specifying an index:
        - 0: Sets PROMPT1 as default
        - 1: Sets PROMPT2 as default
        - Any other value: Sets PROMPT3 as default

        Args:
            value (int): The index of the prompt to set as default
        """
        if not isinstance(value, int):
            raise ValueError("Default prompt index must be an integer")
        self._default = value

OLLAMA_API_PORT = 11434
```
