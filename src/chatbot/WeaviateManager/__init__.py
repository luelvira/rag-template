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
