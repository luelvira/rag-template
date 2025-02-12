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
