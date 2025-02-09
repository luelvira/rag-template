from enum import Enum
from dataclasses import dataclass

class WeaviateConfig(Enum):
    """
    Enum with the default values used to configure Weaviate connection
    """
    DB_HOST = "localhost"
    DB_PORT = 8090
    VECTORIZER_HOST = "localhost"
    VECTORIZER_PORT = 50051

@dataclass
class BaseConfiguration:
    """
    BaseConfiguration Dataclass

    This dataclass serves as the foundation for core chatbot configuration parameters.
    It encapsulates essential settings required for the chatbot's operation.

    Attributes:
        collection_name (str): The name of the collection/database to be used for storing
                               and retrieving documents. This should match the collection
                               name in the vector database.

        embed_model (str):     The name or identifier of the embedding model to be used for
                               text vectorization. This should be compatible with the vector
                               database's requirements.

        llm_model (str):       The name or identifier of the large language model to be used
                               for generating responses. This should be a valid model name
                               supported by the chatbot's LLM interface.
    """
    collection_name: str
    embed_model:     str
    llm_model:       str
configurations_availables = (
    BaseConfiguration("collection_a", "sentence-transformers/all-MiniLM-L6-v2", "llama3.2"),
    BaseConfiguration("collection_b",  "distilbert-base-nli-stsb-mean-tokens", "llama3.2"),
    BaseConfiguration("collection_c", "sentence-transformers/paraphrase-MiniLM-L6-v2", "phi3.5"),
    BaseConfiguration("collection_d", "distilbert-base-nli-stsb-mean-tokens", "phi3.5")
)

DEFAULT_CONFIGURATION = 3
