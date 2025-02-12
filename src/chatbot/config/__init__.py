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


DEFAULT_CONFIGURATION = 3
OLLAMA_API_PORT = 11434
