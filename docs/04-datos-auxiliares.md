# Definición de tipos auxiliares

Para facilitar el desarrollo, es buena práctica definir tipos de datos propios que describan su propósito dentro del sistema. Estos tipos pueden ser simplemente *alias* si son equivalentes a uno ya existente, o bien, estructuras derivadas que combinen varios tipos básicos. En primer lugar, definiremos un *alias* que permita describir los mensajes.

Gradio ofrece una clase `ChatMessage`, que cumple con las especificaciones de la interfaz que propone openAI. Sin embargo, en nuestro caso, solo nos interesa emplear un subconjunto de sus funcionalidades, ya que, el emisor del mensaje siempre será `user` o `assistant`, es decir, lo podemos codificar como una propiedad binaria. Además, el contenido será exclusivamente texto. Dado que esta estructura no requiere lógica compleja, podemos usar el decorador `dataclass`.

Definimos estos nuevos tipos en un archivo específico, llamado `src/chatbot/services/chat_services/types.py`

```python
@dataclass
class Message:
    """
    Represents a single message in a chat conversation.

    This dataclass encapsulates the essential attributes of a chat message,
    including its content and ownership information. It provides properties
    for compatibility with chat interfaces and role-based message handling.

    Attributes:
        text (str): The content of the message, containing the actual text
                    that was sent or received in the conversation.
        own (bool): Flag indicating whether the message was sent by the
                    current user (True) or received from the chatbot (False).

    Properties:
        content: Returns the text content of the message, providing a
                 consistent interface for accessing message text.
        role: Returns the role of the message sender as either "user" or
              "assistant" based on the ownership flag.

    The class is designed to:
    - Provide a simple structure for chat message data
    - Support role-based message handling in chat interfaces
    - Maintain compatibility with Gradio's ChatMessage type
    - Enable easy access to message content and ownership information

    Typical Usage:
    1. Create Message instances for each chat exchange
    2. Use the content property to access message text
    3. Use the role property for role-based message handling
    4. Store messages in Conversation objects for chat history management
    """
    text: str
    own:  bool

    @property
    def content(self):
        """
        alias for the text property
        """
        return self.text

    @property
    def role(self):
        """
        alias for the ownership flag property
        """
        return "user" if self.own else "assistant"

    @classmethod
    def from_chat_message(cls, chat_message: ChatMessage):
        """
        Create a Message instance from a Gradio ChatMessage object.

        This class method converts a Gradio ChatMessage object into a Message instance
        by extracting the text content and ownership information from the chat message.

        Args:
            chat_message (ChatMessage): A Gradio ChatMessage object representing a chat message.

        Returns:
            A message instance with the text content and ownership flag from the chat message.
        """
        return cls(text=chat_message.content, own=chat_message.role == "user")


    def to_chat_message(self):
        """
        Convert a Message instance to a Gradio ChatMessage object.

        This method converts a Message instance into a Gradio ChatMessage object
        by creating a new ChatMessage with the text content and ownership information.

        Returns:
            ChatMessage: A Gradio ChatMessage object representing the message content and ownership.
        """
        return ChatMessage(content=self.text, role="user" if self.own else "assistant")
```

Aquí usamos un *boolean* para determinar si el mensaje proviente del usuario o del asistente, acompañado de un *string* con el contenido del mensaje. Por otra parte, empleamos el decorador `@property` para mantener la compatiblidad con la clase `ChatMessage`. También incluimos dos métodos para poder cambiar entre un tipo y el otro.

Siguiendo con la misma lógica, necesitamos una estructura de datos que represente una conversación. La definiremos como una lista de mensajes junto con la configuración que se mantendrá en la conversación. Es decir, vamos a crear un nuevo tipo de dato por composición. Cada configuración se compone de una colección de textos, un modelo embebido y un modelo LLM. Como las configuraciones son limitadas, podemos almacenar el conjunto de posibilidades en memoria y que nuestra nueva clase acceda a una configuración en específico de las que ya están definidas.

```python
@dataclass
class Conversation:
    """
    Conversation Dataclass

    This dataclass represents a complete conversation between a user and the chatbot.
    It maintains the chat history and configuration settings used for the conversation.

    Attributes:
        chat_history (List[ChatMessage]): A list of ChatMessage objects representing
                                          the sequence of messages in the conversation.
                                          Each message contains the text content and
                                          ownership information (user or assistant).

        conf (int): The configuration index specifying which chat settings to use.
                    Defaults to DEFAULT_CONFIGURATION. This index corresponds to
                    a specific configuration in the configurations_availables list.

    The class is designed to:
    - Maintain a complete history of chat exchanges
    - Track the configuration used for the conversation
    - Support multiple conversation configurations
    - Enable consistent chat behavior across sessions
    """
    chat_history: List[ChatMessage]
    conf: int = DEFAULT_CONFIGURATION
```

Finalmente, incluimos los elementos que necesitamos importar:

```python
from dataclasses import dataclass
from typing import List
from gradio import ChatMessage

from chatbot.config import BaseConfiguration, Prompts, DEFAULT_CONFIGURATION
```

El archivo final queda estructurado de la siguiente manera:

```python
from dataclasses import dataclass
from typing import List
from gradio import ChatMessage

from chatbot.config import BaseConfiguration, Prompts, DEFAULT_CONFIGURATION

@dataclass
class Message:
    """
    Represents a single message in a chat conversation.

    This dataclass encapsulates the essential attributes of a chat message,
    including its content and ownership information. It provides properties
    for compatibility with chat interfaces and role-based message handling.

    Attributes:
        text (str): The content of the message, containing the actual text
                    that was sent or received in the conversation.
        own (bool): Flag indicating whether the message was sent by the
                    current user (True) or received from the chatbot (False).

    Properties:
        content: Returns the text content of the message, providing a
                 consistent interface for accessing message text.
        role: Returns the role of the message sender as either "user" or
              "assistant" based on the ownership flag.

    The class is designed to:
    - Provide a simple structure for chat message data
    - Support role-based message handling in chat interfaces
    - Maintain compatibility with Gradio's ChatMessage type
    - Enable easy access to message content and ownership information

    Typical Usage:
    1. Create Message instances for each chat exchange
    2. Use the content property to access message text
    3. Use the role property for role-based message handling
    4. Store messages in Conversation objects for chat history management
    """
    text: str
    own:  bool

    @property
    def content(self):
        """
        alias for the text property
        """
        return self.text

    @property
    def role(self):
        """
        alias for the ownership flag property
        """
        return "user" if self.own else "assistant"

    @classmethod
    def from_chat_message(cls, chat_message: ChatMessage):
        """
        Create a Message instance from a Gradio ChatMessage object.

        This class method converts a Gradio ChatMessage object into a Message instance
        by extracting the text content and ownership information from the chat message.

        Args:
            chat_message (ChatMessage): A Gradio ChatMessage object representing a chat message.

        Returns:
            A message instance with the text content and ownership flag from the chat message.
        """
        return cls(text=chat_message.content, own=chat_message.role == "user")


    def to_chat_message(self):
        """
        Convert a Message instance to a Gradio ChatMessage object.

        This method converts a Message instance into a Gradio ChatMessage object
        by creating a new ChatMessage with the text content and ownership information.

        Returns:
            ChatMessage: A Gradio ChatMessage object representing the message content and ownership.
        """
        return ChatMessage(content=self.text, role="user" if self.own else "assistant")

@dataclass
class Conversation:
    """
    Conversation Dataclass

    This dataclass represents a complete conversation between a user and the chatbot.
    It maintains the chat history and configuration settings used for the conversation.

    Attributes:
        chat_history (List[ChatMessage]): A list of ChatMessage objects representing
                                          the sequence of messages in the conversation.
                                          Each message contains the text content and
                                          ownership information (user or assistant).

        conf (int): The configuration index specifying which chat settings to use.
                    Defaults to DEFAULT_CONFIGURATION. This index corresponds to
                    a specific configuration in the configurations_availables list.

    The class is designed to:
    - Maintain a complete history of chat exchanges
    - Track the configuration used for the conversation
    - Support multiple conversation configurations
    - Enable consistent chat behavior across sessions
    """
    chat_history: List[ChatMessage]
    conf: int = DEFAULT_CONFIGURATION
```


## Uso de excepciones personalizadas

Otra buena práctica es definir excepciones personalizadas para mejorar el control y la trazabilidad de los errores, especialmente en situaciones donde el flujo del programa se interrumpe. Por ejemplo, si se interrumpe la conexión entre el LLM y el back-end. el sistema no podrá proporcionar respuestas al usuario y será necesario terminar el proceso.

Las excepciones se definirán en el archivo `src/chatbot/Exceptions/__init__.py`. En particular, contemplaremos dos tipos de errores. Por un lado que el fichero no se haya subido correctamente, es decir, que el fichero no exista. Y por otro lado que no sea un archivo pdf, ya que, de momento, nuestro RAG solo va a admitir ficheros en este formato.

```python
"""This module contains custom exceptions for the chatbot application."""


class UploadFileException(RuntimeError):
    """Exception raised when there is an error uploading a file.

    This exception is a base class for file upload related errors.
    It inherits from RuntimeError and provides a custom message.

    Attributes:
        message: The error message describing the upload failure
    """
    def __init__(self, message: str):
        super().__init__(message)


class InvalidFileExtensionError(UploadFileException):
    """Exception raised when a file has an invalid extension.

    This exception is raised when attempting to upload a file with an
    unsupported or invalid file extension. It inherits from UploadFileException
    and provides specific error handling for file extension validation.

    Attributes:
        message: The error message describing the invalid file extension
    """
    def __init__(self, message: str = "The file should be a pdf"):
        super().__init__(message)
```


## Constantes y datos de configuración

En sistemas complejos, es recomendable centralizar las constantes y variables de configuración. Existen diversas maneras de definir estos valores, dependiendo de su propósito y del entorno en el que se ejecutará la aplicación. Por ejemplo, si se trata de un programa que van a descargar los usuarios y no queremos que tengan acceso al código fuente, podemos usar ficheros de texto plano o cualquier otro método que nos permita leer estas constantes cuando el proceso se ejecute. Por el contrario, cuando se trata de un sistema que van a emplear personas familiarizadas con el mismo, que contribuyen al desarrollo o buscamos rapidez suele ser habitual encontrar estos valores definidos dentro del código fuente.

En nuestro caso, optaremos por la segunda opción, ya que *Python* ofrece una sintaxis sencilla y clara para definir estas configuraciones.


### Configuración de Weaviate

Cuando configuramos Docker, mencionamos la importancia de determinar el puerto en el que la base de datos escucha. Ahora debemos indicar a nuestra aplicación cuál es ese puerto. También, en el archivo de configuración de `docker-compose.yml`, usamos la sección `ports` para redirigir tráfico de red entre el sistema operativo y el contenedor. En nuestro caso, usamos el mismo puerto para los dos sistemas, aunque no es necesario. Configurar la re-dirección entre puertos, nos permite poder acceder a la IP de nuestro ordenador, en vez de usar la del docker, que puede ser variable salvo que se configure explícitamente.

Para definir estos valores, utilizamos un `Enum`, una estructura de datos que agrupa valores semánticamente relacionados y actúa como un conjunto de contantes.

```python
class WeaviateConfig(Enum):
    """
    Enum with the default values used to configure Weaviate connection
    """
    DB_HOST = "localhost"
    DB_PORT = 8090
    VECTORIZER_HOST = "localhost"
    VECTORIZER_PORT = 50051
```


### Configuración de modelos LLM

En aplicaciones como la que estamos desarrollando, es posible querer probar diferentes modelos LLM. Este tipo de configuración no es necesario que sea constante durante todo el ciclo de vida del proceso, sino que podemos elegirla de forma dinámica. Incluso, es posible que queramos tener diferentes modelos según el tipo de usuario que interactúe en cada momento. Para ello, definimos una lista de presets accesible globalmente, donde especificamos configuraciones de modelos que hemos probado y validado previamente. Primero, definimos la estructura que tendrán todas las configuraciones.

```python
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
```

Una vez definida esta estructura, creamos las instancias de los modelos que consideremos adecuadas:

```python
configurations_availables = (
    BaseConfiguration("collection_a", "sentence-transformers/all-MiniLM-L6-v2", "llama3.2"),
    BaseConfiguration("collection_b",  "distilbert-base-nli-stsb-mean-tokens", "llama3.2"),
    BaseConfiguration("collection_c", "sentence-transformers/paraphrase-MiniLM-L6-v2", "phi3.5"),
    BaseConfiguration("collection_d", "distilbert-base-nli-stsb-mean-tokens", "phi3.5")
)
```

También podemos tener constantes independientes, como la configuración por defecto del *preset* que utilizará si no se especifica uno. Para ello añadimos una última linea:

```python
DEFAULT_CONFIGURATION = 3
```

El resultado del fichero de configuración quedará parecido al siguiente:

```python
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
```
