# Creación del contexto

Una vez definida la base de datos, es necesario proporcionar el contenido que la aplicación empleará para generar respuestas. Para ello, modificaremos la interfaz que previamente definida, en particular, el método encargado de proporcionar las respuestas del chat.

En el constructor del chat, observamos que el chatbot acepta varios tipos de ficheros gracias al argumento `multimodal`. Además, si probamos a enviar un mensaje, notaremos que este consiste en un diccionario con las claves `"text"` y `"files"`, siendo esta última una lista.

La nueva definición del método se divide en dos partes. Primero, evalúa si el mensaje contiene un fichero. En caso afirmativo, llama a la función `upload_file` cuya finalidad es procesar el fichero y almacenarlo en la base de datos. En caso contrario, se le proporciona el contenido del mensaje a un módulo intermedio entre el chat que estamos creando y el LLM que vayamos a usar.

Dentro de la clase `Chat` en el fichero `src/chatbot/chat.py`, modificamos el método `answer`:

```python
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
    # simulate the response from the service
    response = {"respuesta": "Aún no está implementado"}
    self.history.append(Message(text=response["respuesta"], own=False))
    return response["respuesta"]

```

Como podemos observar, el método hace referencia a varias clases y funciones que aún no existen y debemos implementar. Por tanto, vamos a actualizar la lista de módulos a importar.

```python
import logging
from gradio import ChatInterface, Error

from chatbot.services import files

from chatbot.Exceptions import UploadFileException
from chatbot.services.chat_services.types import Conversation, Message

logger = logging.getLogger(__name__)

```

También debemos actualizar el constructor, ya que vamos a almacenar el contenido del chat en una lista.

```python
history: list[Message]

def __init__(
        self,
        title: str = "Chatbot demo",
        **kargs
):
    super().__init__(
        fn=self.answer,
        multimodal=True,
        type="messages",
        fill_height=True,
        fill_width=True,
        title=title,
        save_history = True,
        **kargs
    )
    self.history = []
```

En el mismo archivo, definiremos la función `upload_file`. Esta se apoya en un servicio que procesará el contenido del fichero y lo cargará en la base de datos. También debe manejar posibles errores que puedan surgir.

```python
def upload_file(file):
    """
    Handles the upload and processing of a file.

    This method takes a file object, attempts to upload it using the files service,
    and handles various exceptions that may occur during the process.

    Args:
        file: The file object to be uploaded and processed

    Returns:
        bool: True if the file was successfully uploaded

    Raises:
        Error: If there are issues reading the file, uploading it, or other unexpected errors
              - FileNotFoundError: When the file cannot be found or accessed
              - UploadFileException: When there are specific issues with the file upload
              - Exception: For any other unexpected errors during the upload process
    """
    try:
        files.upload_file(file)
        return True
    except FileNotFoundError as e:
        raise Error("Problem reading the file") from e
    except UploadFileException as e:
        raise Error(str(e)) from e
    except Exception as e:
        raise Error(e) from e
```

El fichero final debe quedar similar al siguiente:

```python
import logging
from gradio import ChatInterface, Error

from chatbot.services import files

from chatbot.Exceptions import UploadFileException
from chatbot.services.chat_services.types import Conversation, Message

logger = logging.getLogger(__name__)


def upload_file(file):
    """
    Handles the upload and processing of a file.

    This method takes a file object, attempts to upload it using the files service,
    and handles various exceptions that may occur during the process.

    Args:
        file: The file object to be uploaded and processed

    Returns:
        bool: True if the file was successfully uploaded

    Raises:
        Error: If there are issues reading the file, uploading it, or other unexpected errors
              - FileNotFoundError: When the file cannot be found or accessed
              - UploadFileException: When there are specific issues with the file upload
              - Exception: For any other unexpected errors during the upload process
    """
    try:
        files.upload_file(file)
        return True
    except FileNotFoundError as e:
        raise Error("Problem reading the file") from e
    except UploadFileException as e:
        raise Error(str(e)) from e
    except Exception as e:
        raise Error(e) from e

class Chat(ChatInterface):
    history: list[Message]

    def __init__(
            self,
            title: str = "Chatbot demo",
            **kargs
    ):
        super().__init__(
            fn=self.answer,
            multimodal=True,
            type="messages",
            fill_height=True,
            fill_width=True,
            title=title,
            save_history = True,
            **kargs
        )
        self.history = []

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
        # simulate the response from the service
        response = {"respuesta": "Aún no está implementado"}
        self.history.append(Message(text=response["respuesta"], own=False))
        return response["respuesta"]


    def run(self):
        self.launch()

```


# Sistema de ficheros

Para trabajar con ficheros, es necesario contar con un componente que facilite su procesamiento. Para ello, vamos a crear un archivo python, llamado `src/chatbot/services/files.py`, que se encargará de procesar los ficheros proporcionados por el usuario, extraer el texto y almacenarlo en la base de datos.

Comenzando con la función `upload_file`, que es llamada desde `chat.py`. Esta función acepta la ruta a un archivo y devuelve `True` si el proceso se realiza correctamente. Para simplificar la implementación, nuestro programa solo procesará archivos PDF.

```python
def upload_file(file_path) -> bool:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File named {file_path} does not exist")
    if not file_path.lower().endswith('.pdf'):
        raise InvalidFileExtensionError(f"File {file_path} is not a PDF document")
    document = split_documents(read_file(file_path))
    selected_config = configurations_availables[DEFAULT_CONFIGURATION]
    try:
        #     simulate store the content into the database
        pass
    except Exception as e:
        raise UploadFileException(
            f"Error while processing the file {file_path}: {str(e)}"
            ) from e
    return True
```

Esta función emplea dos funciones internas del módulo `split_documents` y `read_file`. La primera divide el texto en fragmentos más pequeños y manejables, mientras que la segunda extrae el contenido del documento. Para la extracción de texto, emplearemos la biblioteca *PyMuPDF* de `Langchain`, que optimiza la extracción de texto dentro de documentos, gracias a técnicas de procesamiento del lenguaje natural.

```python
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
```

La función de `split_documents`, también se apoya en librerías de `langchain` que facilitan la insercción de textos, en concreto emplea `RecursiveCharacterTextSplitter`, como se puede ver a continuación

```python
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
```

Una vez añadidos los *imports* necesarios, el fichero de files.py debe quedar algo similar a:

```python
import os
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from chatbot.config import DEFAULT_CONFIGURATION
from chatbot.config import WeaviateConfig as WEAVIATE_CONFIG, configurations_availables

def upload_file(file_path) -> bool:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File named {file_path} does not exist")
    if not file_path.lower().endswith('.pdf'):
        raise InvalidFileExtensionError(f"File {file_path} is not a PDF document")
    document = split_documents(read_file(file_path))
    selected_config = configurations_availables[DEFAULT_CONFIGURATION]
    try:
        #     simulate store the content into the database
        pass
    except Exception as e:
        raise UploadFileException(
            f"Error while processing the file {file_path}: {str(e)}"
            ) from e
    return True

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
```
