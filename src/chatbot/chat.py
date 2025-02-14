"""
chat.py

This module implements the main chatbot interface using Gradio's ChatInterface. It provides
the user interface and handles the interaction between the user and the chatbot service.

The module includes:
- Chat class: Main chatbot interface that handles user interactions, file uploads, and
  conversation management
- File upload functionality: Allows users to upload PDF documents for the chatbot to use
  as knowledge sources
- Conversation management: Maintains chat history and manages the flow of conversations
- Integration with chat services: Connects to the backend chat service for generating
  responses based on the conversation context

The Chat class extends Gradio's ChatInterface to provide a user-friendly interface with
the following features:
- Multimodal input (text and file uploads)
- Conversation history management
- Integration with the chat service for generating responses
- Error handling for file uploads and chat interactions

Key components:
- ChatInterface: Base class from Gradio for building chatbot interfaces
- chat_service: Backend service that handles the actual conversation processing
- Conversation: Data structure for managing chat history and configuration
- Message: Data structure for individual chat messages
"""

import logging

from gradio import ChatInterface, Error

from chatbot.services import files
from chatbot.services.chat_services import chat_service

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
    """
    Chat class that implements a chatbot interface using Gradio.

    This class extends Gradio's ChatInterface to provide a chatbot with file upload capabilities
    and conversation management. It integrates with the chat service for generating responses.

    Attributes:
        title (str): The title of the chatbot interface (default: "Chatbot demo")
        history (list): A list with the chat history


    Methods:
        __init__(title): Initializes the chatbot interface
        upload_file(file): Handles file uploads and processing
        ask(history): Sends conversation history to the chat service
        answer(msg, history): Processes user messages and generates responses
        run(): Launches the chatbot interface. Alias for self.launch()
    """

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
        """
        Processes a user message and generates a response.

        This method handles both text messages and file uploads. For file uploads, it processes
        each file through the upload_file method. For text messages, it interacts with the chat
        service to generate a response based on the conversation history.

        Args:
            msg (dict): A dictionary containing the message data with keys:
                       - "text": The text content of the message
                       - "files": List of files attached to the message
            history (list): The conversation history

        Returns:
            str: The response text or a message indicating file processing

        Raises:
            Error: If there's an error during file processing or chat service interaction
        """
        if msg["files"]:
            for file in msg["files"]:
                uploaded = upload_file(file)
                if not isinstance(uploaded, bool):
                    return []
            return "Processed files"
        if len(self.history) != len(history):
            self.history = [
                Message(text = msg["content"], own=msg["role"] == "user") for msg in history
            ]
        self.history.append(Message(text=msg["text"], own=True))
        response = chat_service.ask_memory(Conversation(chat_history=self.history))
        if isinstance(response, str):
            logger.error(response)
            raise Error("Se ha producido un error, por favor, intentelo más tarde")
        self.history.append(Message(text=response["respuesta"], own=False))
        return response["respuesta"]

    def run(self):
        """
        Launches the Gradio interface for the chat application.

        This method starts the Gradio Blocks interface, making the chat application
        accessible through a web interface. It handles the initialization and
        deployment of all UI components defined in the class.

        The method should be called after all components and event handlers have
        been properly configured.

        Returns:
            None

        Raises:
            gr.Error: If there are issues launching the Gradio interface
        """
        self.launch()
