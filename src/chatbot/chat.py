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
        response = chat_service.ask_memory(Conversation(chat_history=self.history))
        self.history.append(Message(text=response["respuesta"], own=False))
        return response["respuesta"]


    def run(self):
        self.launch()
