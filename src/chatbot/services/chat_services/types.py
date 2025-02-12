from dataclasses import dataclass
from typing import List
from gradio import ChatMessage

from chatbot.config import BaseConfiguration, DEFAULT_CONFIGURATION, Prompts

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
