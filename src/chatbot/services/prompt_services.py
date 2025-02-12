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
