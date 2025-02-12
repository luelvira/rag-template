"""
chat_service.py

This module implements the core chat service functionality for the chatbot application.
It handles conversation management, memory retrieval, and response generation using
a RAG (Retrieval-Augmented Generation) architecture.

The module provides:
- Conversation processing with memory retrieval
- Integration with Weaviate vector store for document retrieval
- Response generation using Ollama LLM
- Configuration management for different chat settings

Key components:
- ask_memory: Main function for processing conversations with memory retrieval
- _get_retriever: Helper function for setting up document retriever
- _get_rag_chain: Helper function for creating RAG chain
- Integration with WeaviateManager for vector store operations
- Configuration handling using BaseConfiguration and Prompts

The module uses:
- LangChain for chain creation and document processing
- Ollama for LLM integration
- Weaviate for vector store operations
- Custom types for conversation and chatbot options

Typical workflow:
1. Conversation object is received with chat history
2. Configuration is selected based on conversation settings
3. Document retriever is set up using Weaviate
4. RAG chain is created with appropriate prompt
5. Response is generated using LLM with retrieved context

Exceptions:
- Handles errors during RAG chain invocation
- Manages Weaviate connection cleanup
- Returns error messages for insufficient data cases
"""

import os

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaLLM
from chatbot.config import (
    configurations_availables,
    Prompts,
    OLLAMA_API_PORT,
)
from chatbot.WeaviateManager.manager import WeaviateManager
from chatbot.WeaviateManager.arguments import RetrieverParameters
from chatbot.services.chat_services.types import Conversation, ChatbotOptions
from chatbot.services.prompt_services import create_chat_prompt


def ask_memory(conversation: Conversation):
    """
    Processes a conversation with memory retrieval to generate a response.

    This method handles the complete conversation processing workflow including:
    - Setting up the chatbot configuration based on conversation settings
    - Creating the document retriever using Weaviate vector store
    - Building the RAG (Retrieval-Augmented Generation) chain
    - Generating a response using the LLM with retrieved context
    - Formatting the response with source document information

    Args:
        conversation (Conversation): The conversation object containing:
            - chat_history: List of previous messages in the conversation
            - conf: Configuration index for selecting chat settings

    Returns:
        dict: A dictionary containing:
            - "respuesta": The generated response text
            - "documentos_fuente": List of source documents with metadata including:
                * "documento": PDF resource name
                * "página": Page number
                * "dificultad": Difficulty level

        str: Error message if an exception occurs during processing

    Raises:
        Handles exceptions internally and returns error messages
        - Connection errors with Weaviate vector store
        - Errors during RAG chain invocation
        - Issues with LLM response generation
    """
    options = ChatbotOptions(
        endpoint=f"http://localhost:{os.getenv('OLLAMA_API_PORT', str(OLLAMA_API_PORT))}",
        configuration=configurations_availables[conversation.conf],
        propts=Prompts(),
    )
    query = conversation.chat_history[-1].content
    llm, retriever, weaviate_manager = _get_retriever(query, options)
    if retriever is None:
        return {
            "respuesta": "No dispongo de datos suficientes como para responder esa pregunta",
            "documentos_fuente": [],
        }

    rag_chain = _get_rag_chain(
        options.propts.default_prompt,
        llm,
        retriever
    )
    chat_history = [
        HumanMessage(content=x.content) if x.role == "user"
        else AIMessage(content=x.content)
        for x in conversation.chat_history
    ]
    try:
        response = rag_chain.invoke({"input":query, "chat_history": chat_history})
    except (ConnectionError, ValueError, RuntimeError) as e:
        weaviate_manager.close()
        return str(e)
    finally:
        weaviate_manager.close()
    return {
        "respuesta": response['answer'],
        "documentos_fuente": [
            {
                "documento":  doc.metadata['pdf_resource_name'],
                "página":     doc.metadata['page_number'],
                "dificultad": doc.metadata['difficulty_level']
            }
            for doc in response['context']
        ],
    }


def _get_retriever(query, options: ChatbotOptions):
    """
    Retrieves the necessary components for RAG (Retrieval-Augmented Generation) chain.

    This method initializes and returns three key components needed for the RAG process:
    1. LLM (Language Model) instance configured with Ollama
    2. WeaviateManager instance for vector store operations
    3. Retriever instance for document retrieval

    Args:
        query (str): The user query to be used for document retrieval
        options (ChatbotOptions): Configuration options including:
            - endpoint: Ollama API endpoint
            - configuration: Model and collection configuration
            - propts: Prompt settings

    Returns:
        tuple: A tuple containing:
            - llm: Initialized OllamaLLM instance
            - retriever: Document retriever instance
            - weaviate_manager: WeaviateManager instance. This instance needs to be closed

    Note:
        The retriever is configured with parameters from the options, including:
        - Embedding model name
        - Collection name
        - Query string
    """
    llm = OllamaLLM(
        base_url=options.endpoint,
        model=options.configuration.llm_model,
        temperature=0.1
    )
    weaviate_manager = WeaviateManager()
    retriever = weaviate_manager.get_retriever(
        RetrieverParameters(
            model_name=options.configuration.embed_model,
            collection_name=options.configuration.collection_name,
            query=query
        )
    )
    return llm, retriever, weaviate_manager

def _get_rag_chain(prompt, llm, retriever):
    """
    Creates a RAG (Retrieval-Augmented Generation) chain for question answering.

    This method constructs a RAG chain by combining:
    1. A chat prompt template for formatting the question and context
    2. A document processing chain for handling retrieved documents
    3. A retrieval chain that integrates document retrieval with question answering

    Args:
        prompt (str): The prompt template to use for formatting the question and context
        llm (OllamaLLM): The language model instance to use for generating answers
        retriever (BaseRetriever): The document retriever instance for fetching relevant documents

    Returns:
        Runnable: A RAG chain that can be invoked with a question to generate an answer

    The RAG chain performs the following steps:
    1. Retrieves relevant documents based on the question
    2. Formats the question and documents using the provided prompt template
    3. Generates an answer using the language model

    Note:
        The returned RAG chain should be invoked with a dictionary containing:
        - "input": The user's question
        - "context": The retrieved documents (automatically handled by the chain)
    """
    chat_prompt = create_chat_prompt(prompt)
    question_answering_chain=create_stuff_documents_chain(llm, chat_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answering_chain)
    return rag_chain
