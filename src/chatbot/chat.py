# src/chatbot/chat.py 
import logging

from gradio import ChatInterface, Error

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
    def answer(self, message: str, history: list):
        return f"El usuario ha introducido: {message}"

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
