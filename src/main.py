"""
This module serves as the entry point for the chatbot application. It initializes and runs
the main chat interface using Gradio's ChatInterface implementation.

The module handles:
- Application initialization
- Chat interface setup
- Main execution loop

Key components:
- Chat class: The main chatbot interface implementation
- Logging configuration: Basic logging setup for the application
- Main execution block: Launches the chatbot interface when run as a script

The module follows a simple workflow:
1. Initializes logging
2. Creates an instance of the Chat class
3. Launches the chat interface when executed as main
"""

import logging
from chatbot.chat import Chat
logger = logging.getLogger(__name__)

chat = Chat()
if __name__ == '__main__':
    chat.run()
