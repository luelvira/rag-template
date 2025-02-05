import logging
from chatbot.chat import Chat
logger = logging.getLogger(__name__)

chat = Chat()
if __name__ == '__main__':
    chat.run()
