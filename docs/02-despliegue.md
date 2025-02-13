# Implementación de la interfaz

Como se ha mencionado, vamos a usar <https://www.gradio.app/> para el desarrollo de la interfaz gráfica. Gradio es framework de python pensado para el desarrollo de experimentos y demostraciones de software de inteligencia artifical que requieren de una interfaz con la que interactuar, sin ser el foco principal del desarrollo.

Gradio ofrece la clase `ChatInterface` que permite levantar una aplicación web muy similar a *ChatGPT*. Partiendo de esta clase, vamos a crear una nueva que herede de `ChatInterface`, y que permita añadir nuevas funcionalidades como procesar ficheros y usar nuestro LLM para generar las respuestas.


## Definición de la clase

Dentro de la carpeta `src/chatbot`, Creamos el fichero `src/chatbot/chat.py` e importamos las librerías necesarias.

```python
import logging

from gradio import ChatInterface, Error
```

A continuación, definimos nuestra clase personalizada.

```python
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
```

El constructor establece varias configuraciones por defecto a la instancia de `ChatInterface`:

1.  Se define la función que permite establecer la respuesta del chat.
2.  Se permite que el chat acepte varios tipos de archivos, no solo texto.
3.  Se adapta la interfaz al tamaño de la pantalla.
4.  Se proporciona un título por defecto, manteniendo la posibilidad de modificarlo.
5.  Se habilita la opción de guardar las conversaciones en el navegador.
6.  Se mantiene la posibilidad de utilizar el resto de argumentos aceptados por el contructor de la clase `ChatInterface`


## Definición del resto de métodos


### Método `run`

Necesitamos un método que inicie la aplicación. `ChatInterface` cuenta el método `launch`, que cumple esta función. Para facilitar su uso, creamos el método `run` que actúe como alias.

```python
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
```


### Método `answer`

La función o método encargada de controlar el intercambio de mensajes entre el usuario y el sistema, a la que hemos llamado `answer` recibe dos argumentos, el mensaje del usuario y el historial del chat. Retorna un texto como respuesta o un componente de Gradio. En esta versión inicial, simplemente devolverá el mensaje del usuario.

```python
def answer(self, message: str, history: list):
    return f"El usuario ha introducido: {message}"
```


### Archivo final

El código final del archivo src/chatbot/chat.py queda estructurado de la siguiente manera:

```python
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
```


## Script de ejecución

Finalmente, necesitamos un mecanismo que nos permita ejecutar la aplicación. Para ello, vamos a crear un archivo en la raiz del proyecto que importe la clase que acabamos de definir, crear una instancia y ejecutar el programa. Creamos el fichero `src/main.py` con el siguiente contenido:

```python
import logging
from chatbot.chat import Chat
logger = logging.getLogger(__name__)

chat = Chat()
if __name__ == '__main__':
    chat.run()
```

Para ejecutar el script, accedemos a la carpeta `src` y ejecutamos el fichero main:

```shell
cd src
python main.py
```

Si la ejecución es correcta, se mostrará un mensaje similar a:

```shell
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```
