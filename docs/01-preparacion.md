# Retrieval augmented generation (RAG)

Un RAG es un sistema diseñado para mejorar la experiencia del usuario al interactuar con un LLM. Para lograrlo, integra los siguientes componentes:

1.  **Interfaz** Componente diseñado para introducir y mostrar los *prompts* y las respuestas. En este caso, se desarrollará con [gradio](https://www.gradio.app/).
2.  **Base de datos**: Herramienta utilizada para almacenar y organizar información. En sistemas que empleen LLMs predominan las bases de datos vectoriales, ya que permiten realizar búsquedas basadas en la similitud entre la representación vectorial del contexto y del *prompt* introducido por el usuario.
3.  **Large Language Model (LLM)** Un LLM es un modelo de lenguaje entrenado mediante aprendizaje autosupervisado que consta de una red neuronal de gran tamaño. Un ejemplo son los modelos GPT de OpenAI, o DeepSeek.


## Preparación del entorno

El lenguaje de programación utilizado será python debido a su predominio en tareas de inteligencia artificial y su facilidad de uso. Una vez este instalado, es conveniente trabajar en entornos virtuales.

Un entorno virtual o *virutalenv* es un entorno de ejecución que permite instalar una versión específica de python y modificar la variable `$PYTHONPATH` de modo que las librerías instaladas para un desarrollo no interfieran con el resto de librerías instaladas en el sistema.

Existen varios *frameworks* en python que permiten crear estos entornos virtuales. En esta guía se usará `pyenv`, aunque se puede usar cualquier otra alternativa como venv o conda.

Para la creación y activación del entorno virtual con `pyenv`, primero se debe establecer qué versión de Python se va a utilizar. Puesto que no hay ningún requisito específico, se puede emplear la versión predeterminada del sistema o la última disponible. En este caso se usará Python 3.12. Una vez determinada que versión se usará, procedemos a instalarla y crear el entorno virtual:

```shell
pyenv install 3.12
pyevn virtualenv 3.12 gradio-venv
pyenv activate gradio-venv
```

Para verificar que el entorno virtual está activado correctamente, en la terminal debería aparecer el nombre del entorno entre paréntesis, por ejemplo: `(gradio-venv)`.

En el repositorio [rag-template](https://github.com/luelvira/rag-template), se encuentra la lista de dependencias necesarias. Para evitar problemas con las versiones, no se han fijado explícitamente. Al no forzar las versiones `pip` seleccionará automáticamente aquellas que sean compatibles entre sí.

**Nota:** En caso de encontrar problemas y querer emplear las mismas versiones empleadas en el desarrollo de esta guía, se ha incluido el fichero `requirements-with-versions.txt`, el cual si incluye la versión de forma explícita.

Para instalar las dependencias, basta con ejecutar el siguiente comando, asegurándonos que estemos con el entorno virutal activo.

```shell
pip install -r requirements.txt
```

El siguiente paso para la preparación del entorno consiste en la creación de la estructura de carpetas del mismo.

```shell
mkdir -p src/chatbot docs tests
touch src/main.py
touch .gitignore
```

```shell
tree .
.
├── docs
├── LICENSE
├── requirements.txt
├── requirements-with-versions.txt
├── src
│   ├── chatbot
│   └── main.py
└── tests
```
