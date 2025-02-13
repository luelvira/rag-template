# Configuración de la base de datos vectorial

Como mencionamos anteriormente, los RAG son sistemas que permiten interactuar con LLMs proporcionando un contexto específico. Para almacenar este contexto sin que sea necesario procesar el texto en cada petición, se usan bases de datos especializadas, que en vez de almacenar texto como cadenas de caracteres, emplean representaciones vectorial asociadas al contenido original. Estas representaciones son las que emplean los modelos de *Natual Language Processing* (NLP). En este caso, utilizaremos [weaviate](https://weaviate.io/), aunque existen otras opciones.


## Instalación de la base de datos

Para facilitar la instalación, utilizaremos [Docker](https://www.docker.com/) junto con [Docker compose](Https://docs.docker.com/compose/). Docker permite replicar entornos preconfigurados de forma aislada en nuestro sistema. Se puede entender como una máquina virtual ligera construida por capas de configuración, en donde cada capa describe lo que se denominan imágenes. El proceso de instalación de docker dependerá del sistema operativo en el que se esté ejecutando.

Una vez instalado Docker, Docker Compose, y con el servicio corriendo, utilizaremos la imagen oficial de Weaviate mediante el archivo `docker-compose.yml` para establecer su configuración. En la raíz del proyecto, creamos dicho archivo con el siguiente contenido:

```yaml
services:
  weaviate:
    command:
    - --host
    - 0.0.0.0
    - --port
    - '8090'  # Cambiado a 8090
    - --scheme
    - http
    image: cr.weaviate.io/semitechnologies/weaviate:1.27.1
    ports:
    - 8090:8090  # Mapeo de puertos actualizado
    - 50051:50051
    volumes:
    - weaviate_data:/var/lib/weaviate
    restart: on-failure:0
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      ENABLE_API_BASED_MODULES: 'true'
      CLUSTER_HOSTNAME: 'node1'
```


## Configuración del servicio en `docker-compose.yml`

En la sección de `services`, declaramos la imagen que usaremos. Para ello le asignamos:

1.  El nombre, en este caso `weaviate`.
2.  El comando de arranque. En los parámetros empleados para definir este comando, debemos incluir el puerto que va a usar la base de datos para escuchar. Esto es fundamental, ya que lo necesitaremos posteriormente para conectarnos a la misma.
3.  El nombre de la imagen que vamos a usar como referencia
4.  En el apartado `ports`, se define que puertos del sistema anfitrión serán redirigidos al contenedor. En lugar, de acceder directamente a la IP del docker, accedemos al *host*, que se encarga de redirigir el tráfico al contenedor. El resto de opciones establecen una configuración predeterminada para weaviate.

Es importante definir un *volumen* para garantizar la persistencia de los datos. Los volúmenes permiten que los datos almacenados y generados dentro del contenedor permanezcan, incluso después de que el contenedor se detenga o elimine. Dentro del fichero YML, encontramos la categoría `volumes`, la cual permite enlazar volúmenes existentes a rutas dentro del contenedor. Una vez vinculado, procedemos a crear el volumen. No nos hace falta que sea un espacio compartido entre el host y el contenedor, por tanto con incluir las siguientes lineas es suficiente:

```yaml
volumes:
    weaviate_data:
```

Una vez definida la configuración, procedemos a ejecutar el contenedor. Para ello, desde la raíz del proyecto, en una terminal, ejecutamos el siguiente comando:

```shell
docker-compose up -d
```

Esto iniciará la base de datos Weaviate en segundo plano, lista para ser utilizada en nuestro proyecto.
