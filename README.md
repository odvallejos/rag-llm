# RAG con LLM

## Objetivo

Se requiere desarrollar una solucion simple de tipo RAG (retrieved augmented generation), en donde, mediante una API se permita interactuar con un LLM con el fin de generar una respuesta (sobre un documento en particular) a la pregunta brindada por el usuario.

## Solución

* Se desarrolló una API utilizando FastAPI.
* Se utiliza LangChain como framework LLM
* Se utiliza OpenAI como modelo de embedding y chat
* Se utiliza FAISS como base de datos vectorial

## Embeddings

* El documento proporcionado en formato word se convirtió a PDF para tratarlo en LangChain
* Parámetros: *chuncks = 300*, *chunk_overlap = 30*

## Base de datos vectorial

* Se utiliza FAISS como base de datos vectorial, se persiste el contenido de manera física la primera vez que se ejecuta
* Funciona como *retriever* en LangChain
* La búsqueda se realiza por *similaridad*

## Prompt

Se utiliza *prompt template* de LangChain con las instrucciones:

```
Usa el siguiente contexto para responder las preguntas.
Pensemos paso a paso siguiendo las siguientes Instrucciones:
- Se debe responder en el mismo idioma de la pregunta.
- Agrega un emoji que resuma la respuesta.
- La respuesta debe estar en 3ra persona.
- Si en la pregunta se menciona a una persona, el nombre también debe estar en la respuesta.
- Responer siempre en una sola oración.

{context}

Pregunta: {question}
```

## Ejecución en entorno local

* El endpoint es {url}/response de tipo POST y la estructura del Request Body es: *user_name* y *question*
* Descargar/clonar el repositorio
* Crear un entorno virtual
* Instalar las librerías: `pip install -r requirements.txt`
* Levantar el servidor: `uvicorn main:app --reload`
* El archivo .env contiene las variables de entorno necesarias para ejecutar la aplicación, OPENAI_API_KEY, debes colocar allí tu API Key de OpenAI
* Utilizar el endpoint, por ejemplo con VS Code:

```python
import requests
url = "http://127.0.0.1:8000/response"
data = {
    "user_name": "odvallejos",
    "question": "Quien es Zara?"
}
res = requests.post(url, json=data)
if res.status_code == 200:
    print(res.json())
else:
    print("Error")

```

Ejemplos de preguntas:
```
Quien es Zara?
```
```
What did Emma decide to do?
```
```
What is the name of the magical flower?
```

## Docker

El archivo Dockerfile que forma parte del repositorio permite crear una imagen y poder ejecutar la solución en un contenedor Docker

## Postman

Usar la collección en Postman

[Ir a Postman con la colección RAG con LLM](https://www.postman.com/techmkt/workspace/danielv/collection/12610018-e9b5e796-9e89-4de2-a328-05f538a8366a?action=share&creator=12610018)