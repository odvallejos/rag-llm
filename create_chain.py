import os
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from operator import itemgetter
from dotenv import load_dotenv

#cargo las variables de entorno, OPENAI_API_KEY
load_dotenv()

#funcion que crea l base de datos vectorial y la cadena en lenguaje LCEL
def CreateChain():
    embeddings = OpenAIEmbeddings()

    #sino existe la base de datos, la creo. Utilizo FAISS cono DB vectorial
    if not os.path.exists("./db"):
        doc_reader = PdfReader('./documento.pdf')

        #convierto el contenido del pdf en texto
        raw_text = ''
        for i, page in enumerate(doc_reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text

        #divido el texto en fragmentos de 256 caracteres y creo documentos
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"],
            chunk_size = 300,
            chunk_overlap  = 30
        )
        texts = text_splitter.create_documents([raw_text])

        #genero los embeddings de los textos y se almacenan en la base de datos
        #vectorstore = FAISS.from_texts(texts, embeddings)
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local("./db")

    else:
        vectorstore = FAISS.load_local("./db", embeddings)

    #genero el retriever que usará el modelo para buscar las preguntas en la base de datos por similitud
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":1})

    #creo el template con las instrucciones y las variables
    template = """Usa el siguiente contexto para responder las preguntas.
    Pensemos paso a paso siguiendo las siguientes Instrucciones:
    - Se debe responder en el mismo idioma de la pregunta.
    - Agrega un emoji que resuma la respuesta.
    - La respuesta debe estar en 3ra persona.
    - Si en la pregunta se menciona a una persona, el nombre también debe estar en la respuesta.
    - Responer siempre en una sola oración.

    {context}

    Pregunta: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(temperature=0)

    #creao la cadena en lengaje LCEL
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return chain
