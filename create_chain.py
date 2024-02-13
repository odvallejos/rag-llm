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
            chunk_size = 256,
            chunk_overlap  = 20
        )
        texts = text_splitter.create_documents([raw_text])

        #genero los embeddings de los textos y se almacenan en la base de datos
        #vectorstore = FAISS.from_texts(texts, embeddings)
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local("./db")

    else:
        vectorstore = FAISS.load_local("./db", embeddings)

    #genero el retriever que usará el modelo para buscar las preguntas en la base de datos por similitud
    #retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":4})
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":1})

    #creo el template con las instrucciones y las variables
    template = """
    <Paso1>
    Buscar la pregunta: {question} en la historia de esta sesión, considerando unicamente las preguntas que que se hicieron en el mismo idioma del input.
    {history}
    </Paso1>

    <Paso2>
    Si encontró la pregunta en el mismo idioma. Escribir solo la respuesta. Ir directo a RESPUESTA.
    Sino encontrá la pregunta seguir con el siguiente paso.
    </Paso2>

    <Paso3>
    Usar únicamente el siguiente contexto para responder.
    {context}
    Instrucciones:
    - Responder siempre en el mismo idioma de la pregunta.
    - Agrega un emoji que resuma la respuesta.
    - Responder en tercera persona.
    - Responder en una sola oración.
    </Paso3>

    Pregunta: {question}
    RESPUESTA:
    """

    prompt      = ChatPromptTemplate.from_template(template)
    context     = itemgetter("question") | retriever
    first_step  = RunnablePassthrough.assign(context=context)
    model       = ChatOpenAI()

    #creao la cadena en lengaje LCEL
    chain = first_step | prompt | model | StrOutputParser()

    return chain
