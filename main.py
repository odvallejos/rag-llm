import os
from fastapi import FastAPI, Body, Request, Response
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

from create_chain import CreateChain
from dotenv import load_dotenv

class DataQuestion(BaseModel):
    user_name: str
    question: str

#cargo las variables de entorno, OPENAI_API_KEY
load_dotenv()

#creao la cadena en lengaje LCEL
chain = CreateChain()

app = FastAPI()

@app.get("/")
async def root():
    return "ok!"

@app.post("/response")
async def get_response(data: DataQuestion = Body(...)):
    user_name   = data.user_name
    question    = data.question
    session_id  = user_name

    if user_name and question:
        #considero user_name como session_id para recuperar el historial de conversaciones de un usuario guardado en la base de datos
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: SQLChatMessageHistory(
                session_id=session_id, connection_string="sqlite:///sqlite.db"
            ),
            input_messages_key="question",
            history_messages_key="history",
        )

        config = {"configurable": {"session_id": session_id}}

        res = chain_with_history.invoke({"question": question}, config=config)
    else:
        res = "Error, debe ingresar un usuario y una pregunta."
    return res
