import os
from fastapi import FastAPI, Body, Request, Response
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

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
        #realiza la consulta con la pregunta
        res = chain.invoke(question)
    else:
        res = "Error, debe ingresar un usuario y una pregunta."
    return res
