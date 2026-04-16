from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agents.core import ask_question
from schemas.schemas import QuestionSchema, ResponseSchema

app = FastAPI(
    title="RAG Agent API",
    description="API para o agente RAG que responde perguntas sobre curriculos utilizando informações de fontes confiáveis.",
)
# Permitir CORS para todas as origens, métodos e cabeçalhos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ask", response_model=ResponseSchema)
def ask(question: QuestionSchema):
    resp = ask_question(question)
    return resp


@app.get("/")
def read_root():
    return {
        "message": "Bem-vindo à API do Agente RAG! Use o endpoint /ask para fazer perguntas."
    }
