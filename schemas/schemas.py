from typing import Literal, Optional

from pydantic import BaseModel


class Context(BaseModel):
    role: str
    tone: Literal["curto", "formal", "tecnico", "criativo", "humoristico"] = "curto"


class QuestionSchema(BaseModel):
    """Modelo de pergunta na requisição da API."""

    question: str
    context: Optional[Context] = None


class ResponseSchema(BaseModel):
    """Modelo de resposta da API."""
    response: str
    sources: list[str]
    is_found: bool = False
