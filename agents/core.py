"""classe do agente de IA RAG para re-uso atraves da API."""

import os

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from schemas.schemas import Context, QuestionSchema, ResponseSchema

load_dotenv()


# criando o modelo LLM com as credenciais da API do Google GenAI
def create_llm(
    api_key: str,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.1,
):
    # criando o modelo LLM do Google GenAI, que será utilizado para gerar as respostas às perguntas, utilizando o modelo "gemini-2.5-flash" e uma temperatura baixa para respostas mais precisas e menos criativas
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, api_key=api_key)


def create_embedding(
    api_key: str, model="models/gemini-embedding-001", temperature: float = 0.1
):
    # criando o modelo de embedding que será utilizado pelo agente RAG para transformar os documentos em vetores numéricos, facilitando a busca e recuperação de informações relevantes
    return GoogleGenerativeAIEmbeddings(
        model=model, temperature=temperature, google_api_key=api_key
    )


# criando o prompt template para o agente RAG
def create_prompt_template(context: Context):
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"Você é um assistente de IA especializado {context.role}, utilizando informações de fontes confiáveis. Responda de forma {context.tone}.",
            ),
            (
                "human",
                "responda com base nas informações fornecidas: {question}\n\nSendo o contexto as seguintes informações: {context}",
            ),
        ]
    )


def load_documents(base_path: str):
    documents = []
    for f in os.listdir(base_path):
        if f.endswith(".pdf"):
            # carregando o documento PDF usando o PyMuPDFLoader
            f_loader = PyMuPDFLoader(os.path.join(base_path, f))
            # extraindo o texto do PDF e adicionando à lista de documentos
            f_loaded = f_loader.load()
            # extends ao invés de append para adicionar os documentos extraídos do PDF à lista de documentos, pois load() retorna uma lista de documentos, e não um único documento
            documents.extend(f_loaded)
    return documents


# criando o banco de dados vetorial para armazenar as informações das fontes confiáveis
def create_vectorstore_retriver(
    embedding: GoogleGenerativeAIEmbeddings,
    documents: list = [],
    chunck_size: int = 300,
    chunk_overlap: int = 30,
):
    # criando o divisor de texto, sendo chunk_size o tamanho de caracteres do pedaço e chunk_overlap a sobreposição (qntos caracteres antes) entre os pedaços
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunck_size, chunk_overlap=chunk_overlap
    )
    # dividindo os documentos em pedaços menores para facilitar a indexação e recuperação
    chunks = splitter.split_documents(documents)
    # criando o db vetorial a partir dos pedaços de documentos e do modelo de embedding fornecido
    vectorstore_db = FAISS.from_documents(documents=chunks, embedding=embedding)
    # retornando o buscador (retriver) do db vetorial, para que o agente recupere as informações atraves da busca
    # busca do tipo pontuação de similaridade
    search_type = "similarity_score_threshold"
    # definindo o limiar de pontuação (score_threshold) para considerar um documento relevante na busca, e o número máximo de documentos a serem retornados (k)
    search_kwargs = {"score_threshold": 0.3, "k": 4}
    return vectorstore_db.as_retriever(
        search_type=search_type, search_kwargs=search_kwargs
    )


def ask_question(question: QuestionSchema):
    # modelo LLM para responder as perguntas
    llm = create_llm(api_key=os.getenv("GOOGLE_API_KEY"))
    # modelo de embedding para transformar os documentos em vetores numéricos
    embedding = create_embedding(api_key=os.getenv("GOOGLE_API_KEY"))
    # carregando os documentos PDF da pasta "documents"
    documents = load_documents(base_path="documents")
    # criando o buscador (retriver) do banco de dados vetorial a partir dos documentos carregados e do modelo de embedding
    retriever = create_vectorstore_retriver(embedding=embedding, documents=documents)
    # recuperando os documentos relevantes para a pergunta utilizando o retriever
    docs = retriever.invoke(question.question)
    if not docs:
        return ResponseSchema(
            response="Desculpe, não encontrei informações relevantes para responder à sua pergunta.",
            sources=[],
            is_found=False,
        )
    # criando o prompt template para o agente RAG
    prompt_template = create_prompt_template(
        context=question.context or Context(role="TI", tone="tecnico")
    )
    # criando a cadeia de documentos para combinar os documentos recuperados em uma resposta coerente
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
    # gerando a resposta para a pergunta utilizando a cadeia de documentos combinados
    response = combine_docs_chain.invoke({"question": question, "context": docs})
    return ResponseSchema(
        response=response,
        sources=[doc.metadata.get("source", "desconhecida") for doc in docs],
        is_found=True,
    )
