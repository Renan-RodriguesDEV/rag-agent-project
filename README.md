# RAG Agent Project

Projeto simples de API em FastAPI com abordagem RAG (Retrieval-Augmented Generation) para responder perguntas com base em documentos PDF.

## Como funciona

- Carrega PDFs da pasta `documents/`
- Divide os textos em chunks
- Gera embeddings com Google GenAI
- Indexa em FAISS
- Recupera os trechos mais relevantes
- Gera a resposta com Gemini

## Estrutura

- `app.py`: inicializa a API e expõe o endpoint `/ask`
- `agents/core.py`: lógica de RAG (carregamento, embeddings, vetorstore e resposta)
- `schemas/schemas.py`: modelos Pydantic de entrada e saída
- `documents/`: pasta com os PDFs usados como base de conhecimento

## Pré-requisitos

- Python 3.10+
- UV instalado
- Conta/API Key do Google GenAI
- Docker (opcional, para execução em container)

## Instalação

1. Instale as dependências do projeto com UV:

```bash
uv sync
```

## Configuração

Crie um arquivo `.env` na raiz do projeto com:

```env
GOOGLE_API_KEY=sua_chave_aqui
```

## Execução

Inicie a API:

```bash
uv run uvicorn app:app --reload
```

Acesse a documentação interativa em:

- http://127.0.0.1:8000/docs

## Execução com Docker

### 1. Build da imagem

```bash
docker build -t rag-agent-api .
```

### 2. Execute o container

Use o arquivo `.env` criado na raiz do projeto:

```bash
docker run --rm -p 8000:8000 --env-file .env rag-agent-api
```

### 3. Teste a API

Abra:

- http://127.0.0.1:8000/docs

### 4. (Opcional) Montar a pasta de documentos como volume

Se você quiser atualizar os PDFs localmente sem reconstruir a imagem:

```bash
docker run --rm -p 8000:8000 --env-file .env -v ./documents:/app/documents rag-agent-api
```

No Windows PowerShell, se o comando acima falhar por path, use:

```powershell
docker run --rm -p 8000:8000 --env-file .env -v ${PWD}/documents:/app/documents rag-agent-api
```

## Endpoint principal

### POST /ask

Exemplo de corpo da requisição:

```json
{
  "question": "Quem é o engenheiro de software?",
  "context": {
    "role": "especialista em curriculos",
    "tone": "tecnico"
  }
}
```

Exemplo de resposta:

```json
{
  "response": "...",
  "sources": ["documents/seu_arquivo.pdf"],
  "is_found": true
}
```

## Observações

- Coloque seus PDFs na pasta `documents/` antes de consultar o endpoint.
- Avisos de ALTS no terminal são comuns fora de GCP e geralmente não impedem o funcionamento.
- Consulte a documentação oficial do langchain [Langchain](https://reference.langchain.com/python/).

## Autor

- Renan Rodrigues

## Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.
