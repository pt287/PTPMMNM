import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_engine import RAGConfig, ask_question, build_rag_pipeline


class AskPayload(BaseModel):
    question: str


app = FastAPI(title="SmartDoc AI API")
frontend_dir = Path(__file__).parent / "frontend"

app.mount("/assets", StaticFiles(directory=frontend_dir), name="assets")


class AppState:
    def __init__(self) -> None:
        self.qa_chain = None
        self.doc_language = "unknown"
        self.chunk_count = 0


state = AppState()


@app.get("/")
def serve_frontend() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "indexed": state.qa_chain is not None,
        "doc_language": state.doc_language,
        "chunk_count": state.chunk_count,
    }


@app.post("/api/build-index")
async def build_index(
    files: List[UploadFile] = File(...),
    ollama_model: str = Form("qwen2.5:7b"),
    embedding_model: str = Form("sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
    temperature: float = Form(0.1),
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="Please upload at least one PDF file.")

    pdf_items = []
    for uploaded in files:
        if not uploaded.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {uploaded.filename}")
        content = await uploaded.read()
        pdf_items.append((uploaded.filename, content))

    config = RAGConfig(
        embedding_model=embedding_model,
        ollama_model=ollama_model,
        chunk_size=1000,
        chunk_overlap=100,
        top_k=3,
        temperature=temperature,
    )

    try:
        qa_chain, doc_language, chunk_count = build_rag_pipeline(pdf_items, config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    state.qa_chain = qa_chain
    state.doc_language = doc_language
    state.chunk_count = chunk_count

    return {
        "message": "RAG index built successfully.",
        "doc_language": doc_language,
        "chunk_count": chunk_count,
    }


@app.post("/api/ask")
def ask(payload: AskPayload) -> dict:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if state.qa_chain is None:
        raise HTTPException(status_code=400, detail="Build RAG index before asking a question.")

    start = time.time()
    try:
        answer, sources = ask_question(state.qa_chain, question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "answer": answer,
        "sources": sources,
        "response_time": round(time.time() - start, 2),
    }
