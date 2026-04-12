import time
import json
import sqlite3
from threading import Lock
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_engine import RAGConfig, ask_question, build_rag_pipeline


class AskPayload(BaseModel):
    question: str


app = FastAPI(title="SmartDoc AI API")
frontend_dir = Path(__file__).parent / "frontend"
db_dir = Path(__file__).parent / "data"
db_path = db_dir / "history.db"
uploads_dir = db_dir / "uploads"
db_lock = Lock()

app.mount("/assets", StaticFiles(directory=frontend_dir), name="assets")


class AppState:
    def __init__(self) -> None:
        self.qa_chain = None
        self.doc_language = "unknown"
        self.chunk_count = 0
        self.current_session_id: Optional[int] = None
        self.conversation_history: List[tuple[str, str]] = []


state = AppState()


def _get_connection() -> sqlite3.Connection:
    db_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with db_lock:
        with _get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS index_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    doc_language TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    ollama_model TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    temperature REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_pdfs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    file_name TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    stored_path TEXT,
                    FOREIGN KEY(session_id) REFERENCES index_sessions(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS qa_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    session_id INTEGER,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    response_time REAL NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES index_sessions(id) ON DELETE SET NULL
                )
                """
            )

            pdf_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(session_pdfs)").fetchall()
            }
            if "stored_path" not in pdf_columns:
                conn.execute("ALTER TABLE session_pdfs ADD COLUMN stored_path TEXT")
            conn.commit()


def _safe_file_name(name: str) -> str:
    base_name = Path(name).name.strip()
    return base_name or "uploaded.pdf"


def _write_session_file(session_id: int, file_name: str, content: bytes) -> str:
    session_dir = uploads_dir / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    clean_name = _safe_file_name(file_name)
    target = session_dir / clean_name
    if target.exists():
        stem = target.stem
        suffix = target.suffix
        counter = 2
        while target.exists():
            target = session_dir / f"{stem}_{counter}{suffix}"
            counter += 1

    target.write_bytes(content)
    return str(target)


def _store_index_session(
    *,
    doc_language: str,
    chunk_count: int,
    ollama_model: str,
    embedding_model: str,
    temperature: float,
    files: List[Dict[str, Any]],
) -> int:
    with db_lock:
        with _get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO index_sessions (
                    doc_language, chunk_count, ollama_model, embedding_model, temperature
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (doc_language, chunk_count, ollama_model, embedding_model, temperature),
            )
            session_id = int(cursor.lastrowid)

            db_file_rows = []
            for item in files:
                stored_path = _write_session_file(session_id, item["file_name"], item["content"])
                db_file_rows.append(
                    (
                        session_id,
                        item["file_name"],
                        item["file_size"],
                        stored_path,
                    )
                )

            conn.executemany(
                """
                INSERT INTO session_pdfs (session_id, file_name, file_size, stored_path)
                VALUES (?, ?, ?, ?)
                """,
                db_file_rows,
            )
            conn.commit()

    return session_id


def _store_qa(
    *,
    session_id: Optional[int],
    question: str,
    answer: str,
    sources: List[Dict[str, Any]],
    response_time: float,
) -> None:
    with db_lock:
        with _get_connection() as conn:
            conn.execute(
                """
                INSERT INTO qa_history (session_id, question, answer, sources_json, response_time)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, question, answer, json.dumps(sources, ensure_ascii=False), response_time),
            )
            conn.commit()


def _fetch_upload_history(limit: int) -> List[Dict[str, Any]]:
    with db_lock:
        with _get_connection() as conn:
            sessions = conn.execute(
                """
                SELECT id, created_at, doc_language, chunk_count, ollama_model, embedding_model, temperature
                FROM index_sessions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            result: List[Dict[str, Any]] = []
            for row in sessions:
                pdf_rows = conn.execute(
                    """
                    SELECT file_name, file_size
                    FROM session_pdfs
                    WHERE session_id = ?
                    ORDER BY id ASC
                    """,
                    (row["id"],),
                ).fetchall()

                pdf_files = [
                    {"file_name": file_row["file_name"], "file_size": file_row["file_size"]}
                    for file_row in pdf_rows
                ]

                result.append(
                    {
                        "session_id": row["id"],
                        "created_at": row["created_at"],
                        "doc_language": row["doc_language"],
                        "chunk_count": row["chunk_count"],
                        "ollama_model": row["ollama_model"],
                        "embedding_model": row["embedding_model"],
                        "temperature": row["temperature"],
                        "pdf_files": pdf_files,
                    }
                )

    return result


def _fetch_qa_history(limit: int) -> List[Dict[str, Any]]:
    with db_lock:
        with _get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, session_id, question, answer, sources_json, response_time
                FROM qa_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    records: List[Dict[str, Any]] = []
    for row in rows:
        try:
            sources = json.loads(row["sources_json"])
        except json.JSONDecodeError:
            sources = []

        records.append(
            {
                "qa_id": row["id"],
                "created_at": row["created_at"],
                "session_id": row["session_id"],
                "question": row["question"],
                "answer": row["answer"],
                "response_time": row["response_time"],
                "sources": sources,
            }
        )

    return records


def _fetch_session_files(session_id: int) -> List[Dict[str, Any]]:
    with db_lock:
        with _get_connection() as conn:
            rows = conn.execute(
                """
                SELECT file_name, file_size, stored_path
                FROM session_pdfs
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()

    return [
        {
            "file_name": row["file_name"],
            "file_size": row["file_size"],
            "stored_path": row["stored_path"],
        }
        for row in rows
    ]


def _fetch_qa_by_session(session_id: int, limit: int) -> List[Dict[str, Any]]:
    with db_lock:
        with _get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, session_id, question, answer, sources_json, response_time
                FROM qa_history
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

    records: List[Dict[str, Any]] = []
    for row in rows:
        try:
            sources = json.loads(row["sources_json"])
        except json.JSONDecodeError:
            sources = []

        records.append(
            {
                "qa_id": row["id"],
                "created_at": row["created_at"],
                "session_id": row["session_id"],
                "question": row["question"],
                "answer": row["answer"],
                "response_time": row["response_time"],
                "sources": sources,
            }
        )

    return records


def _fetch_session_info(session_id: int) -> Optional[Dict[str, Any]]:
    with db_lock:
        with _get_connection() as conn:
            row = conn.execute(
                """
                SELECT id, created_at, doc_language, chunk_count, ollama_model, embedding_model, temperature
                FROM index_sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()

    if row is None:
        return None

    return {
        "session_id": row["id"],
        "created_at": row["created_at"],
        "doc_language": row["doc_language"],
        "chunk_count": row["chunk_count"],
        "ollama_model": row["ollama_model"],
        "embedding_model": row["embedding_model"],
        "temperature": row["temperature"],
    }


@app.on_event("startup")
def startup_event() -> None:
    _init_db()


@app.get("/")
def serve_frontend() -> FileResponse:
    return FileResponse(
        frontend_dir / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "indexed": state.qa_chain is not None,
        "doc_language": state.doc_language,
        "chunk_count": state.chunk_count,
    }


@app.get("/api/history")
def history(limit: int = 15) -> dict:
    safe_limit = max(1, min(limit, 100))
    return {
        "uploads": _fetch_upload_history(safe_limit),
        "qa": _fetch_qa_history(safe_limit),
    }


@app.get("/api/sessions/{session_id}/history")
def session_history(session_id: int, limit: int = 30) -> dict:
    safe_limit = max(1, min(limit, 200))
    session = _fetch_session_info(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    files = _fetch_session_files(session_id)
    qa_records = _fetch_qa_by_session(session_id, safe_limit)
    return {
        "session": {
            "session_id": session["session_id"],
            "created_at": session["created_at"],
            "doc_language": session["doc_language"],
            "chunk_count": session["chunk_count"],
            "pdf_files": [
                {"file_name": item["file_name"], "file_size": item["file_size"]}
                for item in files
            ],
        },
        "qa": qa_records,
    }


@app.post("/api/sessions/{session_id}/activate")
def activate_session(session_id: int) -> dict:
    session = _fetch_session_info(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    files = _fetch_session_files(session_id)
    if not files:
        raise HTTPException(status_code=400, detail="Session has no files.")

    pdf_items: List[Any] = []
    missing_files: List[str] = []
    for item in files:
        stored_path = item.get("stored_path")
        if not stored_path:
            missing_files.append(item["file_name"])
            continue

        path = Path(stored_path)
        if not path.exists():
            missing_files.append(item["file_name"])
            continue

        pdf_items.append((item["file_name"], path.read_bytes()))

    if missing_files:
        raise HTTPException(
            status_code=409,
            detail=(
                "Không thể khôi phục một số file của session này: "
                + ", ".join(missing_files)
                + ". Hãy upload lại file để tiếp tục hỏi đáp."
            ),
        )

    config = RAGConfig(
        embedding_model=session["embedding_model"],
        ollama_model=session["ollama_model"],
        chunk_size=1000,
        chunk_overlap=100,
        top_k=3,
        temperature=session["temperature"],
    )

    try:
        qa_chain, doc_language, chunk_count = build_rag_pipeline(pdf_items, config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    state.qa_chain = qa_chain
    state.doc_language = doc_language
    state.chunk_count = chunk_count
    state.current_session_id = session_id
    state.conversation_history = []

    return {
        "message": "Session activated.",
        "session_id": session_id,
        "doc_language": doc_language,
        "chunk_count": chunk_count,
    }


@app.post("/api/build-index")
async def build_index(
    files: List[UploadFile] = File(...),
    ollama_model: str = Form("qwen2.5:0.5b"),
    embedding_model: str = Form("sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
    temperature: float = Form(0.1),
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="Please upload at least one PDF or DOCX file.")

    doc_items = []
    file_records: List[Dict[str, Any]] = []
    for uploaded in files:
        lower_name = uploaded.filename.lower()
        if not (lower_name.endswith(".pdf") or lower_name.endswith(".docx")):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {uploaded.filename}. Only PDF and DOCX are supported.",
            )
        content = await uploaded.read()
        doc_items.append((uploaded.filename, content))
        file_records.append(
            {
                "file_name": uploaded.filename,
                "file_size": len(content),
                "content": content,
            }
        )

    config = RAGConfig(
        embedding_model=embedding_model,
        ollama_model=ollama_model,
        chunk_size=1000,
        chunk_overlap=100,
        top_k=3,
        temperature=temperature,
    )

    try:
        qa_chain, doc_language, chunk_count = build_rag_pipeline(doc_items, config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    state.qa_chain = qa_chain
    state.doc_language = doc_language
    state.chunk_count = chunk_count
    state.conversation_history = []
    state.current_session_id = _store_index_session(
        doc_language=doc_language,
        chunk_count=chunk_count,
        ollama_model=ollama_model,
        embedding_model=embedding_model,
        temperature=temperature,
        files=file_records,
    )

    return {
        "message": "RAG index built successfully.",
        "doc_language": doc_language,
        "chunk_count": chunk_count,
        "session_id": state.current_session_id,
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
        answer, sources = ask_question(
            state.qa_chain,
            question,
            conversation_history=state.conversation_history,
        )
    except Exception as exc:
        detail = str(exc)
        if "requires more system memory" in detail.lower():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model hiện tại cần nhiều RAM hơn máy đang có. "
                    "Hãy Build RAG Index lại với model nhẹ hơn như `qwen2.5:0.5b` hoặc `qwen2.5:1.5b`."
                ),
            ) from exc
        if "cudamalloc failed" in detail.lower() or "out of memory" in detail.lower():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Ollama bị thiếu bộ nhớ GPU (CUDA OOM). Hệ thống đã cấu hình ưu tiên CPU, "
                    "hãy Build RAG Index lại và thử lại. Nếu vẫn lỗi, dùng model nhỏ hơn (ví dụ qwen2.5:3b)."
                ),
            ) from exc
        if "llama runner process has terminated" in detail:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Ollama runner bị dừng đột ngột khi sinh câu trả lời. "
                    "Hãy kiểm tra `ollama serve`, đảm bảo model đã pull, rồi thử lại."
                ),
            ) from exc
        raise HTTPException(status_code=500, detail=detail) from exc

    response_time = round(time.time() - start, 2)
    state.conversation_history.append(("user", question))
    state.conversation_history.append(("assistant", answer))
    if len(state.conversation_history) > 24:
        state.conversation_history = state.conversation_history[-24:]

    _store_qa(
        session_id=state.current_session_id,
        question=question,
        answer=answer,
        sources=sources,
        response_time=response_time,
    )

    return {
        "answer": answer,
        "sources": sources,
        "response_time": response_time,
    }
