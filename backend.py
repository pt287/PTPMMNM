import time
import json
import sqlite3
import shutil
import unicodedata
from threading import Lock
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_engine import (
    RAGConfig,
    ask_question,
    ask_question_with_self_rag,
    build_rag_pipeline,
    compare_retrieval_methods,
)


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
        self.chunk_size = 1500
        self.chunk_overlap = 100
        self.current_session_id: Optional[int] = None
        self.conversation_history: List[tuple[str, str]] = []
        self.use_reranking = False
        self.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.rerank_top_n = 10

        # Self-RAG state
        self.use_self_rag = False
        self.enable_query_rewriting = False
        self.enable_multi_hop = False
        self.max_hops = 3
        self.confidence_threshold = 0.7
        self.vectorstore = None  # Store vectorstore for Self-RAG


state = AppState()


def _get_connection() -> sqlite3.Connection:
    db_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
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
                    chunk_size INTEGER NOT NULL DEFAULT 1500,
                    chunk_overlap INTEGER NOT NULL DEFAULT 100,
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

            session_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(index_sessions)").fetchall()
            }
            if "chunk_size" not in session_columns:
                conn.execute("ALTER TABLE index_sessions ADD COLUMN chunk_size INTEGER NOT NULL DEFAULT 1500")
            if "chunk_overlap" not in session_columns:
                conn.execute("ALTER TABLE index_sessions ADD COLUMN chunk_overlap INTEGER NOT NULL DEFAULT 100")
            if "use_reranking" not in session_columns:
                conn.execute("ALTER TABLE index_sessions ADD COLUMN use_reranking INTEGER NOT NULL DEFAULT 0")
            if "reranker_model" not in session_columns:
                conn.execute("ALTER TABLE index_sessions ADD COLUMN reranker_model TEXT")
            if "rerank_top_n" not in session_columns:
                conn.execute("ALTER TABLE index_sessions ADD COLUMN rerank_top_n INTEGER NOT NULL DEFAULT 10")
            # Self-RAG columns
            if "use_self_rag" not in session_columns:
                conn.execute("ALTER TABLE index_sessions ADD COLUMN use_self_rag INTEGER NOT NULL DEFAULT 0")
            if "enable_query_rewriting" not in session_columns:
                conn.execute("ALTER TABLE index_sessions ADD COLUMN enable_query_rewriting INTEGER NOT NULL DEFAULT 0")
            if "enable_multi_hop" not in session_columns:
                conn.execute("ALTER TABLE index_sessions ADD COLUMN enable_multi_hop INTEGER NOT NULL DEFAULT 0")
            if "max_hops" not in session_columns:
                conn.execute("ALTER TABLE index_sessions ADD COLUMN max_hops INTEGER NOT NULL DEFAULT 3")
            if "confidence_threshold" not in session_columns:
                conn.execute("ALTER TABLE index_sessions ADD COLUMN confidence_threshold REAL NOT NULL DEFAULT 0.7")
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
    chunk_size: int,
    chunk_overlap: int,
    ollama_model: str,
    embedding_model: str,
    temperature: float,
    use_reranking: bool,
    reranker_model: str,
    rerank_top_n: int,
    use_self_rag: bool,
    enable_query_rewriting: bool,
    enable_multi_hop: bool,
    max_hops: int,
    confidence_threshold: float,
    files: List[Dict[str, Any]],
) -> int:
    with db_lock:
        with _get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO index_sessions (
                    doc_language, chunk_count, chunk_size, chunk_overlap, ollama_model, embedding_model, temperature,
                    use_reranking, reranker_model, rerank_top_n,
                    use_self_rag, enable_query_rewriting, enable_multi_hop, max_hops, confidence_threshold
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_language,
                    chunk_count,
                    chunk_size,
                    chunk_overlap,
                    ollama_model,
                    embedding_model,
                    temperature,
                    1 if use_reranking else 0,
                    reranker_model if use_reranking else None,
                    rerank_top_n,
                    1 if use_self_rag else 0,
                    1 if enable_query_rewriting else 0,
                    1 if enable_multi_hop else 0,
                    max_hops,
                    confidence_threshold,
                ),
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
                SELECT id, created_at, doc_language, chunk_count, chunk_size, chunk_overlap, ollama_model, embedding_model, temperature,
                       use_reranking, reranker_model, rerank_top_n,
                       use_self_rag, enable_query_rewriting, enable_multi_hop, max_hops, confidence_threshold
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
                        "chunk_size": row["chunk_size"],
                        "chunk_overlap": row["chunk_overlap"],
                        "ollama_model": row["ollama_model"],
                        "embedding_model": row["embedding_model"],
                        "temperature": row["temperature"],
                        "use_reranking": bool(row["use_reranking"]),
                        "reranker_model": row["reranker_model"],
                        "rerank_top_n": row["rerank_top_n"],
                        "use_self_rag": bool(row["use_self_rag"]),
                        "enable_query_rewriting": bool(row["enable_query_rewriting"]),
                        "enable_multi_hop": bool(row["enable_multi_hop"]),
                        "max_hops": row["max_hops"],
                        "confidence_threshold": row["confidence_threshold"],
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


def _resolve_session_file_path(session_id: int, file_name: str) -> Optional[Path]:
    normalized_target = unicodedata.normalize("NFC", Path(file_name).name).casefold()

    with db_lock:
        with _get_connection() as conn:
            row = conn.execute(
                """
                SELECT stored_path
                FROM session_pdfs
                WHERE session_id = ? AND file_name = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_id, file_name),
            ).fetchone()

            if row is None:
                candidate_rows = conn.execute(
                    """
                    SELECT file_name, stored_path
                    FROM session_pdfs
                    WHERE session_id = ?
                    ORDER BY id DESC
                    """,
                    (session_id,),
                ).fetchall()
                for candidate in candidate_rows:
                    candidate_name = unicodedata.normalize(
                        "NFC", Path(candidate["file_name"]).name
                    ).casefold()
                    if candidate_name == normalized_target:
                        row = candidate
                        break

    if row is None:
        return None

    stored_path = row["stored_path"]
    if not stored_path:
        return None

    path = Path(stored_path)
    if not path.exists():
        return None

    return path


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
                SELECT id, created_at, doc_language, chunk_count, chunk_size, chunk_overlap, ollama_model, embedding_model, temperature,
                       use_reranking, reranker_model, rerank_top_n,
                       use_self_rag, enable_query_rewriting, enable_multi_hop, max_hops, confidence_threshold
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
        "chunk_size": row["chunk_size"],
        "chunk_overlap": row["chunk_overlap"],
        "ollama_model": row["ollama_model"],
        "embedding_model": row["embedding_model"],
        "temperature": row["temperature"],
        "use_reranking": bool(row["use_reranking"]),
        "reranker_model": row["reranker_model"],
        "rerank_top_n": row["rerank_top_n"],
        "use_self_rag": bool(row["use_self_rag"]),
        "enable_query_rewriting": bool(row["enable_query_rewriting"]),
        "enable_multi_hop": bool(row["enable_multi_hop"]),
        "max_hops": row["max_hops"],
        "confidence_threshold": row["confidence_threshold"],
    }


def _clear_chat_history(session_id: Optional[int]) -> int:
    if session_id is None:
        return 0

    with db_lock:
        with _get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM qa_history WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
    return max(0, cursor.rowcount)


def _clear_vector_store_data() -> Dict[str, int]:
    with db_lock:
        with _get_connection() as conn:
            # Keep chat rows but detach them from removed upload/index sessions.
            conn.execute("UPDATE qa_history SET session_id = NULL WHERE session_id IS NOT NULL")
            file_cursor = conn.execute("DELETE FROM session_pdfs")
            session_cursor = conn.execute("DELETE FROM index_sessions")
            conn.commit()

    removed_files = max(0, file_cursor.rowcount)
    removed_sessions = max(0, session_cursor.rowcount)

    if uploads_dir.exists():
        shutil.rmtree(uploads_dir, ignore_errors=True)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    return {
        "removed_files": removed_files,
        "removed_sessions": removed_sessions,
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
        "chunk_size": state.chunk_size,
        "chunk_overlap": state.chunk_overlap,
        "use_reranking": state.use_reranking,
        "reranker_model": state.reranker_model,
        "rerank_top_n": state.rerank_top_n,
        "use_self_rag": state.use_self_rag,
        "enable_query_rewriting": state.enable_query_rewriting,
        "enable_multi_hop": state.enable_multi_hop,
        "max_hops": state.max_hops,
        "confidence_threshold": state.confidence_threshold,
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
            "chunk_size": session["chunk_size"],
            "chunk_overlap": session["chunk_overlap"],
            "use_reranking": session["use_reranking"],
            "reranker_model": session["reranker_model"],
            "rerank_top_n": session["rerank_top_n"],
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
        chunk_size=session["chunk_size"],
        chunk_overlap=session["chunk_overlap"],
        top_k=3,
        temperature=session["temperature"],
        use_reranking=session["use_reranking"],
        reranker_model=session["reranker_model"] or "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_top_n=session["rerank_top_n"],
        use_self_rag=session["use_self_rag"],
        enable_query_rewriting=session["enable_query_rewriting"],
        enable_multi_hop=session["enable_multi_hop"],
        max_hops=session["max_hops"],
        confidence_threshold=session["confidence_threshold"],
    )

    try:
        qa_chain, doc_language, chunk_count = build_rag_pipeline(pdf_items, config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    state.qa_chain = qa_chain
    state.doc_language = doc_language
    state.chunk_count = chunk_count
    state.chunk_size = session["chunk_size"]
    state.chunk_overlap = session["chunk_overlap"]
    state.use_reranking = session["use_reranking"]
    state.reranker_model = session["reranker_model"] or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    state.rerank_top_n = session["rerank_top_n"]
    state.use_self_rag = session["use_self_rag"]
    state.enable_query_rewriting = session["enable_query_rewriting"]
    state.enable_multi_hop = session["enable_multi_hop"]
    state.max_hops = session["max_hops"]
    state.confidence_threshold = session["confidence_threshold"]
    # Store vectorstore for Self-RAG
    if hasattr(qa_chain.retriever, 'vectorstore'):
        state.vectorstore = qa_chain.retriever.vectorstore
    state.current_session_id = session_id
    state.conversation_history = []

    return {
        "message": "Session activated.",
        "session_id": session_id,
        "doc_language": doc_language,
        "chunk_count": chunk_count,
        "chunk_size": session["chunk_size"],
        "chunk_overlap": session["chunk_overlap"],
        "use_reranking": session["use_reranking"],
        "reranker_model": session["reranker_model"],
        "rerank_top_n": session["rerank_top_n"],
    }


@app.get("/api/sessions/{session_id}/file")
def get_session_file(session_id: int, file_name: str) -> FileResponse:
    path = _resolve_session_file_path(session_id, file_name)
    if path is None:
        raise HTTPException(status_code=404, detail="File not found in this session.")

    media_type = "application/pdf" if path.suffix.lower() == ".pdf" else None
    return FileResponse(
        path,
        media_type=media_type,
        filename=path.name,
        content_disposition_type="inline",
        headers={
            "Cache-Control": "no-store",
        },
    )


@app.post("/api/build-index")
async def build_index(
    files: List[UploadFile] = File(...),
    ollama_model: str = Form("qwen2.5:0.5b"),
    embedding_model: str = Form("sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
    chunk_size: int = Form(1500),
    chunk_overlap: int = Form(100),
    temperature: float = Form(0.1),
    use_reranking: bool = Form(False),
    reranker_model: str = Form("cross-encoder/ms-marco-MiniLM-L-6-v2"),
    rerank_top_n: int = Form(10),
    # Self-RAG parameters
    use_self_rag: bool = Form(False),
    enable_query_rewriting: bool = Form(False),
    enable_multi_hop: bool = Form(False),
    max_hops: int = Form(3),
    confidence_threshold: float = Form(0.7),
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

    if chunk_size <= 0:
        raise HTTPException(status_code=400, detail="chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise HTTPException(status_code=400, detail="chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap must be smaller than chunk_size.")

    config = RAGConfig(
        embedding_model=embedding_model,
        ollama_model=ollama_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=3,
        temperature=temperature,
        use_reranking=use_reranking,
        reranker_model=reranker_model,
        rerank_top_n=rerank_top_n,
        # Self-RAG
        use_self_rag=use_self_rag,
        enable_query_rewriting=enable_query_rewriting,
        enable_multi_hop=enable_multi_hop,
        max_hops=max_hops,
        confidence_threshold=confidence_threshold,
    )

    try:
        qa_chain, doc_language, chunk_count = build_rag_pipeline(doc_items, config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    state.qa_chain = qa_chain
    state.doc_language = doc_language
    state.chunk_count = chunk_count
    state.chunk_size = chunk_size
    state.chunk_overlap = chunk_overlap
    state.use_reranking = use_reranking
    state.reranker_model = reranker_model
    state.rerank_top_n = rerank_top_n
    # Self-RAG state
    state.use_self_rag = use_self_rag
    state.enable_query_rewriting = enable_query_rewriting
    state.enable_multi_hop = enable_multi_hop
    state.max_hops = max_hops
    state.confidence_threshold = confidence_threshold
    # Store vectorstore for Self-RAG
    if hasattr(qa_chain.retriever, 'vectorstore'):
        state.vectorstore = qa_chain.retriever.vectorstore
    state.conversation_history = []
    state.current_session_id = _store_index_session(
        doc_language=doc_language,
        chunk_count=chunk_count,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ollama_model=ollama_model,
        embedding_model=embedding_model,
        temperature=temperature,
        use_reranking=use_reranking,
        reranker_model=reranker_model,
        rerank_top_n=rerank_top_n,
        use_self_rag=use_self_rag,
        enable_query_rewriting=enable_query_rewriting,
        enable_multi_hop=enable_multi_hop,
        max_hops=max_hops,
        confidence_threshold=confidence_threshold,
        files=file_records,
    )

    return {
        "message": "RAG index built successfully.",
        "doc_language": doc_language,
        "chunk_count": chunk_count,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "use_reranking": use_reranking,
        "reranker_model": reranker_model if use_reranking else None,
        "rerank_top_n": rerank_top_n if use_reranking else None,
        "use_self_rag": use_self_rag,
        "enable_query_rewriting": enable_query_rewriting if use_self_rag else None,
        "enable_multi_hop": enable_multi_hop if use_self_rag else None,
        "max_hops": max_hops if use_self_rag else None,
        "confidence_threshold": confidence_threshold if use_self_rag else None,
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
        # Use Self-RAG if enabled
        if state.use_self_rag and state.vectorstore is not None:
            # Create config for Self-RAG
            self_rag_config = RAGConfig(
                use_self_rag=state.use_self_rag,
                enable_query_rewriting=state.enable_query_rewriting,
                enable_multi_hop=state.enable_multi_hop,
                max_hops=state.max_hops,
                confidence_threshold=state.confidence_threshold,
                top_k=3,
                use_reranking=state.use_reranking,
                reranker_model=state.reranker_model,
                rerank_top_n=state.rerank_top_n,
            )

            answer, sources, metadata = ask_question_with_self_rag(
                state.qa_chain,
                question,
                state.vectorstore,
                self_rag_config,
                conversation_history=state.conversation_history,
            )
        else:
            # Standard RAG
            answer, sources = ask_question(
                state.qa_chain,
                question,
                conversation_history=state.conversation_history,
            )
            metadata = None
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

    response = {
        "answer": answer,
        "sources": sources,
        "response_time": response_time,
    }

    # Add Self-RAG metadata if available
    if state.use_self_rag and metadata is not None:
        response["self_rag_metadata"] = metadata

    return response


@app.post("/api/compare-retrieval")
def compare_retrieval(payload: AskPayload) -> dict:
    """Compare bi-encoder vs cross-encoder retrieval methods."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if state.qa_chain is None:
        raise HTTPException(status_code=400, detail="Build RAG index before comparing retrieval methods.")

    # Get vectorstore from retriever
    if hasattr(state.qa_chain.retriever, 'vectorstore'):
        vectorstore = state.qa_chain.retriever.vectorstore
    else:
        raise HTTPException(
            status_code=500,
            detail="Cannot access vectorstore from current QA chain configuration."
        )

    # Create config for comparison
    config = RAGConfig(
        top_k=3,
        use_reranking=True,
        reranker_model=state.reranker_model,
        rerank_top_n=state.rerank_top_n,
    )

    try:
        comparison = compare_retrieval_methods(vectorstore, question, config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Format documents for response
    def format_docs(docs):
        result = []
        for doc in docs:
            metadata = doc.metadata or {}
            result.append({
                "chunk_id": metadata.get("chunk_id", "?"),
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page", "?"),
                "preview": doc.page_content[:200].replace("\n", " ") + "...",
            })
        return result

    return {
        "query": question,
        "bi_encoder": {
            "time_ms": comparison["bi_encoder"]["time_ms"],
            "num_docs": comparison["bi_encoder"]["num_docs"],
            "documents": format_docs(comparison["bi_encoder"]["docs"]),
        },
        "cross_encoder": {
            "time_ms": comparison["cross_encoder"]["time_ms"],
            "retrieval_time_ms": comparison["cross_encoder"]["retrieval_time_ms"],
            "rerank_time_ms": comparison["cross_encoder"]["rerank_time_ms"],
            "num_candidates": comparison["cross_encoder"]["num_candidates"],
            "num_final": comparison["cross_encoder"]["num_final"],
            "documents": format_docs(comparison["cross_encoder"]["docs"]),
            "scores": comparison["cross_encoder"]["scores"],
        },
        "comparison": comparison["comparison"],
    }


@app.delete("/api/history")
def clear_history(session_id: Optional[int] = None) -> dict:
    target_session_id = session_id if session_id is not None else state.current_session_id
    if target_session_id is None:
        raise HTTPException(
            status_code=400,
            detail="No active session found. Hãy build index hoặc chọn session trước khi xóa lịch sử.",
        )

    deleted_rows = _clear_chat_history(target_session_id)
    state.conversation_history = []
    return {
        "message": "Current session chat history cleared.",
        "session_id": target_session_id,
        "deleted_rows": deleted_rows,
    }


@app.delete("/api/vector-store")
def clear_vector_store() -> dict:
    result = _clear_vector_store_data()

    state.qa_chain = None
    state.doc_language = "unknown"
    state.chunk_count = 0
    state.chunk_size = 1500
    state.chunk_overlap = 100
    state.current_session_id = None
    state.conversation_history = []

    return {
        "message": "Vector store and uploaded documents cleared.",
        "removed_sessions": result["removed_sessions"],
        "removed_files": result["removed_files"],
    }
