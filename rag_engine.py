"""RAG engine for SmartDoc AI using PDFPlumberLoader, FAISS, and Ollama."""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langdetect import LangDetectException, detect


@dataclass
class RAGConfig:
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ollama_model: str = "qwen2.5:7b"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    top_k: int = 3
    temperature: float = 0.1
    num_ctx: int = 1536
    num_predict: int = 220
    num_gpu: int = 0


def _validate_config(config: RAGConfig) -> None:
    if config.chunk_size <= 0:
        raise ValueError("chunk_size phải lớn hơn 0.")
    if config.chunk_overlap < 0:
        raise ValueError("chunk_overlap không được âm.")
    if config.chunk_overlap >= config.chunk_size:
        raise ValueError("chunk_overlap phải nhỏ hơn chunk_size.")
    if config.top_k <= 0:
        raise ValueError("top_k phải lớn hơn 0.")


def load_documents_from_files(file_items: Sequence[Tuple[str, bytes]]):
    """Load documents from supported document bytes (PDF, DOCX)."""

    documents = []
    temp_paths: List[str] = []
    try:
        for file_name, file_bytes in file_items:
            suffix = os.path.splitext(file_name)[1] or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            temp_paths.append(tmp_path)
            ext = suffix.lower()
            if ext == ".pdf":
                loader = PDFPlumberLoader(tmp_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(tmp_path)
            else:
                raise ValueError(f"Định dạng file chưa được hỗ trợ: {file_name}")

            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata = doc.metadata or {}
                doc.metadata["source"] = file_name

            documents.extend(loaded_docs)

        if not documents:
            raise ValueError("Không trích xuất được nội dung từ các file tải lên.")

        return documents
    finally:
        for path in temp_paths:
            try:
                os.remove(path)
            except OSError:
                pass


def detect_main_language(documents: Sequence[Any]) -> str:
    collected: List[str] = []
    total_chars = 0
    for doc in documents:
        content = (getattr(doc, "page_content", "") or "").strip()
        if not content:
            continue

        remaining = 5000 - total_chars
        if remaining <= 0:
            break

        piece = content[:remaining]
        collected.append(piece)
        total_chars += len(piece)

    sample = "\n".join(collected).strip()
    if not sample:
        return "unknown"

    try:
        return detect(sample)
    except LangDetectException:
        return "unknown"


def detect_question_language(question: str) -> str:
    text = (question or "").strip()
    if not text:
        return "unknown"

    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def split_documents(documents, config: RAGConfig):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks, start=1):
        chunk.metadata = chunk.metadata or {}
        chunk.metadata["chunk_id"] = idx
    return chunks


@lru_cache(maxsize=4)
def _get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    # Cache embedding model instance to avoid cold-loading on every rebuild.
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks, config: RAGConfig) -> FAISS:
    if not chunks:
        raise ValueError("Không có chunks để tạo FAISS index.")

    embeddings = _get_embeddings(config.embedding_model)
    return FAISS.from_documents(chunks, embeddings)


def _build_prompt() -> PromptTemplate:
    template = """
Bạn là trợ lý AI chính xác và trung thực.
You are an accurate and honest AI assistant.

Yêu cầu:
1) Chỉ sử dụng thông tin trong phần Context để trả lời.
2) Nếu Context không đủ thông tin, chỉ được trả lời đúng một câu fallback đã nêu trong câu hỏi.
3) Trả lời theo đúng ngôn ngữ của câu hỏi người dùng (Việt hoặc Anh).
4) Trả lời ngắn gọn, đúng trọng tâm (3-4 câu).
5) Không tự suy diễn, không bịa thêm thông tin ngoài Context.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    return PromptTemplate(template=template, input_variables=["context", "question"])


def build_qa_chain(vectorstore: FAISS, config: RAGConfig) -> RetrievalQA:
    llm = OllamaLLM(
        model=config.ollama_model,
        temperature=config.temperature,
        num_ctx=config.num_ctx,
        num_predict=config.num_predict,
        num_gpu=config.num_gpu,
        keep_alive="30m",
    )
    prompt = _build_prompt()

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": config.top_k}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def build_rag_pipeline(
    file_items: Sequence[Tuple[str, bytes]], config: RAGConfig
) -> Tuple[RetrievalQA, str, int]:
    _validate_config(config)
    documents = load_documents_from_files(file_items)
    language = detect_main_language(documents)
    chunks = split_documents(documents, config)

    vectorstore = build_vectorstore(chunks, config)
    qa_chain = build_qa_chain(vectorstore, config)
    return qa_chain, language, len(chunks)


def _format_recent_history(conversation_history: Optional[Sequence[Tuple[str, str]]]) -> str:
    if not conversation_history:
        return "(none)"

    recent = list(conversation_history)[-6:]
    lines: List[str] = []
    for role, content in recent:
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def _unknown_phrase_for_language(lang_code: str) -> str:
    return "Tôi không biết" if lang_code == "vi" else "I don't know"


def _normalize_unknown_answer(answer: str, lang_code: str) -> str:
    fallback = _unknown_phrase_for_language(lang_code)
    cleaned = (answer or "").strip()
    if not cleaned:
        return fallback

    lowered = cleaned.lower()
    unknown_patterns = [
        "khong du thong tin",
        "không đủ thông tin",
        "tai lieu khong co",
        "tài liệu không có",
        "not enough information",
        "does not contain",
        "cannot find in context",
        "insufficient context",
    ]

    if any(pattern in lowered for pattern in unknown_patterns):
        return fallback

    return cleaned


def ask_question(
    qa_chain: RetrievalQA,
    question: str,
    conversation_history: Optional[Sequence[Tuple[str, str]]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    question_lang = detect_question_language(question)
    response_language = "Vietnamese" if question_lang == "vi" else "English"
    unknown_phrase = _unknown_phrase_for_language(question_lang)
    chat_history = _format_recent_history(conversation_history)
    augmented_question = (
        f"User question: {question}\n"
        f"Response language: {response_language}\n"
        f"Fallback when missing context: {unknown_phrase}\n"
        f"Recent conversation:\n{chat_history}"
    )

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            result = qa_chain.invoke({"query": augmented_question})
            break
        except Exception as exc:
            last_error = exc
            # Ollama runner can terminate transiently; retry once.
            if "llama runner process has terminated" not in str(exc) or attempt == 1:
                raise
            time.sleep(0.25)
    else:
        raise RuntimeError("Không thể sinh câu trả lời từ Ollama.") from last_error

    answer = _normalize_unknown_answer(result.get("result", ""), question_lang)
    docs = result.get("source_documents", [])

    sources: List[Dict[str, Any]] = []
    for doc in docs:
        metadata = doc.metadata or {}
        sources.append(
            {
                "chunk_id": metadata.get("chunk_id", "?"),
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page", metadata.get("page_number", "?")),
                "preview": (doc.page_content or "")[:220].replace("\n", " ").strip(),
            }
        )

    return answer, sources
