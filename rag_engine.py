"""RAG engine for SmartDoc AI using PDFPlumberLoader, FAISS, and Ollama."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langdetect import LangDetectException, detect


@dataclass
class RAGConfig:
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ollama_model: str = "qwen2.5:7b"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    top_k: int = 3
    temperature: float = 0.1


def _validate_config(config: RAGConfig) -> None:
    if config.chunk_size <= 0:
        raise ValueError("chunk_size phải lớn hơn 0.")
    if config.chunk_overlap < 0:
        raise ValueError("chunk_overlap không được âm.")
    if config.chunk_overlap >= config.chunk_size:
        raise ValueError("chunk_overlap phải nhỏ hơn chunk_size.")
    if config.top_k <= 0:
        raise ValueError("top_k phải lớn hơn 0.")


def load_documents_from_pdfs(pdf_items: Sequence[Tuple[str, bytes]]):
    """Load documents from PDF bytes using PDFPlumberLoader."""

    documents = []
    temp_paths: List[str] = []
    try:
        for file_name, file_bytes in pdf_items:
            suffix = os.path.splitext(file_name)[1] or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            temp_paths.append(tmp_path)
            loader = PDFPlumberLoader(tmp_path)
            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata = doc.metadata or {}
                doc.metadata["source"] = file_name

            documents.extend(loaded_docs)

        if not documents:
            raise ValueError("Không trích xuất được nội dung từ các file PDF đã tải lên.")

        return documents
    finally:
        for path in temp_paths:
            try:
                os.remove(path)
            except OSError:
                pass


def detect_main_language(text: str) -> str:
    sample = text[:5000].strip()
    if not sample:
        return "unknown"

    try:
        return detect(sample)
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


def build_vectorstore(chunks, config: RAGConfig) -> FAISS:
    if not chunks:
        raise ValueError("Không có chunks để tạo FAISS index.")

    embeddings = HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(chunks, embeddings)


def _build_prompt() -> PromptTemplate:
    template = """
Bạn là trợ lý AI chính xác và trung thực.
You are an accurate and honest AI assistant.

Yêu cầu:
1) Chỉ sử dụng thông tin trong phần Context để trả lời.
2) Nếu Context không đủ thông tin, hãy nói rõ là tài liệu không có dữ liệu cần thiết.
3) Trả lời theo đúng ngôn ngữ của câu hỏi người dùng (Việt hoặc Anh).
4) Trả lời ngắn gọn, đúng trọng tâm.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    return PromptTemplate(template=template, input_variables=["context", "question"])


def build_qa_chain(vectorstore: FAISS, config: RAGConfig) -> RetrievalQA:
    llm = Ollama(model=config.ollama_model, temperature=config.temperature)
    prompt = _build_prompt()

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": config.top_k}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def build_rag_pipeline(
    pdf_items: Sequence[Tuple[str, bytes]], config: RAGConfig
) -> Tuple[RetrievalQA, str, int]:
    _validate_config(config)
    documents = load_documents_from_pdfs(pdf_items)
    chunks = split_documents(documents, config)

    merged_text = "\n".join(chunk.page_content for chunk in chunks if chunk.page_content)
    language = detect_main_language(merged_text)

    vectorstore = build_vectorstore(chunks, config)
    qa_chain = build_qa_chain(vectorstore, config)
    return qa_chain, language, len(chunks)


def ask_question(qa_chain: RetrievalQA, question: str) -> Tuple[str, List[Dict[str, Any]]]:
    result = qa_chain.invoke({"query": question})
    answer = result.get("result", "")
    docs = result.get("source_documents", [])

    sources: List[Dict[str, Any]] = []
    for doc in docs:
        metadata = doc.metadata or {}
        sources.append(
            {
                "chunk_id": metadata.get("chunk_id", "?"),
                "source": metadata.get("source", "unknown"),
                "preview": (doc.page_content or "")[:220].replace("\n", " ").strip(),
            }
        )

    return answer, sources
