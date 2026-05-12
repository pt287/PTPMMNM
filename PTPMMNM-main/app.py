import time
from typing import List, Tuple

import streamlit as st

from rag_engine import RAGConfig, ask_question, build_rag_pipeline


st.set_page_config(page_title="SmartDoc AI", page_icon="📋", layout="wide")

st.markdown(
    """
<style>
    :root {
        --bg: #f4f7fb;
        --card: #ffffff;
        --text: #1e293b;
        --muted: #5b6b84;
        --line: #d9e2ef;
        --primary: #0f766e;
        --accent: #f59e0b;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, #dff7f2 0%, transparent 30%),
            radial-gradient(circle at 90% 10%, #fff2de 0%, transparent 26%),
            var(--bg);
    }

    .block-container {
        max-width: 1080px;
        padding-top: 1.5rem;
        padding-bottom: 2.2rem;
    }

    .hero {
        background: linear-gradient(120deg, #0f766e 0%, #0ea5a0 42%, #14b8a6 100%);
        border-radius: 18px;
        padding: 1.4rem 1.5rem;
        color: #ecfeff;
        box-shadow: 0 8px 25px rgba(15, 118, 110, 0.22);
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 1.65rem;
        letter-spacing: 0.2px;
    }

    .hero p {
        margin: 0.55rem 0 0 0;
        color: #d7fffb;
        font-size: 0.98rem;
    }

    .feature-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
        gap: 0.6rem;
        margin-top: 0.9rem;
    }

    .chip {
        background: rgba(255, 255, 255, 0.17);
        border: 1px solid rgba(255, 255, 255, 0.28);
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
        color: #f0fdfa;
        font-size: 0.84rem;
        text-align: center;
    }

    .section-card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.9rem 1rem 0.3rem;
        margin-bottom: 1rem;
    }

    .section-card h3 {
        margin: 0 0 0.4rem;
        color: var(--text);
        font-size: 1rem;
    }

    .section-card p {
        margin: 0 0 0.5rem;
        color: var(--muted);
        font-size: 0.93rem;
    }

    div[data-testid="stFileUploaderDropzone"] {
        border: 1.4px dashed #8ab9b2;
        border-radius: 12px;
        background: #f8fefe;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 0.5rem 0.7rem;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 12px;
    }

    .workflow {
        border-left: 4px solid var(--accent);
        padding: 0.5rem 0.75rem;
        background: #fffbeb;
        border-radius: 8px;
        color: #6b4a08;
        font-size: 0.9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<section class="hero">
    <h1>📋 SmartDoc AI - Intelligent Document Q&A System</h1>
    <p>Hệ thống RAG chạy local để hỏi đáp tài liệu PDF/DOCX bằng tiếng Việt và tiếng Anh, dùng Ollama + FAISS.</p>
    <div class="feature-row">
        <div class="chip">PDFPlumberLoader</div>
        <div class="chip">Chunk 1000 / Overlap 100</div>
        <div class="chip">FAISS Retrieval (top k=3)</div>
        <div class="chip">Qwen2.5:7b via Ollama</div>
    </div>
</section>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="section-card">
    <h3>Luồng xử lý</h3>
    <p class="workflow">Document: PDF → PDFPlumberLoader → TextSplitter → Embedding → FAISS Index</p>
    <p class="workflow">Query: Question → Vector Search (k=3) → Prompt + Context → LLM Answer</p>
</div>
""",
    unsafe_allow_html=True,
)

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "doc_language" not in st.session_state:
    st.session_state.doc_language = "unknown"
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.subheader("Cấu hình")
    ollama_model = st.text_input("LLM (Ollama)", value="qwen2.5:7b")
    embedding_model = st.text_input(
        "Embedding model",
        value="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    st.markdown("---")
    st.caption("Thông số theo đặc tả đồ án")
    st.caption("- chunk_size = 1000")
    st.caption("- chunk_overlap = 100")
    st.caption("- top_k = 3")
    st.caption("- loader = PDFPlumberLoader")

    st.markdown("---")
    st.caption("Đảm bảo Ollama đang chạy: ollama serve")

uploaded_files = st.file_uploader(
    "Tải lên một hoặc nhiều file PDF/DOCX",
    type=["pdf", "docx"],
    accept_multiple_files=True,
)

build_index = st.button("Build RAG Index", type="primary", use_container_width=True)

if build_index:
    if not uploaded_files:
        st.warning("Vui lòng tải lên ít nhất một file PDF hoặc DOCX.")
    else:
        pdf_items: List[Tuple[str, bytes]] = [(f.name, f.getvalue()) for f in uploaded_files]
        config = RAGConfig(
            embedding_model=embedding_model,
            ollama_model=ollama_model,
            chunk_size=1000,
            chunk_overlap=100,
            top_k=3,
            temperature=temperature,
        )

        with st.spinner("Đang xử lý tài liệu và xây dựng chỉ mục FAISS..."):
            try:
                qa_chain, doc_language, chunk_count = build_rag_pipeline(pdf_items, config)
                st.session_state.qa_chain = qa_chain
                st.session_state.doc_language = doc_language
                st.session_state.chunk_count = chunk_count
                st.session_state.chat_history = []

                st.success(
                    f"Xây dựng index thành công. Language: {doc_language}. Tổng chunk: {chunk_count}."
                )
            except Exception as exc:
                st.error("Không thể build RAG index. Vui lòng kiểm tra file PDF và dependencies.")
                st.exception(exc)

col1, col2 = st.columns(2)
with col1:
    st.metric("Ngôn ngữ tài liệu", st.session_state.doc_language)
with col2:
    st.metric("Số lượng chunks", st.session_state.chunk_count)

st.divider()
st.subheader("Hỏi đáp tài liệu")

for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

question = st.chat_input("Nhập câu hỏi về tài liệu PDF...")

if question:
    if st.session_state.qa_chain is None:
        st.warning("Bạn cần Build RAG Index trước khi đặt câu hỏi.")
    else:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Đang sinh câu trả lời..."):
                try:
                    start = time.time()
                    answer, sources = ask_question(st.session_state.qa_chain, question)
                    elapsed = time.time() - start

                    st.markdown(answer)
                    st.caption(f"Response time: {elapsed:.2f}s")

                    with st.expander("Nguồn ngữ cảnh được truy hồi"):
                        if not sources:
                            st.write("Không có source chunks.")
                        else:
                            for item in sources:
                                st.markdown(
                                    f"- Chunk {item['chunk_id']} ({item['source']}, p.{item.get('page', '?')}): {item['preview']}"
                                )

                    st.session_state.chat_history.append(("assistant", answer))
                except Exception as exc:
                    st.error(
                        "Không thể sinh câu trả lời. Hãy kiểm tra Ollama model và trạng thái runtime."
                    )
                    st.exception(exc)
