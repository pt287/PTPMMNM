"""Test full RAG pipeline with OCR text."""
from pathlib import Path
from rag_engine import RAGConfig, build_rag_pipeline, ask_question

pdf = Path('c:/PTPMMNM/data/uploads/session_36/stamp_test.pdf')

# Build RAG with OCR enabled
cfg = RAGConfig(
    use_ocr=True,
    ocr_confidence_threshold=0.3,
    chunk_size=500,
    chunk_overlap=50,
    ollama_model='qwen2.5:1.5b'  # Use larger model
)

print("[*] Building RAG pipeline with OCR...")
qa, lang, chunk_count = build_rag_pipeline([(pdf.name, pdf.read_bytes())], cfg)
print(f"[OK] Created {chunk_count} chunks, language: {lang}")

print("\n[*] Testing question about stamp...")
answer, sources = ask_question(qa, 'Noi dung dau moc la gi?', doc_language=lang)
print(f"\n[ANSWER]:\n{answer}")
print(f"\n[SOURCES]: {len(sources)} documents")
for i, src in enumerate(sources[:2], 1):
    print(f"  [{i}] {src.page_content[:100]}")
