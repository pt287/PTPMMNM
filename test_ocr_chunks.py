"""Simple test: check if OCR text is in chunks (no LLM needed)."""
from pathlib import Path
from rag_engine import RAGConfig, load_documents_from_files
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf = Path('c:/PTPMMNM/data/uploads/session_36/stamp_test.pdf')

# Load with OCR
cfg = RAGConfig(use_ocr=True, ocr_confidence_threshold=0.3)
print("[*] Loading documents with OCR...")
docs = load_documents_from_files([(pdf.name, pdf.read_bytes())], cfg)
print(f"[OK] Loaded {len(docs)} documents")

# Show content
print("\n[=== DOCUMENT CONTENT ===]")
for i, doc in enumerate(docs, 1):
    print(f"\n[Doc {i}]:")
    print(doc.page_content[:300])
    print("...")

# Check if stamp text is present
all_text = " ".join(d.page_content for d in docs)
stamp_keywords = ["DOANH NGHIEP", "DOANH NGHIỆP", "TU NHAN", "TƯ NHÂN", "THIEN", "THIÊN"]
found = [kw for kw in stamp_keywords if kw.lower() in all_text.lower()]
print(f"\n[STAMP TEXT FOUND]: {found}")
if found:
    print("[SUCCESS] OCR text is in the documents!")
else:
    print("[FAIL] OCR text not found in documents")
