# SmartDoc AI (Local Document Q&A)

Ung dung hoi dap tai lieu PDF/DOCX theo kien truc RAG, ho tro tieng Viet va tieng Anh.

## Cai Dat Day Du (Windows)

### 1) Cai Python

- Cai Python 3.10+ (khuyen nghi 3.10/3.11)
- Khi cai, tick "Add Python to PATH"
- Kiem tra:

```bash
python --version
pip --version
```

### 2) Cai Ollama

Ollama la ung dung he thong, KHONG nam trong requirements.txt.

Cach 1 (nhanh nhat, dung winget):

```bash
winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
```

Cach 2 (thu cong):

- Tai installer tu trang Ollama va cai dat
- Mo lai terminal sau khi cai

Kiem tra da cai thanh cong:

```bash
ollama --version
```

Neu lenh "ollama" chua nhan dien:

- Dong va mo lai VS Code/terminal
- Kiem tra file co ton tai tai:
	- C:\Users\<user>\AppData\Local\Programs\Ollama\ollama.exe
	- C:\Program Files\Ollama\ollama.exe

## Chay Nhanh

### 1) Tao va kich hoat moi truong ao (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2) Cai thu vien

```bash
pip install -r requirements.txt
```

### 3) Chuan bi Ollama

Luu y: Ollama khong nam trong requirements.txt vi day la ung dung he thong (khong phai thu vien Python).

```bash
ollama pull qwen2.5:7b
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:0.5b
ollama serve
```

Neu may RAM thap, uu tien chon model nhe (qwen2.5:0.5b hoac qwen2.5:1.5b) trong form Build RAG Index.

### 4) Chay web app (frontend + backend)

```bash
uvicorn backend:app --reload
```

Sau do mo trinh duyet: http://127.0.0.1:8000

## Cach Dung

1. Upload 1 hoac nhieu file PDF.
2. Bam Build RAG Index.
3. Dat cau hoi trong khung chat.
4. Nhan cau tra loi va xem chunk + trang duoc truy hoi (neu co).

## Cau Hinh Chinh

- LLM: qwen2.5:7b (Ollama)
- Embedding: paraphrase-multilingual-mpnet-base-v2
- Loader: PDFPlumberLoader + python-docx (DOCX)
- Chunk: mac dinh 1500 (co the tuy chinh tren UI)
- Overlap: 100
- Retrieval: top k = 3
- Vector DB: FAISS
- **Hybrid Search**: FAISS + BM25 (EnsembleRetriever, weight 0.7/0.3)
- **GraphRAG**: mo rong context bang do thi lien ket chunk
- **Re-ranking**: Cross-Encoder (tuy chon, tang do chinh xac retrieval)
- **Self-RAG**: query rewriting + multi-hop + tu danh gia do tin cay
- **Metadata Filtering**: loc theo `source` va `doc_id`

## Cong Nghe Moi Da Them

### 1) GraphRAG Retrieval Expansion

- Build do thi lien ket giua cac chunk dua tren token overlap.
- Sau khi retrieval ban dau, he thong mo rong them cac chunk lien quan theo so hop cau hinh.
- Co the bat/tat tren UI bang switch GraphRAG.

Gia tri:

- Tang recall khi thong tin nam rai rac o nhieu phan trong tai lieu.
- Huu ich cho cau hoi tong hop, cau hoi can nhieu context lien thong.

### 2) Hybrid Search (Semantic + Keyword)

- Ket hop semantic retrieval (FAISS) va keyword retrieval (BM25).
- Su dung EnsembleRetriever voi trong so mac dinh:
	- Semantic: 0.7
	- Keyword: 0.3

Gia tri:

- Giam bo sot keyword quan trong.
- Can bang giua matching ngu nghia va matching tu khoa.

### 3) Self-RAG Pipeline

- Query Rewriting: viet lai cau hoi de retrieval ro y hon.
- Multi-hop Retrieval: truy hoi lap nhieu vong cho cau hoi phuc tap.
- Answer Evaluation: tu cham diem grounding/relevance/completeness.

Gia tri:

- Tang do on dinh chat luong tra loi.
- Co confidence metadata de canh bao khi do tin cay thap.

### 4) Multi-document + Metadata

- Ho tro upload nhieu file PDF/DOCX trong cung session.
- Moi chunk duoc gan metadata:
	- `source`
	- `doc_id`
	- `upload_time`
	- `file_type`
- API hoi dap ho tro `filter_metadata` de loc theo tai lieu cu the.

### 5) Session-aware Config Restore

- Moi lan build index, cau hinh retrieval duoc luu theo session.
- Khi mo lai session cu, cac switch va tham so duoc khoi phuc:
	- GraphRAG
	- Re-ranking + model + Top-N
	- Self-RAG + query rewriting + multi-hop + threshold
- Giup tiep tuc lam viec tren session cu ma khong phai cau hinh lai tu dau.

## Cross-Encoder Re-ranking (Moi!)

He thong da tich hop Cross-Encoder re-ranking de cai thien chat luong retrieval.

### Cach hoat dong

1. **Bi-encoder** (FAISS): Retrieval nhanh, lay nhieu candidates (~10 docs)
2. **Cross-encoder**: Re-rank chinh xac hon, chon top-3 docs tot nhat
3. Ket qua: Chat luong cao hon +25-30%, them ~100ms latency

### Su dung

Khi Build RAG Index, chon:
- **Use Re-ranking**: Bat/tat tinh nang
- **Re-ranker Model**: Model su dung (mac dinh: `ms-marco-MiniLM-L-6-v2`)
- **Re-rank Top N**: So candidates lay ve (mac dinh: 10)

### Khuyen nghi model

- **Tieng Anh**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (nhanh)
- **Tieng Viet**: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (da ngon ngu)
- **Chat luong cao**: `cross-encoder/ms-marco-MiniLM-L-12-v2`

### So sanh performance

```bash
# Chay demo de xem so sanh
python demo_reranking.py

# Test API endpoint
curl -X POST http://localhost:8000/api/compare-retrieval \
  -H "Content-Type: application/json" \
  -d '{"question": "Cau hoi test"}'
```

**Chi tiet day du**: Xem file [RERANKING_GUIDE.md](RERANKING_GUIDE.md)

## OCR - Nhan Dien Chu Trong Anh (Moi!)

He thong da tich hop **Optical Character Recognition (OCR)** de doc chu tu trong anh trong PDF/DOCX va nhan dien chu tren dau moc, watermark, seal.

### Tinh nang chinh

- 📸 **Trích xuất ảnh từ PDF/DOCX** - Tự động lấy tất cả ảnh từ tài liệu
- 🔤 **Nhận diện chữ (OCR)** - Sử dụng RapidOCR, nhẹ và chạy tốt trên Windows
- 🇻🇳 **Hỗ trợ Tiếng Việt** - Tối ưu cho tiếng Việt, Anh, Trung
- 🖼️ **Cải thiện chất lượng ảnh** - Tự động denoise, normalize trước OCR
- 🏷️ **Đọc chữ trên dấu mộc** - Nhận diện text từ watermark, seal, stamp
- ⚙️ **GPU Support** - Tăng tốc nếu có NVIDIA GPU + CUDA

### Cách cài đặt

OCR được bao gồm trong `requirements.txt`. Nếu cài thủ công từng phần, chỉ cần:

```bash
pip install rapidocr-onnxruntime opencv-python
```

Lưu ý: project dùng `pypdfium2` để render PDF nên không cần cài Poppler như cách dùng `pdf2image` trên Windows.

### Cách sử dụng

#### Khi Build RAG Index:

Trong form **Build RAG Index**, có các option:
- **Enable OCR** - Bật/tắt OCR (mặc định: ON)
- **OCR Confidence Threshold** - Độ tin cậy tối thiểu (0.0-1.0, mặc định: 0.3)
- **OCR GPU** - Sử dụng GPU (mặc định: OFF, yêu cầu NVIDIA GPU + CUDA)
- **Extract Images Only** - Chỉ lấy text từ ảnh, bỏ qua text thường (mặc định: OFF)

#### API Call:

```bash
curl -X POST http://localhost:8000/api/build-index \
  -F "files=@scanned_document.pdf" \
  -F "use_ocr=true" \
  -F "ocr_confidence_threshold=0.3" \
  -F "ocr_gpu=false"
```

### Ngôn ngữ hỗ trợ

- 🇻🇳 **Tiếng Việt** (vi) - Mặc định và tối ưu tốt
- 🇬🇧 **English** (en) - Mặc định và độ chính xác cao
- **Các ngôn ngữ khác** - Có thể cấu hình thêm cho từng bộ tài liệu nếu cần

Mặc định project dùng `['vi', 'en']` để OCR ổn định trên PDF/DOCX và dấu mộc. RapidOCR chạy nhẹ hơn và phù hợp hơn cho môi trường Windows không có Poppler/Tesseract.

### Performance

| Phương pháp | Tốc độ | Chất lượng |
|------------|-------|-----------|
| CPU | 2-5s/ảnh | Tốt |
| GPU (CUDA) | 0.5-1s/ảnh | Tốt |

### Mẹo tối ưu OCR

1. **Chất lượng ảnh**
   - Sử dụng ảnh độ phân giải cao (300+ DPI cho scan)
   - Tránh bóng, chói sáng quá mức
   - Đảm bảo độ tương phản tốt

2. **Kích thước chữ**
   - Tối thiểu: 12px
   - Tối ưu: 24px trở lên
   - Ảnh hưởng lớn đến độ chính xác

3. **Confidence Threshold**
   - Thấp (0.1-0.2): Nhận nhiều text nhưng có thể sai
   - Trung bình (0.3-0.5): Cân bằng tốt (khuyến nghị)
   - Cao (0.7+): Chỉ text rất rõ ràng

4. **GPU**
   - Nếu có NVIDIA GPU + CUDA: Bật GPU để tăng tốc 5-10x
   - Nếu không: Để OFF, sử dụng CPU

### Tính hỗ trợ định dạng

| Format | Ảnh trong tài liệu | Text thường | Trạng thái |
|--------|------------------|------------|-----------|
| PDF | ✅ OCR | ✅ Extract | Hỗ trợ đầy đủ |
| DOCX | ✅ OCR | ✅ Extract | Hỗ trợ đầy đủ |

### Ví dụ sử dụng

#### Python:
```python
from rag_engine import RAGConfig, load_documents_from_files

config = RAGConfig(
    use_ocr=True,
    ocr_languages=['vi', 'en'],
    ocr_confidence_threshold=0.4,
    ocr_gpu=False
)

with open('scan.pdf', 'rb') as f:
    documents = load_documents_from_files(
        file_items=[('scan.pdf', f.read())],
        config=config
    )

print(f"Extracted {len(documents)} documents with OCR")
```

#### Chỉ lấy text từ ảnh (bỏ text thường):
```python
config = RAGConfig(
    use_ocr=True,
    extract_images_only=True,  # Chỉ OCR
    ocr_confidence_threshold=0.5
)
```

## Tuy Chinh Chunk Parameters

- Trong form Build RAG Index, co the chon cac gia tri sau.
- Chunk size: `500 | 1000 | 1500 | 2000`
- Chunk overlap: `50 | 100 | 200`
- He thong luu theo tung session, khi mo lai session se dung dung cau hinh chunk da build truoc do.

## Benchmark Chunk Strategy

Da them script benchmark de so sanh 12 cau hinh chunk/overlap va bao cao retrieval accuracy.

Chay benchmark:

```bash
python documentation/chunk_strategy_benchmark.py --uploads-dir data/uploads --output documentation/chunk_strategy_report.md
```

Ket qua report se nam o:

- `documentation/chunk_strategy_report.md`

Ket qua gan nhat (tren du lieu hien tai):

- Top-3 tot nhat: `chunk_size=1500`, `chunk_overlap=100` (61.90%)
- Top-1 cao nhat: `chunk_size=1000`, `chunk_overlap=200` (40.00%)

Khuyen nghi:

- Mac dinh su dung `1500/100` de uu tien do bao phu retrieval (Top-3).

## Loi Thuong Gap

### Khong cai duoc faiss-cpu==1.9.0

Ban dang dung Python moi (3.12+). Du an da tu dong dung faiss-cpu >= 1.12.0 trong requirements.

### App khong tra loi

- Kiem tra Ollama dang chay: ollama serve
- Kiem tra model da tai: ollama pull qwen2.5:7b

### Bao loi thieu RAM cho model

Neu gap loi:

`model requires more system memory ... than is available ...`

- Chon model nhe hon trong giao dien: `qwen2.5:0.5b` hoac `qwen2.5:1.5b`
- Hoac pull model nhe:

```bash
ollama pull qwen2.5:0.5b
```

### PDF khong trich xuat duoc text

- Thu file PDF co text layer (khong phai scan anh)

### DOCX khong doc duoc

- Kiem tra da cai dependency: `pip install -r requirements.txt`
- Dam bao file la dinh dang `.docx` hop le (khong phai `.doc`)

### Canh bao Pydantic V1 voi Python 3.14+

Neu thay canh bao:

Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.

Day la canh bao tu dependency, khong lam app crash ngay. De on dinh nhat, nen dung Python 3.10 hoac 3.11 cho du an nay.

### GET /favicon.ico 404 Not Found

Thong bao nay khong anh huong chuc nang chinh cua app. Trinh duyet tu dong goi favicon, nhung du an hien chua cung cap file icon.

## Kien truc moi

- Backend API: FastAPI (`backend.py`)
- Frontend: HTML/CSS/JS tach rieng trong thu muc `frontend/`
- RAG core: `rag_engine.py`

## Cau Truc Thu Muc

```plaintext
Project-LLMs-Rag-Agent/
|- app.py
|- backend.py
|- frontend/
|  |- index.html
|  |- styles.css
|  `- app.js
|- rag_engine.py
|- requirements.txt
|- README.md
|- data/
`- documentation/
```
