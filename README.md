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
