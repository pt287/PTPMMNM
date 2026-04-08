# SmartDoc AI (Local PDF Q&A)

Ung dung hoi dap tai lieu PDF theo kien truc RAG, ho tro tieng Viet va tieng Anh.

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
ollama serve
```

### 4) Chay web app (frontend + backend)

```bash
uvicorn backend:app --reload
```

Sau do mo trinh duyet: http://127.0.0.1:8000

## Cach Dung

1. Upload 1 hoac nhieu file PDF.
2. Bam Build RAG Index.
3. Dat cau hoi trong khung chat.
4. Nhan cau tra loi va xem chunk duoc truy hoi.

## Cau Hinh Chinh

- LLM: qwen2.5:7b (Ollama)
- Embedding: paraphrase-multilingual-mpnet-base-v2
- Loader: PDFPlumberLoader
- Chunk: 1000
- Overlap: 100
- Retrieval: top k = 3
- Vector DB: FAISS

## Loi Thuong Gap

### Khong cai duoc faiss-cpu==1.9.0

Ban dang dung Python moi (3.12+). Du an da tu dong dung faiss-cpu >= 1.12.0 trong requirements.

### App khong tra loi

- Kiem tra Ollama dang chay: ollama serve
- Kiem tra model da tai: ollama pull qwen2.5:7b

### PDF khong trich xuat duoc text

- Thu file PDF co text layer (khong phai scan anh)

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
