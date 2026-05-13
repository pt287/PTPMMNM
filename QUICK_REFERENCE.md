SmartDoc AI OCR - Quick Reference Card
======================================

## Start Here

```bash
# 1. Make sure backend is running
cd c:\PTPMMNM\PTPMMNM-main
uvicorn backend:app --reload

# 2. In another terminal, test OCR
curl http://localhost:8000/api/ocr/status

# 3. Should see:
# {
#   "status": "ready",
#   "ocr_engine": "EasyOCR",
#   "languages": ["vi", "en"],
#   "gpu_enabled": false
# }
```

---

## Four API Endpoints

### 1️⃣ Full Processing (Text + Stamps)

```bash
curl -X POST http://localhost:8000/api/process-document \
  -F "file=@document.pdf" \
  -F "use_ocr=true" \
  -F "detect_stamps=true"
```

**Returns:**
- `full_text` - All extracted text
- `stamps` - Array of detected stamps with text
- `pages` - Per-page details
- `text_length`, `stamp_count` - Metrics

**Use for:** Government docs, contracts, forms

---

### 2️⃣ Text Only (Fast)

```bash
curl -X POST http://localhost:8000/api/extract-text \
  -F "file=@document.pdf"
```

**Returns:**
- `text` - Extracted text
- `text_length` - Character count
- `document_type` - pdf/docx

**Use for:** Quick text extraction, no stamp needed

---

### 3️⃣ Detect Stamps Only

```bash
curl -X POST http://localhost:8000/api/detect-stamps \
  -F "file=@official_doc.pdf"
```

**Returns:**
- `stamps` - Array of detected stamps
- `stamp_count` - Number found

**Use for:** Verification, audit, stamp extraction

---

### 4️⃣ Check Status

```bash
curl http://localhost:8000/api/ocr/status
```

**Returns:**
- `status` - "ready" or "error"
- `ocr_engine` - "EasyOCR"
- `languages` - ["vi", "en"]
- `gpu_enabled` - true/false

**Use for:** Health check, debugging

---

## File Support

✅ **Supported:**
- `.pdf` (native + scans)
- `.docx` (with images)
- `.doc` (basic support)

❌ **Not supported:**
- `.txt`, `.odt`, `.xlsx`
- Images directly (must be in PDF/DOCX)

---

## Response Structure

### Success (200)
```json
{
  "success": true,
  "full_text": "...",
  "stamps": [
    {
      "text": "CÔNG TY ABC",
      "confidence": 0.91,
      "bbox": [150, 200, 400, 450],
      "region_type": "red_seal",
      "best_angle": 0,
      "area": 12450
    }
  ]
}
```

### Error (400/500)
```json
{
  "detail": "Error description"
}
```

---

## Processing Time

| Operation | Time |
|-----------|------|
| Text extract | 50-200ms |
| OCR per page | 1-3s |
| Stamp detect | 200-500ms |
| Full process (10 pages) | 15-30s |

---

## Configuration

### Supported Languages

```python
# In services/ocr_service.py:
self.reader = easyocr.Reader(['vi', 'en'])  # Default
# Add more:
self.reader = easyocr.Reader(['vi', 'en', 'zh', 'ja'])
```

### Stamp Detection Color

```python
# In services/stamp_detector.py:
self.lower_red1 = np.array([0, 100, 100])      # Red range start
self.upper_red1 = np.array([10, 255, 255])     # Red range end
```

---

## Common Tasks

### Upload to RAG

```bash
# Upload document → auto-OCR → searchable
# Just use normal upload in SmartDoc UI
# Scanned PDFs automatically OCR'd
```

### Save Output

```bash
curl -X POST http://localhost:8000/api/process-document \
  -F "file=@document.pdf" \
  > result.json

cat result.json | jq '.full_text' > text.txt
```

### Extract Just Stamps

```bash
curl -X POST http://localhost:8000/api/process-document \
  -F "file=@document.pdf" \
  | jq '.stamps[] | {text, confidence}'
```

### Batch Process

```bash
for file in *.pdf; do
  echo "Processing $file..."
  curl -X POST http://localhost:8000/api/process-document \
    -F "file=@$file" \
    > "${file%.pdf}.json"
done
```

---

## Python Quick Start

```python
import requests

# Check status
r = requests.get("http://localhost:8000/api/ocr/status")
print(r.json())

# Process file
with open("document.pdf", "rb") as f:
    r = requests.post(
        "http://localhost:8000/api/process-document",
        files={"file": f},
        data={"use_ocr": True, "detect_stamps": True}
    )
    result = r.json()
    
    print(result['full_text'][:500])
    for stamp in result['stamps']:
        print(f"Stamp: {stamp['text']}")
```

---

## Troubleshooting

### Connection Refused
```bash
# Backend not running
uvicorn backend:app --reload
```

### No Module Named
```bash
# Dependencies not installed
pip install -r requirements.txt
```

### Low Confidence OCR
- Document too blurry
- Try upscaling before upload
- Check language setting

### No Stamps Detected
- Color range may not match
- Stamp too small (min 500px)
- Try `/api/process-document` instead

---

## Files to Know

- `services/ocr_service.py` - OCR engine
- `services/stamp_detector.py` - Stamp detection
- `services/document_parser.py` - PDF/DOCX parser
- `README_OCR.md` - Full documentation
- `CURL_EXAMPLES.md` - More examples
- `examples_ocr_api.py` - Python examples

---

## Key Features

✅ Extracts text from scanned PDFs
✅ Detects red seals (dấu mộc)
✅ Supports Vietnamese + English
✅ Auto-rotates tilted scans
✅ Processes DOCX images
✅ Returns confidence scores
✅ Multi-angle stamp OCR
✅ Integration with RAG

---

## Next Steps

1. Test: `python examples_ocr_api.py`
2. Upload: Use SmartDoc UI
3. Ask: Questions become searchable
4. Extend: Add more features

---

**Questions?** See README_OCR.md or CURL_EXAMPLES.md

**Ready?** Start backend: `uvicorn backend:app --reload` 🚀
