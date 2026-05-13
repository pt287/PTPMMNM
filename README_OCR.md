Integrated OCR & Document Processing for SmartDoc AI
==================================================

## Overview

SmartDoc AI giờ đã hỗ trợ:
- ✅ Extract text từ PDF scan (không có text layer)
- ✅ Extract images từ DOCX và OCR chúng
- ✅ Detect dấu mộc/con dấu đỏ (Việt Nam)
- ✅ OCR với tiếng Việt + Tiếng Anh
- ✅ Auto-rotate scan ảnh lệch
- ✅ Preprocessing image (denoise, threshold, sharpen)
- ✅ Multi-angle OCR cho dấu (0°, 90°, 180°, 270°)

---

## Architecture

```
backend.py
    ├── /api/process-document    → Full processing (text + stamps)
    ├── /api/extract-text        → Text only (fast)
    ├── /api/detect-stamps       → Stamps only (for verification)
    └── /api/ocr/status          → Health check
    
services/
    ├── ocr_service.py           → EasyOCR + image preprocessing
    ├── stamp_detector.py        → Red seal detection + OCR
    └── document_parser.py       → PDF/DOCX parsing with fallback
```

---

## Installation

### 1. Add OCR Dependencies

Đã cập nhật `requirements.txt`:

```bash
pip install -r requirements.txt

# Or manually:
pip install easyocr==1.7.1 opencv-python==4.8.1.78 pdf2image==1.16.3 PyMuPDF==1.24.0
```

### 2. System Requirements

**Windows:**
```bash
# For PDF to image conversion, you need poppler
# Download from: https://github.com/oschwartz10612/poppler-windows/releases/
# Or use choco:
choco install poppler
```

**Linux:**
```bash
sudo apt-get install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

### 3. Initialize OCR

Lần đầu chạy backend:
```bash
cd c:\PTPMMNM\PTPMMNM-main
python backend.py
```

Lần đầu, EasyOCR sẽ download model (~150MB).

---

## API Endpoints

### 1. POST /api/process-document

**Full processing**: Extract text + detect stamps

```bash
curl -X POST http://localhost:8000/api/process-document \
  -F "file=@contract.pdf" \
  -F "use_ocr=true" \
  -F "detect_stamps=true"
```

**Response:**
```json
{
  "success": true,
  "file_name": "contract.pdf",
  "document_type": "pdf",
  "full_text": "HỢPĐỒNG MUA BÁN...",
  "text_length": 2453,
  "stamps": [
    {
      "text": "CÔNG TY ABC TNHH",
      "confidence": 0.91,
      "bbox": [150, 200, 400, 450],
      "region_type": "red_seal",
      "best_angle": 0,
      "area": 12450
    }
  ],
  "stamp_count": 1,
  "pages": [
    {
      "page_num": 1,
      "text": "...",
      "used_ocr": false,
      "stamps": []
    }
  ]
}
```

---

### 2. POST /api/extract-text

**Fast text extraction** (không detect stamp)

```bash
curl -X POST http://localhost:8000/api/extract-text \
  -F "file=@document.docx"
```

**Response:**
```json
{
  "success": true,
  "file_name": "document.docx",
  "text": "Nội dung...",
  "text_length": 5000,
  "document_type": "docx"
}
```

---

### 3. POST /api/detect-stamps

**Detect stamps only** (fast, for verification)

```bash
curl -X POST http://localhost:8000/api/detect-stamps \
  -F "file=@official_doc.pdf"
```

**Response:**
```json
{
  "success": true,
  "file_name": "official_doc.pdf",
  "document_type": "pdf",
  "stamps": [
    {
      "text": "CỤC THUẾ TP.HCM",
      "confidence": 0.87,
      "bbox": [500, 600, 700, 800],
      "region_type": "red_seal",
      "best_angle": 90,
      "area": 18000
    }
  ],
  "stamp_count": 1
}
```

---

### 4. GET /api/ocr/status

**Health check**

```bash
curl http://localhost:8000/api/ocr/status
```

**Response:**
```json
{
  "status": "ready",
  "ocr_engine": "EasyOCR",
  "languages": ["vi", "en"],
  "gpu_enabled": false
}
```

---

## Features in Detail

### OCR Service (`services/ocr_service.py`)

**Image Preprocessing:**
- Grayscale conversion
- Denoising (fastNlMeansDenoising)
- Contrast enhancement (CLAHE)
- Adaptive threshold
- Sharpening filter
- Auto-rotation detection
- Upscaling for small images

**OCR:**
- EasyOCR reader with Vietnamese + English
- Support for rotated text
- Confidence scoring
- Multi-angle OCR for stamps (0°, 90°, 180°, 270°)

**Usage:**
```python
from services import OCRService
import cv2

ocr = OCRService()

# From image file
result = ocr.extract_text_from_file("image.jpg")
print(result["text"])
print(f"Confidence: {result['confidence']}")

# From numpy array
image = cv2.imread("image.jpg")
result = ocr.extract_text(image)

# Multi-angle for stamps
result = ocr.extract_text_multi_angles(image)
print(f"Best angle: {result['best_angle']}°")
```

---

### Stamp Detector (`services/stamp_detector.py`)

**Detection Methods:**
1. Red seal detection (HSV color range)
2. Circular contour detection
3. Morphological operations (closing, opening)

**Stamp OCR:**
- Multi-angle OCR (4 angles)
- Returns best confidence result
- Bounding box + area info

**Usage:**
```python
from services import StampDetector
import cv2

stamp_detector = StampDetector()

image = cv2.imread("document_with_stamp.jpg")

# Detect stamps
stamps = stamp_detector.detect_stamps(image, min_area=500)
for stamp in stamps:
    print(stamp["region_type"])  # "red_seal" or "circular"
    print(f"Area: {stamp['area']}")

# Extract stamp text
stamp_texts = stamp_detector.extract_stamp_text(image)
for text in stamp_texts:
    print(f"{text['text']} (conf: {text['confidence']})")

# Debug: Mark stamps on image
marked = stamp_detector.mark_stamps_on_image(image, stamps)
cv2.imwrite("output_with_marks.jpg", marked)
```

---

### Document Parser (`services/document_parser.py`)

**Supported Formats:**
- PDF (with fallback to OCR for scans)
- DOCX/DOC (text + embedded images)

**PDF Processing:**
1. Try extract text from PDF layer
2. If empty → OCR per page
3. Detect stamps on each page
4. Merge results

**DOCX Processing:**
1. Extract paragraphs
2. Extract embedded images
3. OCR each image
4. Detect stamps in images
5. Merge text + OCR results

**Usage:**
```python
from services import DocumentParser

parser = DocumentParser()

# Parse any document
result = parser.parse_document(
    "contract.pdf",
    use_ocr=True,
    detect_stamps=True
)

print(result["full_text"])
print(f"Stamps detected: {len(result['all_stamps'])}")

# Or specific types
pdf_result = parser.parse_pdf("scan.pdf")
docx_result = parser.parse_docx("document.docx")
```

---

## Performance

**Approximate processing time (per page/image):**

| Operation | Time | Notes |
|-----------|------|-------|
| Text extraction (PDF layer) | 10-50ms | No OCR needed |
| OCR (EasyOCR) | 1-3s | Per page/image |
| Stamp detection | 200-500ms | Per page |
| Multi-angle stamp OCR | 3-8s | 4 angles × OCR |

**Example:**
- 10-page PDF scan: ~15-30 seconds
- DOCX + 5 images: ~10-20 seconds
- Stamp detection only: ~5 seconds

---

## Error Handling

All endpoints return structured error responses:

```json
{
  "detail": "Error description",
  "status": 400 or 500
}
```

**Common errors:**
- `400`: Unsupported file type, empty file
- `500`: OCR processing error, memory issues
- `503`: Model requires more system memory (from previous fix)

**Debug logging:**
All operations log to Python logger. Check backend console for details.

---

## Configuration

### Customize OCR

Edit `services/ocr_service.py`:

```python
# Change languages
self.reader = easyocr.Reader(['vi', 'en', 'zh'], gpu=False)

# Adjust preprocessing
# - Change denoise strength: h=10
# - CLAHE clip limit: clipLimit=2.0
# - Threshold block size: 11 (must be odd)
```

### Customize Stamp Detection

Edit `services/stamp_detector.py`:

```python
# Red color range (HSV)
self.lower_red1 = np.array([0, 100, 100])
self.upper_red1 = np.array([10, 255, 255])

# Min stamp area
min_area = 500  # Pixels

# Min circularity for circle detection
min_circularity = 0.5
```

---

## Integration with RAG

Already integrated! When building RAG index:

```python
# In backend.py:
parser = DocumentParser()
result = parser.parse_document(file_path, use_ocr=True)
full_text = result["full_text"]

# Then feed to RAG:
chunks = split_documents([Document(page_content=full_text)])
```

All OCR'd text automatically becomes searchable in RAG system.

---

## Examples

### Python Client

```python
import requests

# Process document
with open("contract.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/process-document",
        files={"file": f},
        data={"use_ocr": True, "detect_stamps": True}
    )
    result = response.json()
    
    print("📄 Text:", result["full_text"][:200])
    print("🔴 Stamps:", result["stamp_count"])
    for stamp in result["stamps"]:
        print(f"  - {stamp['text']}")
```

### See also: `examples_ocr_api.py`

Full working examples with curl commands.

---

## Troubleshooting

### Issue: "No module named 'easyocr'"

```bash
pip install easyocr
```

### Issue: "Cannot find poppler"

**Windows:**
```bash
# Option 1: Install via choco
choco install poppler

# Option 2: Download and add to PATH
# https://github.com/oschwartz10612/poppler-windows/releases/
```

**Linux:**
```bash
sudo apt-get install poppler-utils
```

### Issue: "CUDA out of memory"

EasyOCR uses CPU by default. If GPU errors occur:

```python
# Force CPU
ocr_service = OCRService()  # gpu=False is default
```

### Issue: "Low OCR confidence"

If confidence < 0.6, document may be:
- Very blurry
- Wrong language setting
- Text too small

Try:
- Increase image resolution
- Manually rotate PDF before uploading
- Check language setting in config

---

## File Structure

```
services/
├── __init__.py               # Package exports
├── ocr_service.py            # EasyOCR + preprocessing
├── stamp_detector.py         # Stamp/seal detection
└── document_parser.py        # PDF/DOCX parsing

examples_ocr_api.py           # Usage examples + curl commands
README_OCR.md                 # This file

backend.py                    # Updated with /api/process-document
requirements.txt              # Updated with OCR packages
```

---

## Next Steps

1. **Test endpoints:**
   ```bash
   python examples_ocr_api.py
   ```

2. **Use in RAG:**
   - Already built-in! Upload scanned PDF → auto-OCR → searchable

3. **Extend:**
   - Add barcode detection
   - Table extraction
   - Handwriting OCR
   - Custom stamp templates

---

## References

- **EasyOCR**: https://github.com/JaidedAI/EasyOCR
- **OpenCV**: https://docs.opencv.org/
- **PyMuPDF**: https://pymupdf.readthedocs.io/
- **Python-docx**: https://python-docx.readthedocs.io/

---

**Version:** 1.0  
**Last Updated:** May 12, 2026  
**Language:** Python 3.10+
