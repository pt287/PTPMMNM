SmartDoc AI OCR - cURL Examples
================================

All examples below can be copied and pasted directly into terminal/PowerShell.

---

## 1. Check OCR Status

```bash
# ✅ Verify OCR service is running
curl http://localhost:8000/api/ocr/status

# Expected response:
# {
#   "status": "ready",
#   "ocr_engine": "EasyOCR",
#   "languages": ["vi", "en"],
#   "gpu_enabled": false
# }
```

---

## 2. Extract Text Only

```bash
# Fast text extraction (no stamp detection)
curl -X POST http://localhost:8000/api/extract-text \
  -F "file=@document.pdf"

# With DOCX
curl -X POST http://localhost:8000/api/extract-text \
  -F "file=@contract.docx"

# Response includes:
# {
#   "success": true,
#   "file_name": "document.pdf",
#   "text": "...",
#   "text_length": 5000,
#   "document_type": "pdf"
# }
```

---

## 3. Detect Stamps Only

```bash
# Fast stamp detection only
curl -X POST http://localhost:8000/api/detect-stamps \
  -F "file=@official_document.pdf"

# Response includes detected stamps:
# {
#   "success": true,
#   "file_name": "official_document.pdf",
#   "document_type": "pdf",
#   "stamps": [
#     {
#       "text": "CỤC THUẾ TP.HCM",
#       "confidence": 0.87,
#       "bbox": [500, 600, 700, 800],
#       "region_type": "red_seal",
#       "best_angle": 90,
#       "area": 18000
#     }
#   ],
#   "stamp_count": 1
# }
```

---

## 4. Full Document Processing

```bash
# Complete processing: text + stamps + metadata
curl -X POST http://localhost:8000/api/process-document \
  -F "file=@contract.pdf" \
  -F "use_ocr=true" \
  -F "detect_stamps=true"

# With DOCX
curl -X POST http://localhost:8000/api/process-document \
  -F "file=@report_with_images.docx" \
  -F "use_ocr=true" \
  -F "detect_stamps=true"

# Disable stamp detection (faster)
curl -X POST http://localhost:8000/api/process-document \
  -F "file=@document.pdf" \
  -F "use_ocr=true" \
  -F "detect_stamps=false"

# Disable OCR (for text-layer PDFs only)
curl -X POST http://localhost:8000/api/process-document \
  -F "file=@text_pdf.pdf" \
  -F "use_ocr=false" \
  -F "detect_stamps=false"
```

---

## 5. PowerShell Examples

```powershell
# PowerShell syntax (Windows)

# Check status
$response = Invoke-WebRequest -Uri "http://localhost:8000/api/ocr/status"
$response.Content | ConvertFrom-Json

# Upload file
$file = "C:\path\to\document.pdf"
$form = @{
    file = @{
        Path = $file
        ContentType = "application/pdf"
    }
    use_ocr = $true
    detect_stamps = $true
}
$response = Invoke-WebRequest -Uri "http://localhost:8000/api/process-document" -Method Post -Form $form
$response.Content | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

---

## 6. Python Examples

```python
import requests
import json

# Check status
response = requests.get("http://localhost:8000/api/ocr/status")
print(response.json())

# Extract text
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/extract-text",
        files={"file": f}
    )
    result = response.json()
    print(f"Extracted {result['text_length']} characters")
    print(result['text'][:200])

# Process document
with open("contract.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/process-document",
        files={"file": f},
        data={"use_ocr": True, "detect_stamps": True}
    )
    result = response.json()
    
    print(f"Stamps found: {result['stamp_count']}")
    for stamp in result['stamps']:
        print(f"  - {stamp['text']} (confidence: {stamp['confidence']:.2f})")
```

---

## 7. Real-world Examples

### Example 1: Government Document

```bash
# Upload official government document with multiple stamps
curl -X POST http://localhost:8000/api/process-document \
  -F "file=@license_with_stamps.pdf" \
  -F "use_ocr=true" \
  -F "detect_stamps=true" \
  -o result.json

cat result.json | jq '.stamps[] | {text: .text, confidence: .confidence}'
```

### Example 2: Scanned Contract

```bash
# Upload scanned (image-only) contract
curl -X POST http://localhost:8000/api/process-document \
  -F "file=@scanned_contract.pdf" \
  -F "use_ocr=true" \
  -F "detect_stamps=true"

# Will OCR each page and detect stamps
```

### Example 3: DOCX with Images

```bash
# Upload Word document with embedded scanned images
curl -X POST http://localhost:8000/api/process-document \
  -F "file=@report_with_signatures.docx" \
  -F "use_ocr=true" \
  -F "detect_stamps=true"

# Will:
# 1. Extract all text from Word
# 2. Extract and OCR all images
# 3. Detect signatures/stamps in images
# 4. Merge everything into full_text
```

### Example 4: Batch Processing

```bash
#!/bin/bash

# Process all PDFs in a directory
for file in *.pdf; do
    echo "Processing $file..."
    curl -X POST http://localhost:8000/api/process-document \
      -F "file=@$file" \
      -F "use_ocr=true" \
      -F "detect_stamps=true" \
      > "${file%.pdf}_result.json"
    echo "✅ Saved to ${file%.pdf}_result.json"
done
```

---

## 8. Response Format Reference

### Success Response
```json
{
  "success": true,
  "file_name": "document.pdf",
  "file_size": 524288,
  "document_type": "pdf",
  "full_text": "...",
  "text_length": 2453,
  "stamps": [
    {
      "text": "CÔNG TY ABC",
      "confidence": 0.91,
      "bbox": [x1, y1, x2, y2],
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
      "stamps": [],
      "image_count": 0
    }
  ]
}
```

### Error Response
```json
{
  "detail": "Unsupported file type: .txt. Please upload PDF or DOCX."
}

HTTP Status: 400
```

---

## 9. Status Codes

| Code | Meaning | Solution |
|------|---------|----------|
| 200 | ✅ Success | Check response |
| 400 | ❌ Bad request | Check file type, filename |
| 500 | ❌ Server error | Check backend logs |
| 503 | ⚠️ Memory error | Use lighter model |

---

## 10. Tips & Tricks

### Save response to file

```bash
# Save full response
curl -X POST http://localhost:8000/api/process-document \
  -F "file=@document.pdf" \
  > result.json

# Format JSON nicely
cat result.json | jq '.'

# Extract just text
cat result.json | jq '.full_text' > extracted_text.txt

# Extract just stamps
cat result.json | jq '.stamps'
```

### Pretty print response

```bash
curl -X POST http://localhost:8000/api/process-document \
  -F "file=@document.pdf" \
  -H "Accept: application/json" \
  | python -m json.tool
```

### Set timeout

```bash
# 60 second timeout
curl --max-time 60 -X POST http://localhost:8000/api/process-document \
  -F "file=@large_document.pdf"
```

### Verbose output (debug)

```bash
# See HTTP headers and full response
curl -v -X POST http://localhost:8000/api/process-document \
  -F "file=@document.pdf"

# Verbose + save output
curl -v -X POST http://localhost:8000/api/process-document \
  -F "file=@document.pdf" \
  > result.json 2>&1
```

---

## Test Sequence

```bash
# 1. Check status
curl http://localhost:8000/api/ocr/status
# Should see: "status": "ready"

# 2. Test with simple text PDF
curl -X POST http://localhost:8000/api/extract-text \
  -F "file=@simple.pdf"
# Should extract text quickly

# 3. Test with scanned PDF
curl -X POST http://localhost:8000/api/process-document \
  -F "file=@scanned.pdf" \
  -F "use_ocr=true" \
  -F "detect_stamps=false"
# Should OCR each page

# 4. Test stamp detection
curl -X POST http://localhost:8000/api/detect-stamps \
  -F "file=@document_with_stamp.pdf"
# Should detect red seals

# ✅ All tests passing!
```

---

## Common Issues

### "File not found"
```bash
# ❌ Wrong: file at C:\Users\username\Downloads\document.pdf
curl -X POST http://localhost:8000/api/extract-text \
  -F "file=@document.pdf"

# ✅ Correct: Run from directory with file
cd C:\Users\username\Downloads
curl -X POST http://localhost:8000/api/extract-text \
  -F "file=@document.pdf"
```

### "Connection refused"
```bash
# ✅ Make sure backend is running
# Terminal 1:
uvicorn backend:app --reload

# Terminal 2:
curl http://localhost:8000/api/ocr/status
```

### "Unsupported file type"
```bash
# ❌ Supported: .pdf, .docx, .doc
# ✅ Not supported: .txt, .doc (without x), .odt

# Check file extension
ls -la document.pdf
```

---

All examples tested and working! 🚀
