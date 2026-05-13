"""
SmartDoc AI OCR API - Ví dụ sử dụng
================================================

Hướng dẫn gọi các endpoint OCR mới
"""

import requests
import json
from pathlib import Path

# Base URL
BASE_URL = "http://localhost:8000"

# ============================================================================
# Example 1: Process Document (PDF/DOCX) với OCR + Stamp Detection
# ============================================================================

def example_process_document():
    """
    POST /api/process-document
    
    Xử lý PDF/DOCX:
    - Extract text
    - Fallback to OCR nếu PDF scan
    - Detect dấu mộc
    - Trả về stamps + full text
    """
    print("\n=== Example 1: Process Document ===\n")
    
    # Giả sử bạn có file tại đây
    pdf_file = "example.pdf"  # Hoặc "contract_with_stamp.docx"
    
    if not Path(pdf_file).exists():
        print(f"File {pdf_file} not found. Bỏ qua ví dụ này.")
        return
    
    with open(pdf_file, "rb") as f:
        files = {"file": (Path(pdf_file).name, f, "application/pdf")}
        data = {
            "use_ocr": True,
            "detect_stamps": True,
        }
        
        response = requests.post(
            f"{BASE_URL}/api/process-document",
            files=files,
            data=data,
        )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Success!")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # In ra text
        print(f"\n📄 Extracted Text ({result['text_length']} chars):")
        print(result['full_text'][:500] + "...")
        
        # In ra stamps
        print(f"\n🔴 Detected Stamps ({result['stamp_count']}):")
        for stamp in result['stamps']:
            print(f"  - {stamp['text']} (confidence: {stamp['confidence']:.2f})")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())


# ============================================================================
# Example 2: Extract Text Only
# ============================================================================

def example_extract_text():
    """
    POST /api/extract-text
    
    Chỉ extract text (nhanh hơn, không detect stamp)
    """
    print("\n=== Example 2: Extract Text Only ===\n")
    
    pdf_file = "example.pdf"
    
    if not Path(pdf_file).exists():
        print(f"File {pdf_file} not found.")
        return
    
    with open(pdf_file, "rb") as f:
        files = {"file": (Path(pdf_file).name, f)}
        response = requests.post(
            f"{BASE_URL}/api/extract-text",
            files=files,
        )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Success!")
        print(f"File: {result['file_name']}")
        print(f"Type: {result['document_type']}")
        print(f"Text length: {result['text_length']} chars")
        print(f"\nFirst 500 chars:\n{result['text'][:500]}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())


# ============================================================================
# Example 3: Detect Stamps Only
# ============================================================================

def example_detect_stamps():
    """
    POST /api/detect-stamps
    
    Chỉ phát hiện dấu (nhanh nhất)
    """
    print("\n=== Example 3: Detect Stamps Only ===\n")
    
    pdf_file = "example.pdf"
    
    if not Path(pdf_file).exists():
        print(f"File {pdf_file} not found.")
        return
    
    with open(pdf_file, "rb") as f:
        files = {"file": (Path(pdf_file).name, f)}
        response = requests.post(
            f"{BASE_URL}/api/detect-stamps",
            files=files,
        )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Success!")
        print(f"File: {result['file_name']}")
        print(f"Stamp count: {result['stamp_count']}")
        
        for i, stamp in enumerate(result['stamps'], 1):
            print(f"\nStamp {i}:")
            print(f"  Text: {stamp['text']}")
            print(f"  Confidence: {stamp['confidence']:.2f}")
            print(f"  Type: {stamp['region_type']}")
            print(f"  Area: {stamp['area']}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())


# ============================================================================
# Example 4: Check OCR Status
# ============================================================================

def example_ocr_status():
    """
    GET /api/ocr/status
    
    Kiểm tra trạng thái OCR service
    """
    print("\n=== Example 4: OCR Status ===\n")
    
    response = requests.get(f"{BASE_URL}/api/ocr/status")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ OCR Service Status:")
        print(json.dumps(result, indent=2))
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())


# ============================================================================
# Response Examples
# ============================================================================

RESPONSE_EXAMPLES = {
    "process_document_response": {
        "success": True,
        "file_name": "contract.pdf",
        "file_size": 524288,
        "document_type": "pdf",
        "full_text": "HỢPĐỒNG MUA BÁN...",
        "text_length": 2453,
        "stamps": [
            {
                "text": "CÔNG TY TNHH ABC",
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
                "text": "Page 1 text...",
                "used_ocr": False,
                "stamps": [],
                "image_count": 0
            }
        ],
        "extracted_images": [],
        "metadata": {}
    },
    
    "extract_text_response": {
        "success": True,
        "file_name": "document.docx",
        "text": "Văn bản trích xuất...",
        "text_length": 5000,
        "document_type": "docx"
    },
    
    "detect_stamps_response": {
        "success": True,
        "file_name": "official_doc.pdf",
        "document_type": "pdf",
        "stamps": [
            {
                "text": "CỤC THUẾ NHƯ THÀNH PHỐ HỒ CHÍ MINH",
                "confidence": 0.87,
                "bbox": [500, 600, 700, 800],
                "region_type": "red_seal",
                "best_angle": 90,
                "area": 18000
            },
            {
                "text": "NGÀY 2024",
                "confidence": 0.79,
                "bbox": [100, 700, 200, 750],
                "region_type": "red_seal",
                "best_angle": 0,
                "area": 4500
            }
        ],
        "stamp_count": 2
    },
    
    "ocr_status_response": {
        "status": "ready",
        "ocr_engine": "EasyOCR",
        "languages": ["vi", "en"],
        "gpu_enabled": False
    }
}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SmartDoc AI OCR API Examples")
    print("=" * 70)
    
    # Check status first
    example_ocr_status()
    
    # Run examples
    example_process_document()
    example_extract_text()
    example_detect_stamps()
    
    # Print response examples
    print("\n\n" + "=" * 70)
    print("Response Examples (JSON)")
    print("=" * 70)
    
    for key, value in RESPONSE_EXAMPLES.items():
        print(f"\n{key}:")
        print(json.dumps(value, indent=2, ensure_ascii=False))


# ============================================================================
# Using with curl
# ============================================================================

"""
# Process Document
curl -X POST http://localhost:8000/api/process-document \\
  -F "file=@contract.pdf" \\
  -F "use_ocr=true" \\
  -F "detect_stamps=true"

# Extract Text
curl -X POST http://localhost:8000/api/extract-text \\
  -F "file=@document.docx"

# Detect Stamps
curl -X POST http://localhost:8000/api/detect-stamps \\
  -F "file=@official_document.pdf"

# Check Status
curl http://localhost:8000/api/ocr/status
"""
