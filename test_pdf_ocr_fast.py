"""Test OCR on PDF with lower DPI."""
from pathlib import Path
from ocr_processor import OCRProcessor, OCRConfig

pdf = Path('c:/PTPMMNM/data/uploads/session_36/stamp_test.pdf')
print(f"PDF exists: {pdf.exists()}")

# Use lower DPI to speed up
imgs = OCRProcessor.extract_images_from_pdf(str(pdf), dpi=150)
print(f"Pages extracted: {len(imgs)}")

if imgs:
    cfg = OCRConfig(confidence_threshold=0.3)  # Use default confidence
    print("Processing page 1 with OCR...")
    text = OCRProcessor.extract_text_from_image(imgs[0][0], cfg)
    print("=== RESULT ===")
    print(text if text else "(empty)")
else:
    print("No images")
