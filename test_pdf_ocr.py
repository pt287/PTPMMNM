"""Test OCR on real PDF with output to file."""
from pathlib import Path
from ocr_processor import OCRProcessor, OCRConfig

pdf = Path('c:/PTPMMNM/data/uploads/session_36/stamp_test.pdf')
imgs = OCRProcessor.extract_images_from_pdf(str(pdf), dpi=300)
cfg = OCRConfig(confidence_threshold=0.15)

with open('test_result.txt', 'w', encoding='utf-8') as f:
    f.write(f"Total pages: {len(imgs)}\n")
    f.write("=" * 50 + "\n\n")
    
    if imgs:
        print(f"Processing page 1...")
        text = OCRProcessor.extract_text_from_image(imgs[0][0], cfg)
        f.write("=== PAGE 1 ===\n")
        f.write(text + "\n\n")
        print(f"✅ Done! Saved to test_result.txt")
    else:
        f.write("No images found\n")
        print("❌ No images found")
