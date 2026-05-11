"""Test circular seal OCR with improved angle-based sorting."""
import cv2
import numpy as np
from ocr_processor import OCRProcessor, OCRConfig

# Create synthetic circular seal
img = np.full((800, 800, 3), 255, dtype=np.uint8)

# Draw red circle outline
cv2.circle(img, (400, 400), 320, (0, 0, 255), 8)  # outer circle
cv2.circle(img, (400, 400), 280, (0, 0, 255), 6)  # inner circle

# Add text arranged in lines from top to center (as in typical seals)
texts = [
    ("DOANH NGHIEP", 250),
    ("TU NHAN", 350),
    ("THIEN Y DAT", 450)
]

for text, y in texts:
    cv2.putText(img, text, (150, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

# Test with improved OCR
cfg = OCRConfig(confidence_threshold=0.15)
result = OCRProcessor.extract_text_from_image(img, cfg)

print("🔍 Circular Seal OCR Result:")
print("=" * 40)
print(result if result else "(empty)")
print("=" * 40)
print("\n✅ Test completed successfully!")
