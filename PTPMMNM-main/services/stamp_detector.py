"""
Stamp/Seal Detector - phát hiện và trích xuất text từ dấu mộc/con dấu đỏ.
"""

import logging
import cv2
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from .ocr_service import OCRService

logger = logging.getLogger(__name__)


class StampDetector:
    """Phát hiện dấu mộc/con dấu trong ảnh."""
    
    def __init__(self, ocr_service: Optional[OCRService] = None):
        """
        Khởi tạo StampDetector.
        
        Args:
            ocr_service: OCRService instance hoặc None (sẽ tạo mới)
        """
        self.ocr_service = ocr_service or OCRService()
        
        # Màu đỏ range (HSV)
        # Red is at 0° and 180° in HSV
        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])

    def _detect_red_regions(self, image: np.ndarray, threshold: int = 100) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Detect vùng màu đỏ trong ảnh.
        
        Returns:
            (mask, contours)
        """
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Create mask cho màu đỏ
            mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
            mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            return mask, contours
        
        except Exception as e:
            logger.error(f"Error in red region detection: {e}")
            return np.zeros_like(image[:, :, 0]), []

    def _is_circular_contour(self, contour: np.ndarray, min_circularity: float = 0.5) -> bool:
        """
        Check xem contour có phải hình tròn/oval không.
        """
        try:
            if len(contour) < 5:
                return False
            
            area = cv2.contourArea(contour)
            if area < 100:  # Quá nhỏ
                return False
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter < 10:
                return False
            
            # Circularity = 4π * Area / Perimeter²
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            return circularity > min_circularity
        
        except:
            return False

    def detect_stamps(self, image: np.ndarray, min_area: int = 500) -> List[Dict[str, Any]]:
        """
        Phát hiện dấu mộc/con dấu trong ảnh.
        
        Returns:
            [
                {
                    "bbox": [x1, y1, x2, y2],
                    "area": 1000,
                    "is_red": True,
                    "image": cropped_image_array,
                    "region_type": "red_seal" | "circular"
                }
            ]
        """
        try:
            stamps = []
            
            # Detect red regions
            mask_red, contours_red = self._detect_red_regions(image)
            
            for contour in contours_red:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                if w < 20 or h < 20:  # Quá nhỏ
                    continue
                
                # Extract region
                stamp_region = image[y:y+h, x:x+w].copy()
                
                stamps.append({
                    "bbox": [x, y, x + w, y + h],
                    "area": int(area),
                    "is_red": True,
                    "image": stamp_region,
                    "region_type": "red_seal"
                })
            
            # Detect circular contours (dấu tròn)
            if len(contours_red) == 0:
                # If no red regions, try grayscale circular detection
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                circles = cv2.HoughCircles(
                    gray,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=50,
                    param1=50,
                    param2=30,
                    minRadius=20,
                    maxRadius=150
                )
                
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for circle in circles[0]:
                        x, y, r = circle
                        if cv2.contourArea(np.array([[x-r, y], [x+r, y], [x, y-r], [x, y+r]])) < min_area:
                            continue
                        
                        x1, y1 = max(0, x - r), max(0, y - r)
                        x2, y2 = min(image.shape[1], x + r), min(image.shape[0], y + r)
                        
                        stamp_region = image[y1:y2, x1:x2].copy()
                        
                        stamps.append({
                            "bbox": [x1, y1, x2, y2],
                            "area": int(np.pi * r ** 2),
                            "is_red": False,
                            "image": stamp_region,
                            "region_type": "circular"
                        })
            
            logger.info(f"Detected {len(stamps)} potential stamps")
            return stamps
        
        except Exception as e:
            logger.error(f"Error in stamp detection: {e}")
            return []

    def extract_stamp_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Phát hiện dấu mộc và trích xuất text từ chúng.
        
        Returns:
            [
                {
                    "text": "CÔNG TY ABC",
                    "confidence": 0.91,
                    "bbox": [x1, y1, x2, y2],
                    "region_type": "red_seal",
                    "best_angle": 0
                }
            ]
        """
        try:
            stamps = self.detect_stamps(image)
            stamp_texts = []
            
            for stamp in stamps:
                stamp_image = stamp["image"]
                
                # OCR từ nhiều angles để lấy kết quả tốt nhất
                ocr_result = self.ocr_service.extract_text_multi_angles(stamp_image)
                
                if ocr_result.get("text", "").strip():
                    stamp_texts.append({
                        "text": ocr_result["text"],
                        "confidence": ocr_result.get("confidence", 0.0),
                        "bbox": stamp["bbox"],
                        "region_type": stamp["region_type"],
                        "best_angle": ocr_result.get("best_angle", 0),
                        "area": stamp["area"]
                    })
            
            logger.info(f"Extracted text from {len(stamp_texts)} stamps")
            return stamp_texts
        
        except Exception as e:
            logger.error(f"Error extracting stamp text: {e}")
            return []

    def mark_stamps_on_image(self, image: np.ndarray, stamps: List[Dict]) -> np.ndarray:
        """
        Vẽ bounding box quanh các dấu được detect (để debug).
        """
        try:
            marked_image = image.copy()
            
            for stamp in stamps:
                x1, y1, x2, y2 = stamp["bbox"]
                color = (0, 0, 255) if stamp["is_red"] else (0, 255, 0)  # Red or Green
                cv2.rectangle(marked_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    marked_image,
                    stamp["region_type"],
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
            
            return marked_image
        
        except Exception as e:
            logger.error(f"Error marking stamps: {e}")
            return image
