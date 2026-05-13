"""
OCR Service sử dụng EasyOCR cho Tiếng Việt và Anh.
Xử lý ảnh scan tài liệu với preprocessing.
"""

import logging
import cv2
import numpy as np
import easyocr
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class OCRService:
    """Dịch vụ OCR cho ảnh tài liệu với preprocessing."""
    
    def __init__(self):
        """Khởi tạo EasyOCR Reader cho Việt và Anh."""
        try:
            logger.info("Initializing EasyOCR Reader...")
            self.reader = easyocr.Reader(['vi', 'en'], gpu=False)
            logger.info("EasyOCR Reader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Xử lý ảnh trước OCR.
        
        - Grayscale
        - Denoise
        - Threshold
        - Sharpen
        - Auto-rotate
        """
        try:
            # Convert to grayscale nếu chưa
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
            
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Adaptive threshold
            thresholded = cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
            
            # Sharpen
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]]) / 1.0
            sharpened = cv2.filter2D(thresholded, -1, kernel)
            
            # Auto-rotate nếu cần (dùng Hough line detection)
            rotated = self._auto_rotate(sharpened)
            
            # Upscale nếu ảnh quá nhỏ
            height, width = rotated.shape
            if height < 200 or width < 200:
                scale = max(200.0 / height, 200.0 / width)
                rotated = cv2.resize(rotated, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            return rotated
        
        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            return image  # Fallback: return original

    def _auto_rotate(self, image: np.ndarray) -> np.ndarray:
        """Tự động rotate ảnh nếu cần."""
        try:
            # Detect edges
            edges = cv2.Canny(image, 50, 150)
            
            # Detect lines
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
            
            if lines is None or len(lines) == 0:
                return image
            
            # Tính angle
            angles = []
            for line in lines[:10]:  # Take first 10 lines
                rho, theta = line[0]
                angle = theta * 180 / np.pi
                angles.append(angle)
            
            # Median angle
            if angles:
                median_angle = np.median(angles)
                # Normalize to [-45, 45]
                if median_angle > 45:
                    median_angle -= 90
                elif median_angle < -45:
                    median_angle += 90
                
                # Rotate nếu angle > 2 degrees
                if abs(median_angle) > 2:
                    h, w = image.shape
                    center = (w // 2, h // 2)
                    matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
                    logger.debug(f"Auto-rotated image by {median_angle:.1f} degrees")
                    return rotated
            
            return image
        
        except Exception as e:
            logger.warning(f"Auto-rotate failed: {e}")
            return image

    def extract_text(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Trích xuất text từ ảnh.
        
        Args:
            image: numpy array ảnh
            
        Returns:
            {
                "text": "...",
                "confidence": 0.95,
                "details": [...raw results...]
            }
        """
        try:
            # Preprocess
            preprocessed = self._preprocess_image(image)
            
            # OCR
            results = self.reader.readtext(preprocessed, detail=1)
            
            if not results:
                logger.warning("No text detected in image")
                return {
                    "text": "",
                    "confidence": 0.0,
                    "details": []
                }
            
            # Merge results
            full_text_lines = []
            confidences = []
            details = []
            
            for (bbox, text, conf) in results:
                full_text_lines.append(text)
                confidences.append(conf)
                details.append({
                    "text": text,
                    "confidence": float(conf),
                    "bbox": [[float(x), float(y)] for x, y in bbox]
                })
            
            full_text = " ".join(full_text_lines)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            logger.info(f"OCR extracted {len(full_text_lines)} text segments, avg confidence: {avg_confidence:.2f}")
            
            return {
                "text": full_text,
                "confidence": float(avg_confidence),
                "details": details
            }
        
        except Exception as e:
            logger.error(f"Error in text extraction: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "details": [],
                "error": str(e)
            }

    def extract_text_from_file(self, image_path: str) -> Dict[str, Any]:
        """Trích xuất text từ file ảnh."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            result = self.extract_text(image)
            result["source"] = image_path
            return result
        
        except Exception as e:
            logger.error(f"Error reading image file {image_path}: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "details": [],
                "error": str(e),
                "source": image_path
            }

    def extract_text_multi_angles(self, image: np.ndarray) -> Dict[str, Any]:
        """
        OCR từ nhiều góc xoay, lấy kết quả confidence cao nhất.
        Hữu ích cho dấu mộc/con dấu.
        """
        try:
            best_result = None
            best_confidence = 0.0
            
            for angle in [0, 90, 180, 270]:
                if angle != 0:
                    h, w = image.shape[:2]
                    center = (w // 2, h // 2)
                    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(image, matrix, (w, h))
                else:
                    rotated = image.copy()
                
                result = self.extract_text(rotated)
                conf = result.get("confidence", 0.0)
                
                if conf > best_confidence:
                    best_confidence = conf
                    best_result = result
                    best_result["best_angle"] = angle
            
            if best_result:
                return best_result
            else:
                return self.extract_text(image)
        
        except Exception as e:
            logger.error(f"Error in multi-angle OCR: {e}")
            return self.extract_text(image)
