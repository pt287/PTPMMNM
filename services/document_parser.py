"""
Document Parser - xử lý PDF/DOCX với OCR fallback cho ảnh scan.
"""

import logging
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import cv2
import numpy as np
from pdf2image import convert_from_path, convert_from_bytes
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from docx.oxml.ns import qn
from docx.oxml import parse_xml

from .ocr_service import OCRService
from .stamp_detector import StampDetector

logger = logging.getLogger(__name__)


class DocumentParser:
    """Parse PDF/DOCX với support OCR."""
    
    def __init__(self):
        """Khởi tạo parser với OCR services."""
        self.ocr_service = OCRService()
        self.stamp_detector = StampDetector(self.ocr_service)

    def parse_pdf(self, pdf_path: str, use_ocr: bool = True, detect_stamps: bool = True) -> Dict[str, Any]:
        """
        Parse PDF file.
        
        Args:
            pdf_path: Đường dẫn PDF
            use_ocr: Dùng OCR nếu không extract được text
            detect_stamps: Phát hiện dấu mộc
            
        Returns:
            {
                "document_type": "pdf",
                "full_text": "...",
                "pages": [
                    {
                        "page_num": 1,
                        "text": "...",
                        "used_ocr": False,
                        "stamps": [...]
                    }
                ],
                "all_stamps": [...],
                "total_pages": 10
            }
        """
        try:
            logger.info(f"Parsing PDF: {pdf_path}")
            result = {
                "document_type": "pdf",
                "full_text": "",
                "pages": [],
                "all_stamps": [],
                "total_pages": 0,
                "metadata": {}
            }
            
            # Open with PyMuPDF
            doc = fitz.open(pdf_path)
            result["total_pages"] = len(doc)
            
            # Extract metadata
            if doc.metadata:
                result["metadata"] = doc.metadata
            
            page_texts = []
            all_stamps = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_result = {
                    "page_num": page_num + 1,
                    "text": "",
                    "used_ocr": False,
                    "stamps": [],
                    "image_count": 0
                }
                
                # Try to extract text from PDF text layer.
                text_layer = (page.get_text() or "").strip()
                merged_page_text = text_layer

                # Convert page to image once when OCR or stamp detection is enabled.
                image_bgr = None
                if use_ocr or detect_stamps:
                    try:
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)  # 2x zoom
                        image_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR) if pix.n >= 3 else image_array
                    except Exception as e:
                        logger.error(f"Failed rendering page {page_num+1} to image: {e}")

                # OCR path: fallback when no text layer, or augment when page has images.
                if use_ocr and image_bgr is not None:
                    image_infos = page.get_images(full=True)
                    page_result["image_count"] = len(image_infos)
                    should_ocr_page = (not text_layer) or bool(image_infos)
                    if should_ocr_page:
                        try:
                            ocr_result = self.ocr_service.extract_text(image_bgr)
                            ocr_text = (ocr_result.get("text") or "").strip()
                            if ocr_text:
                                if text_layer:
                                    merged_page_text = f"{text_layer}\n\n[OCR_IMAGE_TEXT]\n{ocr_text}"
                                else:
                                    merged_page_text = ocr_text
                                page_result["used_ocr"] = True
                                page_result["ocr_confidence"] = ocr_result.get("confidence", 0.0)
                        except Exception as e:
                            logger.error(f"OCR failed for page {page_num+1}: {e}")

                # Stamp detection runs independently from OCR fallback logic.
                if detect_stamps and image_bgr is not None:
                    try:
                        stamps = self.stamp_detector.extract_stamp_text(image_bgr)
                        page_result["stamps"] = stamps
                        all_stamps.extend(stamps)
                    except Exception as e:
                        logger.error(f"Stamp detection failed for page {page_num+1}: {e}")

                page_result["text"] = merged_page_text
                
                page_texts.append(page_result["text"])
                result["pages"].append(page_result)
            
            doc.close()
            
            # Merge all text
            result["full_text"] = "\n\n".join(filter(None, page_texts))
            result["all_stamps"] = all_stamps
            
            logger.info(f"PDF parsing completed: {result['total_pages']} pages, {len(all_stamps)} stamps detected")
            return result
        
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            return {
                "document_type": "pdf",
                "error": str(e),
                "full_text": "",
                "pages": [],
                "all_stamps": [],
                "total_pages": 0
            }

    def parse_docx(self, docx_path: str, use_ocr: bool = True, detect_stamps: bool = True) -> Dict[str, Any]:
        """
        Parse DOCX file.
        
        Args:
            docx_path: Đường dẫn DOCX
            use_ocr: Dùng OCR cho ảnh embedded
            detect_stamps: Phát hiện dấu
            
        Returns:
            {
                "document_type": "docx",
                "full_text": "...",
                "extracted_images": [
                    {
                        "image_num": 1,
                        "ocr_text": "...",
                        "stamps": [...]
                    }
                ],
                "all_stamps": [...]
            }
        """
        try:
            logger.info(f"Parsing DOCX: {docx_path}")
            result = {
                "document_type": "docx",
                "full_text": "",
                "extracted_images": [],
                "all_stamps": [],
                "metadata": {}
            }
            
            doc = DocxDocument(docx_path)
            
            # Extract text
            text_parts = []
            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)
            
            result["full_text"] = "\n".join(text_parts)
            
            # Extract images
            if use_ocr:
                images = self._extract_images_from_docx(docx_path)
                
                for img_idx, img_data in enumerate(images):
                    try:
                        # Convert bytes to numpy array
                        img_array = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                        
                        if img_array is None:
                            continue
                        
                        # OCR
                        ocr_result = self.ocr_service.extract_text(img_array)
                        
                        img_result = {
                            "image_num": img_idx + 1,
                            "ocr_text": ocr_result.get("text", ""),
                            "ocr_confidence": ocr_result.get("confidence", 0.0),
                            "stamps": []
                        }
                        
                        # Detect stamps
                        if detect_stamps:
                            stamps = self.stamp_detector.extract_stamp_text(img_array)
                            img_result["stamps"] = stamps
                            result["all_stamps"].extend(stamps)
                        
                        result["extracted_images"].append(img_result)
                        
                        # Add OCR text to full text
                        if ocr_result.get("text"):
                            text_parts.append(f"\n[Image {img_idx+1}]\n{ocr_result['text']}")
                    
                    except Exception as e:
                        logger.error(f"Error processing image {img_idx}: {e}")
                        continue
                
                result["full_text"] = "\n".join(text_parts)
            
            logger.info(f"DOCX parsing completed: {len(images)} images extracted")
            return result
        
        except Exception as e:
            logger.error(f"Error parsing DOCX: {e}")
            return {
                "document_type": "docx",
                "error": str(e),
                "full_text": "",
                "extracted_images": [],
                "all_stamps": []
            }

    def _extract_images_from_docx(self, docx_path: str) -> List[bytes]:
        """Extract all images từ DOCX file."""
        try:
            import zipfile
            
            images = []
            
            with zipfile.ZipFile(docx_path, 'r') as zip_ref:
                # List tất cả files trong media folder
                media_files = [f for f in zip_ref.namelist() if f.startswith('word/media/')]
                
                for media_file in media_files:
                    try:
                        img_data = zip_ref.read(media_file)
                        images.append(img_data)
                        logger.debug(f"Extracted image: {media_file}")
                    except Exception as e:
                        logger.warning(f"Failed to extract {media_file}: {e}")
            
            return images
        
        except Exception as e:
            logger.error(f"Error extracting images from DOCX: {e}")
            return []

    def parse_document(self, file_path: str, use_ocr: bool = True, detect_stamps: bool = True) -> Dict[str, Any]:
        """
        Parse bất kỳ file nào (PDF/DOCX).
        
        Returns: Dict với full_text, stamps, metadata
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == ".pdf":
            return self.parse_pdf(file_path, use_ocr=use_ocr, detect_stamps=detect_stamps)
        elif file_ext in [".docx", ".doc"]:
            return self.parse_docx(file_path, use_ocr=use_ocr, detect_stamps=detect_stamps)
        else:
            logger.error(f"Unsupported file type: {file_ext}")
            return {
                "error": f"Unsupported file type: {file_ext}",
                "full_text": "",
                "all_stamps": []
            }
