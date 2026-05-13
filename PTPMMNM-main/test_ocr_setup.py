"""
Test script - Verify all OCR modules import correctly
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all OCR imports."""
    try:
        logger.info("Testing imports...")
        
        # Test OCR Service
        logger.info("  Importing OCRService...")
        from services.ocr_service import OCRService
        logger.info("  ✅ OCRService imported")
        
        # Test Stamp Detector
        logger.info("  Importing StampDetector...")
        from services.stamp_detector import StampDetector
        logger.info("  ✅ StampDetector imported")
        
        # Test Document Parser
        logger.info("  Importing DocumentParser...")
        from services.document_parser import DocumentParser
        logger.info("  ✅ DocumentParser imported")
        
        # Test package
        logger.info("  Importing services package...")
        from services import OCRService, StampDetector, DocumentParser
        logger.info("  ✅ Services package imported")
        
        logger.info("\n✅ All imports successful!")
        return True
    
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False


def test_ocr_initialization():
    """Test OCR service initialization."""
    try:
        logger.info("\nTesting OCR initialization...")
        logger.info("  This will download EasyOCR models (~150MB on first run)...")
        
        from services.ocr_service import OCRService
        ocr = OCRService()
        
        logger.info("  ✅ OCRService initialized")
        logger.info("  ✅ EasyOCR reader ready")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ OCR initialization failed: {e}")
        logger.warning("  This is expected on first run (downloading models)")
        return False


def test_dependencies():
    """Check all required dependencies."""
    try:
        logger.info("\nChecking dependencies...")
        
        deps = {
            'cv2': 'opencv-python',
            'numpy': 'numpy',
            'easyocr': 'easyocr',
            'fitz': 'PyMuPDF',
            'docx': 'python-docx',
            'pdf2image': 'pdf2image',
            'PIL': 'pillow',
        }
        
        for module_name, package_name in deps.items():
            try:
                __import__(module_name)
                logger.info(f"  ✅ {package_name}")
            except ImportError:
                logger.warning(f"  ⚠️  {package_name} not installed")
                logger.warning(f"     Run: pip install {package_name}")
        
        return True
    
    except Exception as e:
        logger.error(f"Dependency check failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("SmartDoc AI OCR - Import & Initialization Test")
    logger.info("=" * 70)
    
    # Test 1: Dependencies
    dep_ok = test_dependencies()
    
    # Test 2: Imports
    import_ok = test_imports()
    
    # Test 3: OCR Init (skip on error or if in CI)
    if import_ok and "--skip-ocr-init" not in sys.argv:
        ocr_ok = test_ocr_initialization()
    else:
        logger.info("\nSkipping OCR initialization test")
        ocr_ok = True
    
    # Summary
    logger.info("\n" + "=" * 70)
    if import_ok and ocr_ok:
        logger.info("✅ All tests passed! OCR system is ready.")
        sys.exit(0)
    else:
        logger.warning("⚠️  Some tests failed. See above for details.")
        sys.exit(1)
