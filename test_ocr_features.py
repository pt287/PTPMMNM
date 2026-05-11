"""
Test script for OCR functionality in SmartDoc AI
"""

import sys
from pathlib import Path

# Test imports
try:
    from ocr_processor import OCRProcessor, OCRConfig, extract_ocr_text_from_files
    print("✅ ocr_processor imported successfully")
except ImportError as e:
    print(f"❌ Failed to import ocr_processor: {e}")
    sys.exit(1)

try:
    from rag_engine import RAGConfig, load_documents_from_files
    print("✅ rag_engine imported successfully")
except ImportError as e:
    print(f"❌ Failed to import rag_engine: {e}")
    sys.exit(1)

try:
    from backend import app, state
    print("✅ backend imported successfully")
except ImportError as e:
    print(f"❌ Failed to import backend: {e}")
    sys.exit(1)


def test_ocr_config():
    """Test OCR configuration"""
    print("\n🔍 Testing OCR Configuration...")
    
    config = OCRConfig()
    assert config.languages == ['vi', 'en', 'ch_sim'], "Default languages not set correctly"
    assert config.gpu == False, "GPU should be False by default"
    assert config.confidence_threshold == 0.3, "Confidence threshold not correct"
    print("✅ OCRConfig initialized correctly")
    
    # Test custom config
    custom_config = OCRConfig(
        languages=['en', 'vi'],
        gpu=False,
        confidence_threshold=0.5
    )
    assert custom_config.languages == ['en', 'vi'], "Custom languages not set"
    print("✅ Custom OCRConfig works")


def test_rag_config():
    """Test RAG configuration with OCR"""
    print("\n🔍 Testing RAG Configuration with OCR...")
    
    config = RAGConfig()
    assert config.use_ocr == True, "OCR should be enabled by default"
    assert config.ocr_languages == ['vi', 'en', 'ch_sim'], "Default OCR languages not set"
    assert config.ocr_gpu == False, "OCR GPU should be False by default"
    assert config.extract_images_only == False, "Extract images only should be False by default"
    print("✅ RAGConfig with OCR initialized correctly")
    
    # Test __post_init__
    config2 = RAGConfig(ocr_languages=None)
    assert config2.ocr_languages == ['vi', 'en', 'ch_sim'], "__post_init__ should set default languages"
    print("✅ RAGConfig.__post_init__() works correctly")


def test_app_state():
    """Test application state with OCR settings"""
    print("\n🔍 Testing App State with OCR Settings...")
    
    assert state.use_ocr == True, "State: OCR should be enabled by default"
    assert state.ocr_languages == ['vi', 'en', 'ch_sim'], "State: Default OCR languages not set"
    assert state.ocr_gpu == False, "State: OCR GPU should be False by default"
    assert state.ocr_confidence_threshold == 0.3, "State: Confidence threshold not correct"
    assert state.extract_images_only == False, "State: Extract images only should be False"
    print("✅ App state OCR settings initialized correctly")


def test_ocr_processor_methods():
    """Test OCR processor static methods exist"""
    print("\n🔍 Testing OCR Processor Methods...")
    
    # Check that methods exist
    assert hasattr(OCRProcessor, '_get_reader'), "OCRProcessor missing _get_reader"
    assert hasattr(OCRProcessor, 'extract_text_from_image'), "OCRProcessor missing extract_text_from_image"
    assert hasattr(OCRProcessor, '_enhance_image'), "OCRProcessor missing _enhance_image"
    assert hasattr(OCRProcessor, 'extract_images_from_pdf'), "OCRProcessor missing extract_images_from_pdf"
    assert hasattr(OCRProcessor, 'extract_images_from_docx'), "OCRProcessor missing extract_images_from_docx"
    assert hasattr(OCRProcessor, 'process_pdf_with_ocr'), "OCRProcessor missing process_pdf_with_ocr"
    assert hasattr(OCRProcessor, 'process_docx_with_ocr'), "OCRProcessor missing process_docx_with_ocr"
    print("✅ All OCR processor methods exist")


def test_load_documents_signature():
    """Test that load_documents_from_files accepts config parameter"""
    print("\n🔍 Testing load_documents_from_files Signature...")
    
    import inspect
    sig = inspect.signature(load_documents_from_files)
    params = sig.parameters
    
    assert 'config' in params, "load_documents_from_files missing config parameter"
    assert params['config'].default is None, "config parameter default should be None"
    print("✅ load_documents_from_files has correct signature")


def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 SmartDoc AI - OCR Feature Tests")
    print("=" * 60)
    
    try:
        test_ocr_config()
        test_rag_config()
        test_app_state()
        test_ocr_processor_methods()
        test_load_documents_signature()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        print("\n📝 Next Steps:")
        print("1. Install required libraries: pip install -r requirements.txt")
        print("2. Test with sample PDF: python test_reranking.py")
        print("3. Start backend: python -m uvicorn backend:app --reload")
        print("4. Upload a PDF with images to test OCR")
        print("\n📖 See OCR_FEATURES.md for detailed documentation")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
