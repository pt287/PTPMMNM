"""
Quick verification of OCR integration without installing heavy dependencies
"""

import sys
import ast
from pathlib import Path


def check_file_exists(filepath):
    """Check if a file exists"""
    path = Path(filepath)
    if path.exists():
        print(f"✅ {filepath}")
        return True
    else:
        print(f"❌ {filepath} - NOT FOUND")
        return False


def check_import_in_file(filepath, import_str):
    """Check if a specific import exists in file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if import_str in content:
                print(f"  ✅ Found: {import_str}")
                return True
            else:
                print(f"  ❌ Missing: {import_str}")
                return False
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        return False


def check_function_in_file(filepath, func_name):
    """Check if a function/class exists in file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if node.name == func_name:
                        print(f"  ✅ Found: {func_name}()")
                        return True
            print(f"  ❌ Missing: {func_name}()")
            return False
    except Exception as e:
        print(f"  ❌ Error parsing file: {e}")
        return False


def check_requirements():
    """Check if OCR libraries are in requirements.txt"""
    try:
        with open('requirements.txt', 'r') as f:
            content = f.read().lower()
            libs = ['rapidocr-onnxruntime', 'opencv-python', 'pypdfium2']
            results = []
            for lib in libs:
                if lib in content:
                    print(f"  ✅ {lib}")
                    results.append(True)
                else:
                    print(f"  ❌ {lib}")
                    results.append(False)
            return all(results)
    except Exception as e:
        print(f"  ❌ Error reading requirements.txt: {e}")
        return False


def main():
    print("=" * 70)
    print("🔍 SmartDoc AI - OCR Integration Verification")
    print("=" * 70)

    all_good = True

    # Check files
    print("\n📁 Checking new/updated files...")
    files = [
        'ocr_processor.py',
        'OCR_FEATURES.md',
        'OCR_IMPLEMENTATION_SUMMARY.md',
        'test_ocr_features.py'
    ]
    for file in files:
        if not check_file_exists(file):
            all_good = False

    # Check rag_engine.py updates
    print("\n📝 Checking rag_engine.py updates...")
    print("  Imports:")
    if not check_import_in_file('rag_engine.py', 'from ocr_processor import'):
        all_good = False
    
    print("  RAGConfig class:")
    if not check_function_in_file('rag_engine.py', 'RAGConfig'):
        all_good = False
    
    print("  OCR configuration fields:")
    with open('rag_engine.py', 'r', encoding='utf-8') as f:
        content = f.read()
        fields = ['use_ocr:', 'ocr_languages:', 'ocr_gpu:', 'ocr_confidence_threshold:', 'extract_images_only:']
        for field in fields:
            if field in content:
                print(f"  ✅ {field}")
            else:
                print(f"  ❌ {field}")
                all_good = False

    # Check backend.py updates
    print("\n🔧 Checking backend.py updates...")
    print("  AppState OCR fields:")
    with open('backend.py', 'r', encoding='utf-8') as f:
        content = f.read()
        fields = ['self.use_ocr', 'self.ocr_languages', 'self.ocr_gpu', 'self.ocr_confidence_threshold']
        for field in fields:
            if field in content:
                print(f"  ✅ {field}")
            else:
                print(f"  ❌ {field}")
                all_good = False

    # Check requirements.txt
    print("\n📦 Checking requirements.txt...")
    if not check_requirements():
        all_good = False

    # Check ocr_processor.py
    print("\n🎨 Checking ocr_processor.py...")
    print("  Classes:")
    if not check_function_in_file('ocr_processor.py', 'OCRConfig'):
        all_good = False
    if not check_function_in_file('ocr_processor.py', 'OCRProcessor'):
        all_good = False
    
    print("  Key methods:")
    methods = [
        'extract_images_from_pdf',
        'extract_images_from_docx',
        'extract_text_from_image',
        'process_pdf_with_ocr',
        'process_docx_with_ocr'
    ]
    for method in methods:
        if not check_function_in_file('ocr_processor.py', method):
            all_good = False

    # Summary
    print("\n" + "=" * 70)
    if all_good:
        print("✅ All integration checks passed!")
        print("\n📋 Next Steps:")
        print("  1. Install OCR libraries:")
        print("     pip install rapidocr-onnxruntime opencv-python")
        print("")
        print("  2. Run tests:")
        print("     python test_ocr_features.py")
        print("")
        print("  3. Start backend:")
        print("     python -m uvicorn backend:app --reload")
        print("")
        print("  4. Upload PDF/DOCX with images to test OCR")
        print("\n📖 Documentation:")
        print("  - OCR_FEATURES.md - Detailed OCR documentation")
        print("  - OCR_IMPLEMENTATION_SUMMARY.md - Implementation overview")
    else:
        print("❌ Some checks failed. Please review the output above.")
    print("=" * 70)

    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
