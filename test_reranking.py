"""
Quick test script to verify Cross-Encoder re-ranking implementation.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")

    try:
        from sentence_transformers import CrossEncoder
        print("✓ sentence-transformers imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import sentence-transformers: {e}")
        print("  Fix: pip install -r requirements.txt")
        return False

    try:
        from rag_engine import (
            RAGConfig,
            RerankingRetriever,
            rerank_documents,
            compare_retrieval_methods,
        )
        print("✓ rag_engine imports successful")
    except ImportError as e:
        print(f"✗ Failed to import from rag_engine: {e}")
        return False

    return True


def test_config():
    """Test RAGConfig with re-ranking parameters."""
    print("\nTesting RAGConfig...")

    try:
        from rag_engine import RAGConfig

        # Test default config
        config1 = RAGConfig()
        assert config1.use_reranking == False
        assert config1.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert config1.rerank_top_n == 10
        print("✓ Default config works")

        # Test custom config
        config2 = RAGConfig(
            use_reranking=True,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
            rerank_top_n=15,
        )
        assert config2.use_reranking == True
        assert config2.reranker_model == "cross-encoder/ms-marco-MiniLM-L-12-v2"
        assert config2.rerank_top_n == 15
        print("✓ Custom config works")

        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


def test_cross_encoder_loading():
    """Test loading a cross-encoder model."""
    print("\nTesting cross-encoder model loading...")

    try:
        from rag_engine import _get_cross_encoder

        # Load a small model for testing
        model = _get_cross_encoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("✓ Cross-encoder model loaded successfully")

        # Test prediction
        scores = model.predict([
            ["What is Python?", "Python is a programming language."],
            ["What is Python?", "The sky is blue."],
        ])

        assert len(scores) == 2
        assert scores[0] > scores[1], "First pair should be more relevant"
        print(f"✓ Model prediction works (scores: {scores[0]:.3f} vs {scores[1]:.3f})")

        return True
    except Exception as e:
        print(f"✗ Cross-encoder test failed: {e}")
        print("  Note: First run may take time to download model (~80MB)")
        return False


def test_rerank_function():
    """Test the rerank_documents function."""
    print("\nTesting rerank_documents function...")

    try:
        from langchain_core.documents import Document
        from rag_engine import RAGConfig, rerank_documents

        # Create test documents
        docs = [
            Document(page_content="Python is a programming language.", metadata={"id": 1}),
            Document(page_content="The sky is blue and beautiful.", metadata={"id": 2}),
            Document(page_content="Python is used for web development.", metadata={"id": 3}),
        ]

        query = "What is Python?"
        config = RAGConfig(
            use_reranking=True,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_k=2,
        )

        # Test without scores
        reranked = rerank_documents(query, docs, config, return_scores=False)
        assert len(reranked) == 2, "Should return top_k=2 documents"
        print(f"✓ Re-ranking works (returned {len(reranked)} docs)")

        # Test with scores
        reranked_with_scores, scores = rerank_documents(query, docs, config, return_scores=True)
        assert len(scores) == 2
        assert scores[0] >= scores[1], "Scores should be sorted descending"
        print(f"✓ Re-ranking with scores works (scores: {scores[0]:.3f}, {scores[1]:.3f})")

        # Check that Python-related docs are ranked higher
        top_doc = reranked_with_scores[0]
        assert "Python" in top_doc.page_content, "Top doc should be about Python"
        print("✓ Relevance ranking is correct")

        return True
    except Exception as e:
        print(f"✗ Re-ranking function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backend_integration():
    """Test backend integration (imports only, no server required)."""
    print("\nTesting backend integration...")

    try:
        from backend import AppState

        state = AppState()
        assert hasattr(state, 'use_reranking')
        assert hasattr(state, 'reranker_model')
        assert hasattr(state, 'rerank_top_n')
        print("✓ Backend AppState has re-ranking attributes")

        return True
    except Exception as e:
        print(f"✗ Backend integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("Cross-Encoder Re-ranking Implementation Test")
    print("=" * 70)

    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("Cross-Encoder Loading", test_cross_encoder_loading),
        ("Re-rank Function", test_rerank_function),
        ("Backend Integration", test_backend_integration),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Re-ranking implementation is ready.")
        print("\nNext steps:")
        print("  1. Run: python demo_reranking.py")
        print("  2. Start backend: uvicorn backend:app --reload")
        print("  3. Test API: curl http://localhost:8000/api/health")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
