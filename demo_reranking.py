"""
Demo script for Cross-Encoder Re-ranking in SmartDoc AI RAG system.

This script demonstrates:
1. Bi-encoder (FAISS only) retrieval
2. Cross-encoder re-ranking
3. Performance comparison
4. Quality comparison
"""

import time
from pathlib import Path
from typing import List, Tuple

from rag_engine import (
    RAGConfig,
    build_rag_pipeline,
    compare_retrieval_methods,
    ask_question,
)


def print_separator(title: str = "") -> None:
    """Print a formatted separator."""
    if title:
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}\n")
    else:
        print(f"{'=' * 80}\n")


def print_documents(docs: List, title: str, scores: List[float] = None) -> None:
    """Print retrieved documents with optional scores."""
    print(f"\n{title}:")
    print("-" * 80)
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata or {}
        preview = doc.page_content[:150].replace("\n", " ")
        score_str = f" (score: {scores[i-1]:.4f})" if scores else ""
        print(f"{i}. [Chunk {metadata.get('chunk_id', '?')}]{score_str}")
        print(f"   {preview}...")
        print()


def demo_basic_reranking(file_path: str, query: str) -> None:
    """Demonstrate basic re-ranking functionality."""
    print_separator("DEMO 1: Basic Re-ranking")

    # Read the file
    with open(file_path, "rb") as f:
        file_content = f.read()

    file_items = [(Path(file_path).name, file_content)]

    # Configuration WITHOUT re-ranking (bi-encoder only)
    print("Building RAG pipeline WITHOUT re-ranking...")
    config_no_rerank = RAGConfig(
        embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        ollama_model="qwen2.5:0.5b",
        chunk_size=1000,
        chunk_overlap=100,
        top_k=3,
        use_reranking=False,
    )

    qa_chain_no_rerank, _, _ = build_rag_pipeline(file_items, config_no_rerank)

    # Configuration WITH re-ranking
    print("Building RAG pipeline WITH re-ranking...")
    config_with_rerank = RAGConfig(
        embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        ollama_model="qwen2.5:0.5b",
        chunk_size=1000,
        chunk_overlap=100,
        top_k=3,
        use_reranking=True,
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_top_n=10,  # Retrieve 10 candidates, re-rank to top 3
    )

    qa_chain_with_rerank, doc_language, chunk_count = build_rag_pipeline(
        file_items, config_with_rerank
    )

    print(f"✓ Document language: {doc_language}")
    print(f"✓ Total chunks: {chunk_count}")
    print(f"\nQuery: {query}\n")

    # Ask question WITHOUT re-ranking
    print("WITHOUT Re-ranking (Bi-encoder only):")
    start = time.time()
    answer_no_rerank, sources_no_rerank = ask_question(qa_chain_no_rerank, query)
    time_no_rerank = time.time() - start
    print(f"Time: {time_no_rerank:.3f}s")
    print(f"Answer: {answer_no_rerank}")
    print(f"Sources: {len(sources_no_rerank)} chunks")

    # Ask question WITH re-ranking
    print("\nWITH Re-ranking (Bi-encoder + Cross-encoder):")
    start = time.time()
    answer_with_rerank, sources_with_rerank = ask_question(qa_chain_with_rerank, query)
    time_with_rerank = time.time() - start
    print(f"Time: {time_with_rerank:.3f}s")
    print(f"Answer: {answer_with_rerank}")
    print(f"Sources: {len(sources_with_rerank)} chunks")

    print(f"\nTime difference: {abs(time_with_rerank - time_no_rerank):.3f}s")
    print(f"Overhead: {((time_with_rerank / time_no_rerank - 1) * 100):.1f}%")


def demo_retrieval_comparison(file_path: str, query: str) -> None:
    """Compare bi-encoder vs cross-encoder retrieval methods."""
    print_separator("DEMO 2: Detailed Retrieval Comparison")

    # Read the file
    with open(file_path, "rb") as f:
        file_content = f.read()

    file_items = [(Path(file_path).name, file_content)]

    # Build pipeline with re-ranking enabled
    config = RAGConfig(
        embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        chunk_size=1000,
        chunk_overlap=100,
        top_k=3,
        use_reranking=True,
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_top_n=10,
    )

    qa_chain, _, _ = build_rag_pipeline(file_items, config)
    vectorstore = qa_chain.retriever.vectorstore

    print(f"Query: {query}\n")

    # Compare retrieval methods
    comparison = compare_retrieval_methods(vectorstore, query, config)

    # Print bi-encoder results
    print("BI-ENCODER (FAISS Only):")
    print(f"  Time: {comparison['bi_encoder']['time_ms']:.2f}ms")
    print(f"  Documents retrieved: {comparison['bi_encoder']['num_docs']}")
    print_documents(comparison["bi_encoder"]["docs"], "Top documents")

    # Print cross-encoder results
    print("\nCROSS-ENCODER (Re-ranking):")
    print(f"  Candidates retrieved: {comparison['cross_encoder']['num_candidates']}")
    print(f"  Retrieval time: {comparison['cross_encoder']['retrieval_time_ms']:.2f}ms")
    print(f"  Re-ranking time: {comparison['cross_encoder']['rerank_time_ms']:.2f}ms")
    print(f"  Total time: {comparison['cross_encoder']['time_ms']:.2f}ms")
    print(f"  Final documents: {comparison['cross_encoder']['num_final']}")
    print_documents(
        comparison["cross_encoder"]["docs"],
        "Top re-ranked documents",
        comparison["cross_encoder"]["scores"],
    )

    # Print comparison summary
    print("\nCOMPARISON SUMMARY:")
    print(f"  Total time: {comparison['comparison']['total_time_ms']:.2f}ms")
    print(f"  Overhead: {comparison['comparison']['overhead_ms']:.2f}ms")
    print(
        f"  Speed ratio: Bi-encoder is {comparison['comparison']['speedup']:.2f}x faster"
    )
    print(
        f"  Trade-off: Cross-encoder adds {comparison['comparison']['overhead_ms']:.0f}ms for better relevance"
    )


def demo_model_comparison() -> None:
    """Compare different cross-encoder models."""
    print_separator("DEMO 3: Cross-Encoder Model Comparison")

    models = [
        {
            "name": "ms-marco-MiniLM-L-6-v2",
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "description": "Fast, lightweight (80MB), good for English",
        },
        {
            "name": "ms-marco-MiniLM-L-12-v2",
            "model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "description": "Better accuracy, medium size (130MB)",
        },
        {
            "name": "mmarco-mMiniLMv2-L12-H384-v1",
            "model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            "description": "Multilingual (supports Vietnamese), larger (470MB)",
        },
    ]

    print("Available Cross-Encoder Models for Re-ranking:\n")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model['name']}")
        print(f"   Model ID: {model['model']}")
        print(f"   Description: {model['description']}")
        print()

    print("Recommendation:")
    print("  - For English documents: ms-marco-MiniLM-L-6-v2 (fast)")
    print("  - For Vietnamese documents: mmarco-mMiniLMv2-L12-H384-v1 (multilingual)")
    print("  - For best accuracy: ms-marco-MiniLM-L-12-v2")


def demo_latency_optimization() -> None:
    """Show latency optimization strategies."""
    print_separator("DEMO 4: Latency Optimization Strategies")

    strategies = [
        {
            "strategy": "1. Adjust rerank_top_n",
            "description": "Retrieve fewer candidates (e.g., 5-10 instead of 20)",
            "impact": "Lower latency, may reduce quality slightly",
            "example": "rerank_top_n=5  # Faster but less comprehensive",
        },
        {
            "strategy": "2. Use lighter cross-encoder",
            "description": "Choose ms-marco-MiniLM-L-6-v2 instead of L-12",
            "impact": "~40% faster re-ranking",
            "example": "reranker_model='cross-encoder/ms-marco-MiniLM-L-6-v2'",
        },
        {
            "strategy": "3. Batch processing",
            "description": "Cross-encoder already batches predictions internally",
            "impact": "Already optimized in current implementation",
            "example": "# Handled automatically by sentence-transformers",
        },
        {
            "strategy": "4. Model caching",
            "description": "Re-use loaded models with @lru_cache",
            "impact": "Faster on subsequent queries (already implemented)",
            "example": "@lru_cache(maxsize=2) on _get_cross_encoder()",
        },
        {
            "strategy": "5. Selective re-ranking",
            "description": "Only use re-ranking for complex/ambiguous queries",
            "impact": "Best of both worlds: speed + quality when needed",
            "example": "use_reranking = query_complexity > threshold",
        },
    ]

    print("Optimization Strategies:\n")
    for strategy in strategies:
        print(f"{strategy['strategy']}")
        print(f"  Description: {strategy['description']}")
        print(f"  Impact: {strategy['impact']}")
        print(f"  Example: {strategy['example']}")
        print()

    print("Typical Performance Numbers:")
    print("  Bi-encoder retrieval: 10-50ms")
    print("  Cross-encoder re-ranking (10 docs): 50-150ms")
    print("  Total overhead: 60-200ms")
    print("\nConclusion: Re-ranking adds ~100ms but significantly improves relevance")


if __name__ == "__main__":
    print_separator("SmartDoc AI - Cross-Encoder Re-ranking Demo")

    # Check if sample file exists
    sample_file = "sample.pdf"  # Replace with your test file
    test_query = "What is the main topic of the document?"

    print("This demo showcases Cross-Encoder re-ranking capabilities.\n")
    print("To run the demos, you need:")
    print("  1. A sample PDF file")
    print("  2. Ollama running with a model (e.g., qwen2.5:0.5b)")
    print("  3. Internet connection (first time to download cross-encoder models)")
    print()

    # Run demos
    demo_model_comparison()
    demo_latency_optimization()

    # Uncomment to run live demos with your file
    # if Path(sample_file).exists():
    #     demo_basic_reranking(sample_file, test_query)
    #     demo_retrieval_comparison(sample_file, test_query)
    # else:
    #     print(f"\nSkipping live demos - {sample_file} not found")
    #     print(f"Create a sample PDF and update the 'sample_file' variable to run live demos")

    print_separator("Demo Complete")
    print("\nNext Steps:")
    print("  1. Install: pip install -r requirements.txt")
    print("  2. Update backend.py and app.py to expose re-ranking options")
    print("  3. Test with your own documents")
    print("  4. Monitor performance and adjust rerank_top_n as needed")
