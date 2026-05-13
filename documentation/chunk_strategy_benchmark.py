from __future__ import annotations

import argparse
import itertools
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_engine import RAGConfig, build_vectorstore, load_documents_from_files, split_documents


CHUNK_SIZES = [500, 1000, 1500, 2000]
CHUNK_OVERLAPS = [50, 100, 200]


@dataclass
class EvalQuery:
    question: str
    target_chunk_id: int


@dataclass
class EvalResult:
    chunk_size: int
    chunk_overlap: int
    chunk_count: int
    query_count: int
    top1_accuracy: float
    top3_accuracy: float


def collect_supported_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.suffix.lower() in {".pdf", ".docx"} and path.is_file():
            files.append(path)
    return sorted(files)


def build_eval_queries(chunks, max_queries: int = 30) -> List[EvalQuery]:
    queries: List[EvalQuery] = []
    for chunk in chunks:
        content = (chunk.page_content or "").strip()
        if len(content) < 120:
            continue

        words = re.findall(r"[0-9A-Za-z_\u00C0-\u1EF9]{4,}", content)
        if len(words) < 12:
            continue

        start = max(0, len(words) // 3)
        end = min(len(words), start + 9)
        phrase = " ".join(words[start:end]).strip()
        if not phrase:
            continue

        target_chunk_id = int(chunk.metadata.get("chunk_id", -1))
        if target_chunk_id <= 0:
            continue

        queries.append(EvalQuery(question=phrase, target_chunk_id=target_chunk_id))
        if len(queries) >= max_queries:
            break

    return queries


def evaluate_config(file_items: Sequence[Tuple[str, bytes]], chunk_size: int, chunk_overlap: int) -> EvalResult:
    config = RAGConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = load_documents_from_files(file_items)
    chunks = split_documents(documents, config)

    queries = build_eval_queries(chunks)
    if not queries:
        return EvalResult(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_count=len(chunks),
            query_count=0,
            top1_accuracy=0.0,
            top3_accuracy=0.0,
        )

    vectorstore = build_vectorstore(chunks, config)

    top1_hits = 0
    top3_hits = 0
    for query in queries:
        retrieved = vectorstore.similarity_search(query.question, k=3)
        retrieved_ids = [int((doc.metadata or {}).get("chunk_id", -1)) for doc in retrieved]
        if retrieved_ids and retrieved_ids[0] == query.target_chunk_id:
            top1_hits += 1
        if query.target_chunk_id in retrieved_ids:
            top3_hits += 1

    total = len(queries)
    return EvalResult(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_count=len(chunks),
        query_count=total,
        top1_accuracy=top1_hits / total,
        top3_accuracy=top3_hits / total,
    )


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_report(results: Sequence[EvalResult], source_files: Sequence[Path]) -> str:
    ordered = sorted(results, key=lambda x: (x.top3_accuracy, x.top1_accuracy), reverse=True)
    best = ordered[0] if ordered else None

    lines: List[str] = []
    lines.append("# Chunk Strategy Benchmark Report")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    for file in source_files:
        lines.append(f"- {file.as_posix()}")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- Evaluate 12 configurations: chunk_size in [500, 1000, 1500, 2000], chunk_overlap in [50, 100, 200].")
    lines.append("- Build chunks and FAISS index for each configuration.")
    lines.append("- Create pseudo-queries from chunk content and check whether the originating chunk is retrieved.")
    lines.append("- Report retrieval accuracy (Top-1, Top-3) as a proxy for RAG context precision.")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| chunk_size | chunk_overlap | chunks | eval_queries | top1_accuracy | top3_accuracy |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in ordered:
        lines.append(
            "| "
            f"{row.chunk_size} | {row.chunk_overlap} | {row.chunk_count} | {row.query_count} | "
            f"{format_percent(row.top1_accuracy)} | {format_percent(row.top3_accuracy)} |"
        )

    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    if best is None:
        lines.append("- No valid evaluation data was produced.")
    else:
        lines.append(
            f"- Recommended default: chunk_size={best.chunk_size}, chunk_overlap={best.chunk_overlap} "
            f"(Top-1={format_percent(best.top1_accuracy)}, Top-3={format_percent(best.top3_accuracy)})."
        )
        lines.append("- Keep Top-3 retrieval as default to reduce miss rate on multi-topic questions.")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This benchmark measures retrieval accuracy, not final answer faithfulness.")
    lines.append("- Re-run after uploading more representative documents for more stable conclusions.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark chunk strategy for SmartDoc RAG retrieval.")
    parser.add_argument(
        "--uploads-dir",
        default="data/uploads",
        help="Directory containing uploaded files (PDF/DOCX).",
    )
    parser.add_argument(
        "--output",
        default="documentation/chunk_strategy_report.md",
        help="Output markdown report path.",
    )
    args = parser.parse_args()

    uploads_dir = Path(args.uploads_dir)
    files = collect_supported_files(uploads_dir)
    if not files:
        print(f"No PDF/DOCX files found in: {uploads_dir}")
        return 1

    file_items = [(path.name, path.read_bytes()) for path in files]

    results: List[EvalResult] = []
    for chunk_size, chunk_overlap in itertools.product(CHUNK_SIZES, CHUNK_OVERLAPS):
        if chunk_overlap >= chunk_size:
            continue
        result = evaluate_config(file_items, chunk_size, chunk_overlap)
        print(
            f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, "
            f"top1={format_percent(result.top1_accuracy)}, top3={format_percent(result.top3_accuracy)}"
        )
        results.append(result)

    report = build_report(results, files)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Report written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
