# Chunk Strategy Benchmark Report

## Dataset

- data/uploads/session_10/Báo cáo tiến độ.docx

## Method

- Evaluate 12 configurations: chunk_size in [500, 1000, 1500, 2000], chunk_overlap in [50, 100, 200].
- Build chunks and FAISS index for each configuration.
- Create pseudo-queries from chunk content and check whether the originating chunk is retrieved.
- Report retrieval accuracy (Top-1, Top-3) as a proxy for RAG context precision.

## Results

| chunk_size | chunk_overlap | chunks | eval_queries | top1_accuracy | top3_accuracy |
|---:|---:|---:|---:|---:|---:|
| 1500 | 100 | 21 | 21 | 23.81% | 61.90% |
| 1000 | 200 | 37 | 30 | 40.00% | 56.67% |
| 1000 | 100 | 33 | 30 | 26.67% | 56.67% |
| 1000 | 50 | 32 | 30 | 23.33% | 53.33% |
| 500 | 50 | 72 | 30 | 23.33% | 50.00% |
| 500 | 100 | 73 | 30 | 20.00% | 50.00% |
| 500 | 200 | 88 | 30 | 20.00% | 50.00% |
| 2000 | 200 | 16 | 16 | 31.25% | 43.75% |
| 2000 | 100 | 15 | 15 | 20.00% | 40.00% |
| 1500 | 50 | 21 | 21 | 19.05% | 38.10% |
| 1500 | 200 | 22 | 22 | 27.27% | 36.36% |
| 2000 | 50 | 15 | 15 | 20.00% | 33.33% |

## Recommendation

- Recommended default: chunk_size=1500, chunk_overlap=100 (Top-1=23.81%, Top-3=61.90%).
- Keep Top-3 retrieval as default to reduce miss rate on multi-topic questions.

## Notes

- This benchmark measures retrieval accuracy, not final answer faithfulness.
- Re-run after uploading more representative documents for more stable conclusions.
