# Ghi chú ôn vấn đáp RAG System với LLMs

Tài liệu này tóm tắt các ý quan trọng để trả lời ngắn gọn, đúng trọng tâm khi giáo viên hỏi về các tính năng của hệ thống.

## 12) Ôn tập theo cấu trúc chuẩn (nội dung cơ bản + 10 câu 8.2.x)

### A. Nội dung cơ bản của ứng dụng

1) Tên hàm: `buildIndex` (`frontend/app.js`)  
Số dòng: 1002  
Chức năng: Thu thập file + cấu hình (model, chunk, graph/rerank/self-rag), gọi API build index, cập nhật trạng thái hệ thống và lịch sử phiên.  
Câu hỏi có thể hỏi: "Luồng build index từ frontend sang backend diễn ra thế nào?"  
Câu trả lời: "Frontend gửi `FormData` lên `/api/build-index`, backend xử lý tách chunk + tạo vector store + lưu session, rồi trả lại metadata để UI cập nhật."

2) Tên hàm: `askQuestion` (`frontend/app.js`)  
Số dòng: 1072  
Chức năng: Gửi câu hỏi lên `/api/ask`, hiển thị câu trả lời, nguồn trích dẫn và metadata Self-RAG (nếu có).  
Câu hỏi có thể hỏi: "Tại sao phần trả lời hiển thị được citation ngay trong chat?"  
Câu trả lời: "Vì backend trả về `sources` theo từng chunk, frontend render thành danh sách nguồn và cho mở context chi tiết."

3) Tên hàm: `addMessage` (`frontend/app.js`)  
Số dòng: 491  
Chức năng: Render tin nhắn user/assistant và dựng khối nguồn tham chiếu trong mỗi câu trả lời.  
Câu hỏi có thể hỏi: "Phần nguồn hiển thị những gì?"  
Câu trả lời: "Hiển thị `chunk_id`, tên file, trang, vị trí và preview để người dùng kiểm chứng câu trả lời."

4) Tên hàm: `openCitationModal` (`frontend/app.js`)  
Số dòng: 582  
Chức năng: Mở modal xem context gốc của citation, hiển thị trang/vị trí và đoạn highlight.  
Câu hỏi có thể hỏi: "Lợi ích của modal context là gì?"  
Câu trả lời: "Giúp người dùng xác minh trực tiếp đoạn tài liệu được dùng để trả lời, tăng tính minh bạch."

5) Tên hàm: `history` + `session_history` + `activate_session` (`backend.py`)  
Số dòng: 637, 646, 683  
Chức năng: Cung cấp lịch sử upload/hỏi đáp theo session và cho phép kích hoạt lại phiên cũ để hỏi tiếp.  
Câu hỏi có thể hỏi: "Session có ý nghĩa gì trong ứng dụng này?"  
Câu trả lời: "Session gom cấu hình index + lịch sử Q&A theo từng lần upload để có thể mở lại đúng ngữ cảnh làm việc."

### B. 10 câu hỏi 8.2.x theo cấu trúc hàm

1) Câu 8.2.1 - Thêm hỗ trợ file DOCX  
Tên hàm: `load_documents_from_files` + `_load_docx_with_python_docx` (`rag_engine.py`)  
Số dòng: 167, 231  
Chức năng: Nhận nhiều file PDF/DOCX, dùng `PDFPlumberLoader` cho PDF và `python-docx` cho DOCX, trích xuất paragraph + table rồi chuẩn hóa text.  
Câu hỏi có thể hỏi: "Vì sao chọn python-docx thay vì đọc DOCX như text thường?"  
Câu trả lời: "DOCX có cấu trúc tài liệu (paragraph, table), nên phải parse đúng cấu trúc để không mất nội dung quan trọng."

2) Câu 8.2.2 - Lưu trữ lịch sử hội thoại  
Tên hàm: `_store_qa` + `session_history` + `renderSessionQa`  
Số dòng: 275 (`backend.py`), 646 (`backend.py`), 690 (`frontend/app.js`)  
Chức năng: Lưu Q&A vào SQLite theo `session_id`, đọc lại qua API lịch sử, hiển thị trong sidebar để xem lại câu hỏi đã hỏi.  
Câu hỏi có thể hỏi: "Dữ liệu nào cần lưu để khôi phục hội thoại?"  
Câu trả lời: "Tối thiểu gồm `session_id`, question, answer, sources, response_time, created_at để hiển thị lại đầy đủ."

3) Câu 8.2.3 - Thêm nút xóa lịch sử  
Tên hàm: `clearHistory` + `clearVectorStore` + `clear_history` + `clear_vector_store`  
Số dòng: 1138 (`frontend/app.js`), 1166 (`frontend/app.js`), 1157 (`backend.py`), 1175 (`backend.py`)  
Chức năng: Nút xóa lịch sử chat theo session và nút xóa toàn bộ vector store/file upload, đều có `window.confirm` trước khi xóa.  
Câu hỏi có thể hỏi: "Tại sao tách 2 nút xóa khác nhau?"  
Câu trả lời: "Vì nhu cầu khác nhau: có lúc chỉ muốn xóa hội thoại, có lúc cần reset toàn bộ dữ liệu đã index."

4) Câu 8.2.4 - Cải thiện chunk strategy  
Tên hàm: `split_documents` + `buildIndex`  
Số dòng: 317 (`rag_engine.py`), 1002 (`frontend/app.js`)  
Chức năng: Cho phép tùy chỉnh `chunk_size`/`chunk_overlap` từ UI, backend chia chunk theo cấu hình để benchmark và so sánh độ chính xác retrieval.  
Câu hỏi có thể hỏi: "Vì sao phải benchmark nhiều cấu hình chunk?"  
Câu trả lời: "Để tìm điểm cân bằng giữa giữ ngữ cảnh, độ chính xác truy hồi và chi phí xử lý/token."

5) Câu 8.2.5 - Thêm citation/source tracking  
Tên hàm: `_format_source_documents` + `addMessage` + `openCitationModal`  
Số dòng: 1739 (`rag_engine.py`), 491 (`frontend/app.js`), 582 (`frontend/app.js`)  
Chức năng: Trả về metadata nguồn (trang, vị trí, chunk, preview, context), hiển thị trong chat và cho click xem context gốc kèm highlight.  
Câu hỏi có thể hỏi: "Citation cần tối thiểu các trường nào để kiểm chứng tốt?"  
Câu trả lời: "Nên có source, page, vị trí, chunk_id và context_text để người dùng xác thực được bằng chứng trả lời."

6) Câu 8.2.6 - Implement Conversational RAG  
Tên hàm: `contextualize_question` + `ask_question` + `ask`  
Số dòng: 795 (`rag_engine.py`), 1879 (`rag_engine.py`), 967 (`backend.py`)  
Chức năng: Dùng history để phát hiện follow-up và rewrite câu hỏi độc lập trước retrieval; backend duy trì `conversation_history` theo phiên.  
Câu hỏi có thể hỏi: "Follow-up question được xử lý ở đâu?"  
Câu trả lời: "Ở `contextualize_question`, nơi câu mơ hồ được viết lại thành truy vấn đầy đủ ngữ cảnh."

7) Câu 8.2.7 - Thêm hybrid search  
Tên hàm: `HybridSearchRetriever._get_relevant_documents` + `compare_retrieval_methods`  
Số dòng: 493 (`rag_engine.py`, trong class), 1649 (`rag_engine.py`)  
Chức năng: Kết hợp semantic search (FAISS) và keyword search (BM25) qua `EnsembleRetriever`, có trọng số semantic/keyword và endpoint so sánh hiệu năng.  
Câu hỏi có thể hỏi: "Hybrid search cải thiện điểm gì so với vector-only?"  
Câu trả lời: "Hybrid tăng recall cho từ khóa quan trọng (tên riêng, số liệu) mà vector-only có thể bỏ sót."

8) Câu 8.2.8 - Multi-document RAG với metadata filtering  
Tên hàm: `load_documents_from_files` + `_merge_document_metadata` + `filter_documents_by_metadata`  
Số dòng: 167, 82, 140 (`rag_engine.py`)  
Chức năng: Hỗ trợ upload nhiều tài liệu, gắn metadata (`source`, `doc_id`, `upload_time`, `file_type`) và filter theo metadata khi search.  
Câu hỏi có thể hỏi: "Filter metadata giúp gì trong multi-document?"  
Câu trả lời: "Giúp giới hạn retrieval vào đúng tài liệu mục tiêu, tránh lẫn nguồn khi có nhiều file trong cùng session."

9) Câu 8.2.9 - Implement Re-ranking với Cross-Encoder  
Tên hàm: `rerank_documents` + `RerankingRetriever._get_relevant_documents` + `compare_retrieval_methods`  
Số dòng: 635 (`rag_engine.py`), 433 (`rag_engine.py`, trong class), 1649 (`rag_engine.py`)  
Chức năng: Lấy candidate bằng retrieval nhanh rồi dùng cross-encoder chấm lại độ liên quan, so sánh với bi-encoder và đo overhead latency.  
Câu hỏi có thể hỏi: "Cách tối ưu latency khi thêm rerank là gì?"  
Câu trả lời: "Chỉ rerank top-N candidate nhỏ (`rerank_top_n`) thay vì toàn bộ tập chunk."

10) Câu 8.2.10 - Advanced RAG với Self-RAG  
Tên hàm: `ask_question_with_self_rag` + `rewrite_query` + `multi_hop_retrieval` + `evaluate_answer_quality`  
Số dòng: 1770, 1371, 1523, 1424 (`rag_engine.py`)  
Chức năng: Tích hợp query rewriting, multi-hop reasoning, tự đánh giá chất lượng (grounding/relevance/completeness) và confidence scoring trước khi trả lời.  
Câu hỏi có thể hỏi: "Self-RAG khác gì so với RAG truyền thống?"  
Câu trả lời: "Self-RAG không chỉ trả lời, mà còn tự kiểm tra chất lượng và tự cải thiện quá trình truy hồi để tăng độ tin cậy."
