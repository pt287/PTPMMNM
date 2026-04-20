const buildForm = document.getElementById("build-form");
const askForm = document.getElementById("ask-form");
const chat = document.getElementById("chat");
const questionInput = document.getElementById("question");
const filesInput = document.getElementById("pdf-files");
const selectedFilesSummary = document.getElementById("selected-files-summary");
const selectedFilesPanel = document.getElementById("selected-files-panel");
const selectedFilesTitle = document.getElementById("selected-files-title");
const selectedFilesList = document.getElementById("selected-files-list");
const clearSelectedFilesButton = document.getElementById("clear-selected-files");
const clearChatButton = document.getElementById("clear-chat");
const clearHistoryButton = document.getElementById("clear-history");
const clearVectorStoreButton = document.getElementById("clear-vector-store");
const refreshHistoryButton = document.getElementById("refresh-history");
const historyToggleButton = document.getElementById("history-toggle");
const openGuideButton = document.getElementById("open-guide");
const closeGuideButton = document.getElementById("close-guide");
const guidePopup = document.getElementById("guide-popup");
const citationModal = document.getElementById("citation-modal");
const citationCloseButton = document.getElementById("close-citation");
const citationTitle = document.getElementById("citation-title");
const citationMeta = document.getElementById("citation-meta");
const citationContext = document.getElementById("citation-context");
const appGrid = document.getElementById("app-grid");
const systemStatus = document.getElementById("system-status");
const suggestionButtons = document.querySelectorAll(".suggestion");
const languageSelect = document.getElementById("ui-language");
const historyUploads = document.getElementById("history-uploads");

const buildStatus = document.getElementById("build-status");
const askStatus = document.getElementById("ask-status");
const askStatusText = document.getElementById("ask-status-text");
const askSpinner = document.getElementById("ask-spinner");
const buildProgressWrap = document.getElementById("build-progress");
const buildProgressBar = document.getElementById("build-progress-bar");
const modelSelect = document.getElementById("ollama-model");
const chunkSizeSelect = document.getElementById("chunk-size");
const chunkOverlapSelect = document.getElementById("chunk-overlap");
const metricLanguage = document.getElementById("metric-language");
const metricChunks = document.getElementById("metric-chunks");
const useGraphRagCheckbox = document.getElementById("use-graph-rag");
const useRerankingCheckbox = document.getElementById("use-reranking");
const rerankerModelSelect = document.getElementById("reranker-model");
const rerankTopNSelect = document.getElementById("rerank-top-n");
const rerankerModelLabel = document.getElementById("reranker-model-label");
const rerankTopNLabel = document.getElementById("rerank-top-n-label");

const useSelfRagCheckbox = document.getElementById("use-self-rag");
const enableQueryRewritingCheckbox = document.getElementById("enable-query-rewriting");
const enableMultiHopCheckbox = document.getElementById("enable-multi-hop");
const maxHopsSelect = document.getElementById("max-hops");
const confidenceThresholdSelect = document.getElementById("confidence-threshold");
const queryRewritingLabel = document.getElementById("query-rewriting-label");
const multiHopLabel = document.getElementById("multi-hop-label");
const maxHopsLabel = document.getElementById("max-hops-label");
const confidenceThresholdLabel = document.getElementById("confidence-threshold-label");

const translations = {
  vi: {
    pageTitle: "SmartDoc Workspace",
    heroTitle: "Trợ lý đọc tài liệu cho văn phòng và sinh viên",
    languageLabel: "Ngôn ngữ",
    systemReady: "Hệ thống sẵn sàng",
    workflowTitle: "Quy trình 3 bước",
    step1Title: "Nạp tài liệu",
    step1Desc: "Tải lên một hoặc nhiều file PDF hoặc DOCX để tạo bộ nhớ nội bộ.",
    step2Title: "Tạo RAG Index",
    step2Desc: "Hệ thống chia đoạn và lập chỉ mục để tra cứu nhanh.",
    step3Title: "Hỏi đáp theo ngữ cảnh",
    step3Desc: "Đặt câu hỏi và xem nguồn trích dẫn ngay trong khung chat.",
    metricLanguage: "Ngôn ngữ",
    metricChunks: "Số chunk",
    metricChunkSize: "Chunk size",
    metricChunkOverlap: "Chunk overlap",
    metricGraphRag: "GraphRAG tăng ngữ cảnh",
    useGraphRagLabel: "Bật GraphRAG (mở rộng ngữ cảnh liên quan)",
    metricReranking: "Re-ranking",
    useRerankingLabel: "Bật Re-ranking (Cross-Encoder)",
    rerankerModelLabel: "Re-ranker Model",
    rerankTopNLabel: "Re-rank Top N candidates",
    metricSelfRag: "Self-RAG",
    useSelfRagLabel: "Bật Self-RAG (Tự đánh giá)",
    enableQueryRewritingLabel: "Query Rewriting (Cải thiện câu hỏi)",
    enableMultiHopLabel: "Multi-hop Reasoning (Suy luận nhiều bước)",
    maxHopsLabel: "Max Hops",
    confidenceThresholdLabel: "Confidence Threshold",
    sectionConfigTitle: "1. Tài liệu",
    sectionConfigDesc: "Dành cho bài tập, báo cáo, quy trình nghiệp vụ, tài liệu họp.",
    pdfLabel: "Chọn file PDF/DOCX",
    modelLabel: "Model Ollama",
    chunkSizeLabel: "Chunk size",
    chunkOverlapLabel: "Chunk overlap",
    buildButton: "Tạo RAG Index",
    sectionChatTitle: "2. Hỏi đáp thông minh",
    sectionChatDesc: "Đặt câu hỏi ngắn gọn, bối cảnh rõ ràng để nhận câu trả lời chính xác hơn.",
    clearChat: "Xóa màn hình chat",
    clearHistory: "Clear History",
    clearVectorStore: "Clear Vector Store",
    suggestionSummary: "Tóm tắt 5 ý chính",
    suggestionAction: "Liệt kê mốc hành động",
    suggestionCompare: "So sánh 2 nội dung",
    questionPlaceholder: "Ví dụ: Rút gọn tài liệu thành checklist 1 trang",
    askButton: "Gửi câu hỏi",
    systemIndexed: "Đã index, sẵn sàng hỏi đáp",
    systemOffline: "Không kết nối được backend",
    statusBuilding: "Đang tạo RAG index...",
    statusNeedPdf: "Vui lòng chọn ít nhất 1 file PDF hoặc DOCX.",
    statusBuildError: "Không thể tạo index.",
    statusBuildSuccess: "Tạo index thành công.",
    statusReadyToAsk: "Bạn có thể bắt đầu đặt câu hỏi.",
    statusGenerating: "Đang tạo câu trả lời...",
    statusAskError: "Không thể tạo câu trả lời.",
    statusDone: "Hoàn tất.",
    statusFailed: "Thất bại.",
    statusChatCleared: "Đã xóa lịch sử chat.",
    statusHistoryCleared: "Đã xóa lịch sử chat của phiên hiện tại.",
    statusVectorStoreCleared: "Đã xóa tài liệu upload và vector store.",
    confirmClearHistory: "Bạn có chắc muốn xóa toàn bộ lịch sử chat không?",
    confirmClearVectorStore: "Bạn có chắc muốn xóa toàn bộ tài liệu đã upload và vector store không?",
    statusNoSessionToClear: "Không có session để xóa lịch sử. Hãy build index hoặc chọn 1 session trong lịch sử.",
    responseTime: "Thời gian phản hồi",
    sourcePrefix: "Đoạn",
    sourcesLabel: "Nguồn tham chiếu",
    openSourceContext: "Xem context gốc",
    sourceUnknownPage: "Không rõ trang",
    sourceUnknownPosition: "Không rõ vị trí",
    sourcePageLabel: "Trang",
    sourcePositionLabel: "Vị trí",
    sourceChunkLabel: "Chunk",
    sourcePreviewLabel: "Trích đoạn",
    sourceNoContext: "Không có context cho nguồn này.",
    sectionHistoryTitle: "3. Lịch sử làm việc",
    sectionHistoryDesc: "Chỉ hiển thị lịch sử gửi tài liệu. Bấm vào từng mục để xem hội thoại và hỏi tiếp.",
    refreshHistory: "Làm mới",
    guideButton: "Hướng dẫn",
    closeGuide: "Đóng",
    collapseHistory: "Thu gọn",
    expandHistory: "Mở lịch sử",
    uploadHistoryTitle: "Lịch sử gửi tài liệu",
    noUploadHistory: "Chưa có lịch sử gửi tài liệu.",
    uploadLabel: "Lần index",
    filesLabel: "Tệp",
    questionLabel: "Hỏi",
    answerLabel: "Trả lời",
    viewConversation: "Xem hội thoại",
    hideConversation: "Ẩn hội thoại",
    continueThisFile: "Tiếp tục hỏi file này",
    activatingSession: "Đang mở lại phiên tài liệu...",
    sessionReady: "Đã mở phiên. Bạn có thể hỏi tiếp trên file này.",
    noConversationYet: "Phiên này chưa có lịch sử hỏi đáp.",
    sessionIdLabel: "Phiên",
    activeLabel: "Đang dùng",
  },
  en: {
    pageTitle: "SmartDoc Workspace",
    heroTitle: "PDF Reading Assistant for Office Work and Students",
    languageLabel: "Language",
    systemReady: "System ready",
    workflowTitle: "3-step workflow",
    step1Title: "Upload documents",
    step1Desc: "Upload one or more PDF or DOCX files to build internal context.",
    step2Title: "Build RAG index",
    step2Desc: "The system chunks content and builds an index for fast retrieval.",
    step3Title: "Ask with context",
    step3Desc: "Ask questions and review cited sources in chat.",
    metricLanguage: "Language",
    metricChunks: "Chunks",
    metricChunkSize: "Chunk size",
    metricChunkOverlap: "Chunk overlap",
    metricGraphRag: "GraphRAG context boost",
    useGraphRagLabel: "Enable GraphRAG (expand related context)",
    metricReranking: "Re-ranking",
    useRerankingLabel: "Enable Re-ranking (Cross-Encoder)",
    rerankerModelLabel: "Re-ranker Model",
    rerankTopNLabel: "Re-rank Top N candidates",
    metricSelfRag: "Self-RAG",
    useSelfRagLabel: "Enable Self-RAG (Self-Evaluation)",
    enableQueryRewritingLabel: "Query Rewriting (Improve Question)",
    enableMultiHopLabel: "Multi-hop Reasoning",
    maxHopsLabel: "Max Hops",
    confidenceThresholdLabel: "Confidence Threshold",
    sectionConfigTitle: "1. Documents",
    sectionConfigDesc: "Great for homework, reports, business workflows, and meeting docs.",
    pdfLabel: "Select PDF/DOCX files",
    modelLabel: "Ollama model",
    chunkSizeLabel: "Chunk size",
    chunkOverlapLabel: "Chunk overlap",
    buildButton: "Build RAG Index",
    sectionChatTitle: "2. Smart Q&A",
    sectionChatDesc: "Ask concise, contextual questions for more accurate answers.",
    clearChat: "Clear Chat Screen",
    clearHistory: "Clear History",
    clearVectorStore: "Clear Vector Store",
    suggestionSummary: "Summarize top 5 points",
    suggestionAction: "List action items",
    suggestionCompare: "Compare 2 sections",
    questionPlaceholder: "Example: Turn this document into a one-page checklist",
    askButton: "Send question",
    systemIndexed: "Indexed and ready for questions",
    systemOffline: "Cannot connect to backend",
    statusBuilding: "Building RAG index...",
    statusNeedPdf: "Please upload at least one PDF or DOCX file.",
    statusBuildError: "Cannot build index.",
    statusBuildSuccess: "Index built successfully.",
    statusReadyToAsk: "You can start asking questions.",
    statusGenerating: "Generating answer...",
    statusAskError: "Cannot generate answer.",
    statusDone: "Done.",
    statusFailed: "Failed.",
    statusChatCleared: "Chat history has been cleared.",
    statusHistoryCleared: "Current session chat history has been cleared.",
    statusVectorStoreCleared: "Uploaded documents and vector store have been cleared.",
    confirmClearHistory: "Are you sure you want to clear all chat history?",
    confirmClearVectorStore: "Are you sure you want to clear all uploaded documents and vector store?",
    statusNoSessionToClear: "No session available to clear. Build index or choose a session from history first.",
    responseTime: "Response time",
    sourcePrefix: "Chunk",
    sourcesLabel: "Citations",
    openSourceContext: "Open source context",
    sourceUnknownPage: "Unknown page",
    sourceUnknownPosition: "Unknown position",
    sourcePageLabel: "Page",
    sourcePositionLabel: "Position",
    sourceChunkLabel: "Chunk",
    sourcePreviewLabel: "Excerpt",
    sourceNoContext: "No context is available for this source.",
    sectionHistoryTitle: "3. Activity history",
    sectionHistoryDesc: "Shows document upload history only. Click an item to view Q&A and continue with that file.",
    refreshHistory: "Refresh",
    guideButton: "Guide",
    closeGuide: "Close",
    collapseHistory: "Collapse",
    expandHistory: "Open history",
    uploadHistoryTitle: "Document upload history",
    noUploadHistory: "No document upload history yet.",
    uploadLabel: "Index session",
    filesLabel: "Files",
    questionLabel: "Question",
    answerLabel: "Answer",
    viewConversation: "View conversation",
    hideConversation: "Hide conversation",
    continueThisFile: "Continue with this file",
    activatingSession: "Reopening document session...",
    sessionReady: "Session is active. You can continue asking on this file.",
    noConversationYet: "No Q&A history for this session yet.",
    sessionIdLabel: "Session",
    activeLabel: "Active",
  },
};

const savedLanguage = localStorage.getItem("smartdoc_ui_language");
let currentLanguage = savedLanguage && translations[savedLanguage] ? savedLanguage : "vi";
let hasIndexedData = false;
let activeSessionId = null;
let latestSessionId = null;
let isHistoryCollapsed = localStorage.getItem("smartdoc_history_collapsed") === "1";
let isGuideDismissed = localStorage.getItem("smartdoc_guide_dismissed") === "1";
let selectedBuildFiles = [];

function t(key) {
  return translations[currentLanguage][key] || translations.vi[key] || key;
}

function setSystemStatus(online, label) {
  const textNode = systemStatus.querySelector("span:last-child");
  textNode.textContent = label;
  systemStatus.classList.toggle("offline", !online);
}

function applyTranslations() {
  document.documentElement.lang = currentLanguage;
  document.title = t("pageTitle");

  document.querySelectorAll("[data-i18n]").forEach((element) => {
    const key = element.dataset.i18n;
    element.textContent = t(key);
  });

  document.querySelectorAll("[data-i18n-placeholder]").forEach((element) => {
    const key = element.dataset.i18nPlaceholder;
    element.setAttribute("placeholder", t(key));
  });

  document.querySelectorAll("[data-i18n-title]").forEach((element) => {
    const key = element.dataset.i18nTitle;
    element.setAttribute("title", t(key));
  });

  document.querySelectorAll("[data-i18n-aria]").forEach((element) => {
    const key = element.dataset.i18nAria;
    element.setAttribute("aria-label", t(key));
  });

  suggestionButtons.forEach((button) => {
    button.dataset.question = currentLanguage === "en" ? button.dataset.questionEn : button.dataset.questionVi;
  });

  if (historyToggleButton) {
    const historyLabel = isHistoryCollapsed ? t("expandHistory") : t("collapseHistory");
    historyToggleButton.setAttribute("title", historyLabel);
    historyToggleButton.setAttribute("aria-label", historyLabel);
  }

  if (hasIndexedData) {
    if (activeSessionId) {
      setSystemStatus(true, `${t("systemIndexed")} - ${t("sessionIdLabel")} #${activeSessionId}`);
    } else {
      setSystemStatus(true, t("systemIndexed"));
    }
  } else {
    setSystemStatus(true, t("systemReady"));
  }

  updateSelectedFilesSummary();
}

function updateSelectedFilesSummary() {
  if (!selectedFilesSummary) {
    return;
  }

  const count = selectedBuildFiles.length;
  if (currentLanguage === "vi") {
    selectedFilesSummary.textContent = count === 0
      ? "Đã chọn 0 tệp"
      : `Đã chọn ${count} tệp`;
  } else {
    selectedFilesSummary.textContent = count === 0
      ? "0 file selected"
      : `${count} file(s) selected`;
  }

  renderSelectedFilesPanel();
}

function formatFileSize(bytes) {
  const kb = Math.max(1, Math.round((Number(bytes) || 0) / 1024));
  return `${kb} KB`;
}

function removeSelectedFileByIndex(index) {
  if (index < 0 || index >= selectedBuildFiles.length) {
    return;
  }
  selectedBuildFiles.splice(index, 1);
  syncFileInputFromSelection();
  updateSelectedFilesSummary();
}

function clearAllSelectedFiles() {
  selectedBuildFiles = [];
  filesInput.value = "";
  syncFileInputFromSelection();
  updateSelectedFilesSummary();
}

function renderSelectedFilesPanel() {
  if (!selectedFilesPanel || !selectedFilesList || !selectedFilesTitle || !clearSelectedFilesButton) {
    return;
  }

  const hasFiles = selectedBuildFiles.length > 0;
  selectedFilesPanel.classList.toggle("show", hasFiles);
  selectedFilesList.innerHTML = "";

  if (currentLanguage === "vi") {
    selectedFilesTitle.textContent = "Danh sách file đã chọn";
    clearSelectedFilesButton.textContent = "Xóa tất cả";
    clearSelectedFilesButton.setAttribute("aria-label", "Xóa tất cả file đã chọn");
  } else {
    selectedFilesTitle.textContent = "Selected files";
    clearSelectedFilesButton.textContent = "Clear all";
    clearSelectedFilesButton.setAttribute("aria-label", "Clear all selected files");
  }

  if (!hasFiles) {
    return;
  }

  selectedBuildFiles.forEach((file, index) => {
    const row = document.createElement("div");
    row.className = "selected-file-item";

    const meta = document.createElement("div");
    meta.className = "selected-file-meta";

    const name = document.createElement("p");
    name.className = "selected-file-name";
    name.textContent = file.name;
    name.title = file.name;

    const size = document.createElement("p");
    size.className = "selected-file-size";
    size.textContent = formatFileSize(file.size);

    meta.appendChild(name);
    meta.appendChild(size);

    const removeButton = document.createElement("button");
    removeButton.type = "button";
    removeButton.className = "btn-ghost small remove-selected-file";
    removeButton.dataset.index = String(index);
    if (currentLanguage === "vi") {
      removeButton.setAttribute("title", "Xóa file");
      removeButton.setAttribute("aria-label", `Xóa file ${file.name}`);
    } else {
      removeButton.setAttribute("title", "Remove file");
      removeButton.setAttribute("aria-label", `Remove file ${file.name}`);
    }
    removeButton.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M19 6h-3.5l-1-1h-5l-1 1H5v2h14zm-1 3H6v10a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2z" fill="currentColor"/></svg>';

    row.appendChild(meta);
    row.appendChild(removeButton);
    selectedFilesList.appendChild(row);
  });
}

function fileKey(file) {
  return `${file.name}|${file.size}|${file.lastModified}`;
}

function syncFileInputFromSelection() {
  if (typeof DataTransfer === "undefined") {
    return;
  }

  const dataTransfer = new DataTransfer();
  selectedBuildFiles.forEach((file) => {
    dataTransfer.items.add(file);
  });
  filesInput.files = dataTransfer.files;
}

function mergeSelectedFiles(newFiles) {
  const merged = [...selectedBuildFiles];
  const seen = new Set(merged.map(fileKey));
  newFiles.forEach((file) => {
    const key = fileKey(file);
    if (!seen.has(key)) {
      seen.add(key);
      merged.push(file);
    }
  });
  selectedBuildFiles = merged;
  syncFileInputFromSelection();
  updateSelectedFilesSummary();
}

function handleFileSelectionChange() {
  const incomingFiles = Array.from(filesInput.files || []);
  if (!incomingFiles.length) {
    return;
  }
  mergeSelectedFiles(incomingFiles);
}

function handleSelectedFilesListClick(event) {
  const button = event.target.closest(".remove-selected-file");
  if (!button) {
    return;
  }

  const index = Number(button.dataset.index);
  if (Number.isInteger(index)) {
    removeSelectedFileByIndex(index);
  }
}

function setHistoryCollapsed(collapsed) {
  isHistoryCollapsed = Boolean(collapsed);
  appGrid.classList.toggle("sidebar-collapsed", isHistoryCollapsed);
  localStorage.setItem("smartdoc_history_collapsed", isHistoryCollapsed ? "1" : "0");
  if (historyToggleButton) {
    const historyLabel = isHistoryCollapsed ? t("expandHistory") : t("collapseHistory");
    historyToggleButton.setAttribute("title", historyLabel);
    historyToggleButton.setAttribute("aria-label", historyLabel);
  }
}

function setGuideVisible(visible) {
  guidePopup.classList.toggle("show", Boolean(visible));
  guidePopup.setAttribute("aria-hidden", visible ? "false" : "true");
}

function dismissGuide() {
  isGuideDismissed = true;
  localStorage.setItem("smartdoc_guide_dismissed", "1");
  setGuideVisible(false);
}

function addMessage(role, text, sources = []) {
  const item = document.createElement("div");
  item.className = `message ${role}`;
  item.textContent = text;

  if (role === "assistant" && sources.length > 0) {
    const sourceBox = document.createElement("div");
    sourceBox.className = "sources";

    const sourceTitle = document.createElement("p");
    sourceTitle.className = "sources-title";
    sourceTitle.textContent = t("sourcesLabel");
    sourceBox.appendChild(sourceTitle);

    sources.forEach((source) => {
      const sourceButton = document.createElement("button");
      sourceButton.type = "button";
      sourceButton.className = "source-item";

      const pageText = source.page !== undefined && source.page !== null && source.page !== "?"
        ? source.page
        : t("sourceUnknownPage");

      let positionText = t("sourceUnknownPosition");
      if (Number.isInteger(source.position_start) && Number.isInteger(source.position_end)) {
        positionText = `${source.position_start}-${source.position_end}`;
      }

      const meta = `${t("sourcePrefix")} ${source.chunk_id} | ${source.source} | ${t("sourcePageLabel")}: ${pageText} | ${t("sourcePositionLabel")}: ${positionText}`;
      const safeMeta = escapeHtml(meta);
      const safePreview = escapeHtml(source.preview || "");
      sourceButton.innerHTML = `
        <span class="source-meta">${safeMeta}</span>
        <span class="source-preview">${safePreview}</span>
        <span class="source-open">${escapeHtml(t("openSourceContext"))}</span>
      `;

      sourceButton.addEventListener("click", () => {
        openCitationModal(source);
      });

      sourceBox.appendChild(sourceButton);
    });

    item.appendChild(sourceBox);
  }

  chat.appendChild(item);
  chat.scrollTop = chat.scrollHeight;
}

function escapeHtml(input) {
  return String(input || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function escapeRegExp(input) {
  return String(input || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function renderCitationContext(contextText, highlightText) {
  const safeContext = String(contextText || "").trim();
  if (!safeContext) {
    return `<p class="citation-empty">${escapeHtml(t("sourceNoContext"))}</p>`;
  }

  let marker = String(highlightText || "").trim();
  marker = marker.replace(/\.{3}$/g, "").trim();
  if (!marker) {
    return escapeHtml(safeContext);
  }

  const normalizedMarker = marker.replace(/\s+/g, " ");
  const pattern = new RegExp(escapeRegExp(normalizedMarker), "i");
  const match = pattern.exec(safeContext);
  if (!match) {
    return escapeHtml(safeContext);
  }

  const start = match.index;
  const end = start + match[0].length;
  const before = escapeHtml(safeContext.slice(0, start));
  const middle = escapeHtml(safeContext.slice(start, end));
  const after = escapeHtml(safeContext.slice(end));
  return `${before}<mark>${middle}</mark>${after}`;
}

function openCitationModal(source) {
  const pageText = source.page !== undefined && source.page !== null && source.page !== "?"
    ? String(source.page)
    : t("sourceUnknownPage");

  let positionText = t("sourceUnknownPosition");
  if (Number.isInteger(source.position_start) && Number.isInteger(source.position_end)) {
    positionText = `${source.position_start}-${source.position_end}`;
  }

  citationTitle.textContent = `${t("sourceChunkLabel")} ${source.chunk_id} | ${source.source}`;
  citationMeta.textContent = `${t("sourcePageLabel")}: ${pageText} | ${t("sourcePositionLabel")}: ${positionText}`;

  const contextText = String(source.context_text || "").trim();
  const previewText = String(source.preview || "").trim();
  const contentText = contextText || previewText;
  const highlightText = source.highlight_text || previewText;
  citationContext.style.display = "block";
  citationContext.scrollTop = 0;
  citationContext.innerHTML = renderCitationContext(contentText, highlightText);

  citationModal.classList.add("show");
  citationModal.setAttribute("aria-hidden", "false");
}

function closeCitationModal() {
  citationModal.classList.remove("show");
  citationModal.setAttribute("aria-hidden", "true");
}

function setAskStatus(text, loading = false) {
  askStatusText.textContent = text;
  askSpinner.classList.toggle("show", loading);
}

function startBuildProgress() {
  buildProgressWrap.classList.add("show");
  buildProgressBar.style.width = "8%";
}

function updateBuildProgress(value) {
  buildProgressBar.style.width = `${Math.max(0, Math.min(100, value))}%`;
}

function stopBuildProgress() {
  updateBuildProgress(100);
  setTimeout(() => {
    buildProgressWrap.classList.remove("show");
    buildProgressBar.style.width = "0%";
  }, 320);
}

function formatTimestamp(value) {
  if (!value) {
    return "-";
  }

  const date = new Date(`${value}Z`);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString(currentLanguage === "vi" ? "vi-VN" : "en-US");
}

function formatHistoryTimestamp(value) {
  if (!value) {
    return "-";
  }

  const date = new Date(`${value}Z`);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  const locale = currentLanguage === "vi" ? "vi-VN" : "en-US";
  return date.toLocaleString(locale, {
    hour: "2-digit",
    minute: "2-digit",
    day: "numeric",
    month: "numeric",
  });
}

function createHistoryItem(title, subtitle, bodyLines = []) {
  const item = document.createElement("article");
  item.className = "history-item";

  const heading = document.createElement("h4");
  heading.textContent = title;

  const meta = document.createElement("p");
  meta.className = "history-meta";
  meta.textContent = subtitle;

  item.appendChild(heading);
  item.appendChild(meta);

  bodyLines.forEach((line) => {
    const paragraph = document.createElement("p");
    paragraph.className = "history-line";
    paragraph.textContent = line;
    item.appendChild(paragraph);
  });

  return item;
}

function renderSessionQa(container, qaItems) {
  container.innerHTML = "";

  if (!qaItems.length) {
    const empty = document.createElement("p");
    empty.className = "history-empty";
    empty.textContent = t("noConversationYet");
    container.appendChild(empty);
    return;
  }

  const ordered = [...qaItems].reverse();
  ordered.forEach((entry) => {
    const qaWrap = document.createElement("div");
    qaWrap.className = "session-qa-item";

    const q = document.createElement("p");
    q.className = "history-line";
    q.textContent = `${t("questionLabel")}: ${entry.question}`;

    const a = document.createElement("p");
    a.className = "history-line";
    a.textContent = `${t("answerLabel")}: ${entry.answer}`;

    const m = document.createElement("p");
    m.className = "history-meta";
    m.textContent = `${formatTimestamp(entry.created_at)} | ${t("responseTime")}: ${entry.response_time}s`;

    qaWrap.appendChild(q);
    qaWrap.appendChild(a);
    qaWrap.appendChild(m);
    container.appendChild(qaWrap);
  });
}

function applySessionConfigToControls(data) {
  useGraphRagCheckbox.checked = Boolean(data.use_graph_rag);
  useRerankingCheckbox.checked = Boolean(data.use_reranking);
  useSelfRagCheckbox.checked = Boolean(data.use_self_rag);

  if (data.reranker_model) {
    rerankerModelSelect.value = String(data.reranker_model);
  }
  if (data.rerank_top_n != null) {
    rerankTopNSelect.value = String(data.rerank_top_n);
  }

  if (data.enable_query_rewriting != null) {
    enableQueryRewritingCheckbox.checked = Boolean(data.enable_query_rewriting);
  }
  if (data.enable_multi_hop != null) {
    enableMultiHopCheckbox.checked = Boolean(data.enable_multi_hop);
  }
  if (data.max_hops != null) {
    maxHopsSelect.value = String(data.max_hops);
  }
  if (data.confidence_threshold != null) {
    confidenceThresholdSelect.value = String(data.confidence_threshold);
  }

  toggleRerankingOptions();
  toggleSelfRagOptions();
}

async function activateSession(sessionId) {
  setAskStatus(t("activatingSession"), true);

  const response = await fetch(`/api/sessions/${sessionId}/activate`, {
    method: "POST",
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || t("statusFailed"));
  }

  activeSessionId = sessionId;
  hasIndexedData = true;
  metricLanguage.textContent = data.doc_language;
  metricChunks.textContent = String(data.chunk_count);
  applySessionConfigToControls(data);
  chunkSizeSelect.value = String(data.chunk_size || 1500);
  chunkOverlapSelect.value = String(data.chunk_overlap || 100);
  setSystemStatus(true, `${t("systemIndexed")} - ${t("sessionIdLabel")} #${sessionId}`);
  setAskStatus(t("sessionReady"), false);
}

function attachHistoryActions(root, sessionId, qaContainer, toggleButton, continueButton) {
  function updateConversationButtonState(isOpen) {
    const label = isOpen
      ? (currentLanguage === "vi" ? "Ẩn hội thoại" : "Hide conversation")
      : (currentLanguage === "vi" ? "Xem hội thoại" : "View conversation");
    toggleButton.classList.toggle("active", isOpen);
    toggleButton.setAttribute("title", label);
    toggleButton.setAttribute("aria-label", label);
  }

  updateConversationButtonState(false);

  const continueLabel = currentLanguage === "vi" ? "Mở phiên này" : "Activate this session";
  continueButton.setAttribute("title", continueLabel);
  continueButton.setAttribute("aria-label", continueLabel);

  toggleButton.addEventListener("click", async () => {
    const isOpen = qaContainer.classList.contains("open");
    if (isOpen) {
      qaContainer.classList.remove("open");
      qaContainer.innerHTML = "";
      updateConversationButtonState(false);
      return;
    }

    qaContainer.classList.add("open");
    updateConversationButtonState(true);
    qaContainer.innerHTML = `<p class="history-empty">${t("statusGenerating")}</p>`;

    try {
      const response = await fetch(`/api/sessions/${sessionId}/history?limit=20`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || t("statusFailed"));
      }
      renderSessionQa(qaContainer, data.qa || []);
    } catch (error) {
      qaContainer.innerHTML = `<p class="history-empty">${error.message}</p>`;
    }
  });

  continueButton.addEventListener("click", async () => {
    try {
      await activateSession(sessionId);

      document.querySelectorAll(".history-item").forEach((node) => {
        node.classList.toggle("active", node === root);
      });

      const response = await fetch(`/api/sessions/${sessionId}/history?limit=12`);
      const data = await response.json();
      if (response.ok) {
        chat.innerHTML = "";
        const rows = [...(data.qa || [])].reverse();
        rows.forEach((entry) => {
          addMessage("user", entry.question);
          addMessage("assistant", `${entry.answer}\n\n${t("responseTime")}: ${entry.response_time}s`, entry.sources || []);
        });
      }
      loadHistory();
    } catch (error) {
      setAskStatus(error.message, false);
    }
  });
}

function renderUploadHistory(items) {
  historyUploads.innerHTML = "";
  latestSessionId = items.length ? items[0].session_id : null;

  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "history-empty";
    empty.textContent = t("noUploadHistory");
    historyUploads.appendChild(empty);
    return;
  }

  items.forEach((entry) => {
    const files = entry.pdf_files || [];
    const firstFile = files[0]?.file_name || "-";
    const extraFiles = Math.max(0, files.length - 1);
    const primaryFileText = extraFiles > 0
      ? `${firstFile} +${extraFiles}`
      : firstFile;

    const item = document.createElement("article");
    item.className = "history-item";
    item.classList.toggle("active", activeSessionId === entry.session_id);

    const row = document.createElement("div");
    row.className = "history-row history-row-primary";

    const main = document.createElement("div");
    main.className = "history-main";

    const index = document.createElement("span");
    index.className = "history-index";
    index.textContent = `#${entry.session_id}`;

    const fileName = document.createElement("span");
    fileName.className = "history-file-primary";
    fileName.title = firstFile;
    fileName.textContent = primaryFileText;

    main.appendChild(index);
    main.appendChild(fileName);

    const actionRow = document.createElement("div");
    actionRow.className = "history-row history-row-secondary";

    const secondaryMeta = document.createElement("span");
    secondaryMeta.className = "history-time";
    secondaryMeta.textContent = formatHistoryTimestamp(entry.created_at);

    const infoButton = document.createElement("button");
    infoButton.type = "button";
    infoButton.className = "history-icon-btn history-tech-trigger";
    infoButton.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M11 10h2v7h-2zm0-3h2v2h-2zm1-5a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z" fill="currentColor"/></svg>';

    const infoLabel = currentLanguage === "vi" ? "Thông số kỹ thuật" : "Technical details";
    infoButton.setAttribute("title", infoLabel);
    infoButton.setAttribute("aria-label", infoLabel);

    const techPanel = document.createElement("div");
    techPanel.className = "history-tech-panel";
    techPanel.innerHTML = `
      <p class="history-tech-line"><strong>${currentLanguage === "vi" ? "File" : "Files"}:</strong> ${files.length}</p>
      <p class="history-tech-line"><strong>Chunk:</strong> ${entry.chunk_count} | cs ${entry.chunk_size} | ov ${entry.chunk_overlap}</p>
      <p class="history-tech-line"><strong>Model:</strong> ${entry.ollama_model}</p>
      <p class="history-tech-line"><strong>${t("metricGraphRag")}:</strong> ${entry.use_graph_rag ? (currentLanguage === "vi" ? "Bật" : "On") : (currentLanguage === "vi" ? "Tắt" : "Off")}</p>
      <p class="history-tech-line"><strong>${t("metricReranking")}:</strong> ${entry.use_reranking ? (currentLanguage === "vi" ? "Bật" : "On") : (currentLanguage === "vi" ? "Tắt" : "Off")}</p>
      <p class="history-tech-line"><strong>${t("metricSelfRag")}:</strong> ${entry.use_self_rag ? (currentLanguage === "vi" ? "Bật" : "On") : (currentLanguage === "vi" ? "Tắt" : "Off")}</p>
    `;

    infoButton.addEventListener("click", (event) => {
      event.stopPropagation();
      const isOpen = item.classList.contains("show-tech");
      document.querySelectorAll(".history-item.show-tech").forEach((node) => {
        node.classList.remove("show-tech");
      });
      item.classList.toggle("show-tech", !isOpen);
    });

    const viewBtn = document.createElement("button");
    viewBtn.type = "button";
    viewBtn.className = "history-icon-btn";
    viewBtn.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M4 4h16v10H7l-3 3zm2 2v6.17L6.83 12H18V6z" fill="currentColor"/></svg>';

    const continueBtn = document.createElement("button");
    continueBtn.type = "button";
    continueBtn.className = "history-icon-btn primary";
    continueBtn.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M8 5v14l11-7z" fill="currentColor"/></svg>';

    const qaContainer = document.createElement("div");
    qaContainer.className = "session-qa-list";

    const actionButtons = document.createElement("div");
    actionButtons.className = "history-row-actions";
    actionButtons.appendChild(infoButton);
    actionButtons.appendChild(viewBtn);
    actionButtons.appendChild(continueBtn);

    actionRow.appendChild(secondaryMeta);
    actionRow.appendChild(actionButtons);

    row.appendChild(main);
    item.appendChild(row);
    item.appendChild(actionRow);
    item.appendChild(techPanel);
    item.appendChild(qaContainer);
    attachHistoryActions(item, entry.session_id, qaContainer, viewBtn, continueBtn);

    historyUploads.appendChild(item);
  });
}

async function loadHistory() {
  try {
    const response = await fetch("/api/history?limit=12");
    if (!response.ok) {
      throw new Error(currentLanguage === "vi" ? "Không tải được lịch sử" : "Cannot load history");
    }

    const data = await response.json();
    renderUploadHistory(data.uploads || []);
  } catch (error) {
    historyUploads.innerHTML = "";

    const message = document.createElement("p");
    message.className = "history-empty";
    message.textContent = error.message;

    historyUploads.appendChild(message);
  }
}

async function syncHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error(currentLanguage === "vi" ? "Kiểm tra trạng thái thất bại" : "Health check failed");
    }

    const data = await response.json();
    metricLanguage.textContent = data.doc_language;
    metricChunks.textContent = String(data.chunk_count);
    applySessionConfigToControls(data);
    chunkSizeSelect.value = String(data.chunk_size || 1500);
    chunkOverlapSelect.value = String(data.chunk_overlap || 100);
    hasIndexedData = Boolean(data.indexed);

    if (data.indexed) {
      if (activeSessionId) {
        setSystemStatus(true, `${t("systemIndexed")} - ${t("sessionIdLabel")} #${activeSessionId}`);
      } else {
        setSystemStatus(true, t("systemIndexed"));
      }
    } else {
      setSystemStatus(true, t("systemReady"));
    }
  } catch (error) {
    setSystemStatus(false, t("systemOffline"));
  }
}

async function buildIndex(event) {
  event.preventDefault();
  buildStatus.textContent = t("statusBuilding");
  startBuildProgress();

  if (!selectedBuildFiles.length) {
    buildStatus.textContent = t("statusNeedPdf");
    buildProgressWrap.classList.remove("show");
    return;
  }

  let fakeProgress = 8;
  const progressTimer = setInterval(() => {
    fakeProgress = Math.min(92, fakeProgress + 7);
    updateBuildProgress(fakeProgress);
  }, 180);

  const formData = new FormData();
  for (const file of selectedBuildFiles) {
    formData.append("files", file);
  }
  formData.append("ollama_model", modelSelect.value);
  formData.append("chunk_size", chunkSizeSelect.value);
  formData.append("chunk_overlap", chunkOverlapSelect.value);
  formData.append("use_graph_rag", useGraphRagCheckbox.checked);
  formData.append("use_reranking", useRerankingCheckbox.checked);
  formData.append("reranker_model", rerankerModelSelect.value);
  formData.append("rerank_top_n", rerankTopNSelect.value);
  formData.append("use_self_rag", useSelfRagCheckbox.checked);
  formData.append("enable_query_rewriting", enableQueryRewritingCheckbox.checked);
  formData.append("enable_multi_hop", enableMultiHopCheckbox.checked);
  formData.append("max_hops", maxHopsSelect.value);
  formData.append("confidence_threshold", confidenceThresholdSelect.value);

  try {
    const response = await fetch("/api/build-index", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || t("statusBuildError"));
    }

    metricLanguage.textContent = data.doc_language;
    metricChunks.textContent = String(data.chunk_count);
    chunkSizeSelect.value = String(data.chunk_size);
    chunkOverlapSelect.value = String(data.chunk_overlap);
    buildStatus.textContent = t("statusBuildSuccess");
    setAskStatus(t("statusReadyToAsk"), false);
    chat.innerHTML = "";
    activeSessionId = data.session_id || null;
    hasIndexedData = true;
    if (activeSessionId) {
      setSystemStatus(true, `${t("systemIndexed")} - ${t("sessionIdLabel")} #${activeSessionId}`);
    } else {
      setSystemStatus(true, t("systemIndexed"));
    }
    clearAllSelectedFiles();
    loadHistory();
    clearInterval(progressTimer);
    stopBuildProgress();
  } catch (error) {
    buildStatus.textContent = error.message;
    clearInterval(progressTimer);
    buildProgressWrap.classList.remove("show");
  }
}

async function askQuestion(event) {
  event.preventDefault();
  const question = questionInput.value.trim();

  if (!question) {
    return;
  }

  addMessage("user", question);
  setAskStatus(t("statusGenerating"), true);
  questionInput.value = "";

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || t("statusAskError"));
    }

    let messageText = `${data.answer}\n\n${t("responseTime")}: ${data.response_time}s`;

    // Add Self-RAG metadata if available
    if (data.self_rag_metadata) {
      const meta = data.self_rag_metadata;
      messageText += "\n\n📊 Self-RAG Metadata:";

      if (meta.query_rewrite && meta.query_rewrite.used_rewriting) {
        messageText += `\n• Original query: "${meta.query_rewrite.original}"`;
        messageText += `\n• Rewritten query: "${meta.query_rewrite.rewritten}"`;
      }

      if (meta.evaluation) {
        const evaluation = meta.evaluation;
        messageText += `\n• Confidence: ${(evaluation.confidence * 100).toFixed(1)}%`;
        messageText += `\n• Grounding: ${evaluation.grounding_score}/10`;
        messageText += `\n• Relevance: ${evaluation.relevance_score}/10`;
        messageText += `\n• Completeness: ${evaluation.completeness_score}/10`;
        messageText += `\n• Status: ${evaluation.passed ? "✓ Passed" : "⚠ Low confidence"}`;
      }

      if (meta.multi_hop && meta.multi_hop.total_hops > 1) {
        messageText += `\n• Multi-hop: ${meta.multi_hop.total_hops} hops (${meta.multi_hop.completed ? "completed" : "incomplete"})`;
      }
    }

    addMessage("assistant", messageText, data.sources);
    setAskStatus(t("statusDone"), false);
    loadHistory();
  } catch (error) {
    addMessage("assistant", `Error: ${error.message}`);
    setAskStatus(t("statusFailed"), false);
  }
}

function clearChat() {
  chat.innerHTML = "";
  setAskStatus(t("statusChatCleared"), false);
}

async function clearHistory() {
  const targetSessionId = activeSessionId || latestSessionId;
  if (!targetSessionId) {
    setAskStatus(t("statusNoSessionToClear"), false);
    return;
  }

  if (!window.confirm(t("confirmClearHistory"))) {
    return;
  }

  try {
    const response = await fetch(`/api/history?session_id=${targetSessionId}`, { method: "DELETE" });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || t("statusFailed"));
    }

    if (activeSessionId === targetSessionId) {
      chat.innerHTML = "";
    }
    setAskStatus(t("statusHistoryCleared"), false);
    loadHistory();
  } catch (error) {
    setAskStatus(error.message, false);
  }
}

async function clearVectorStore() {
  if (!window.confirm(t("confirmClearVectorStore"))) {
    return;
  }

  try {
    const response = await fetch("/api/vector-store", { method: "DELETE" });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || t("statusFailed"));
    }

    activeSessionId = null;
    hasIndexedData = false;
    chat.innerHTML = "";
    metricLanguage.textContent = "unknown";
    metricChunks.textContent = "0";
    chunkSizeSelect.value = "1500";
    chunkOverlapSelect.value = "100";
    clearAllSelectedFiles();
    setSystemStatus(true, t("systemReady"));
    buildStatus.textContent = t("statusVectorStoreCleared");
    setAskStatus(t("statusVectorStoreCleared"), false);
    loadHistory();
  } catch (error) {
    setAskStatus(error.message, false);
  }
}

function fillSuggestedQuestion(event) {
  const text = event.currentTarget.dataset.question || event.currentTarget.dataset.questionVi || "";
  questionInput.value = text;
  questionInput.focus();
}

function handleLanguageChange(event) {
  const selected = event.target.value;
  currentLanguage = translations[selected] ? selected : "vi";
  localStorage.setItem("smartdoc_ui_language", currentLanguage);
  applyTranslations();
  loadHistory();
}

clearChatButton.addEventListener("click", clearChat);
clearHistoryButton.addEventListener("click", clearHistory);
clearVectorStoreButton.addEventListener("click", clearVectorStore);
refreshHistoryButton.addEventListener("click", loadHistory);
historyToggleButton.addEventListener("click", () => {
  setHistoryCollapsed(!isHistoryCollapsed);
});
openGuideButton.addEventListener("click", () => {
  isGuideDismissed = false;
  localStorage.setItem("smartdoc_guide_dismissed", "0");
  setGuideVisible(true);
});
closeGuideButton.addEventListener("click", dismissGuide);
citationCloseButton.addEventListener("click", closeCitationModal);
guidePopup.addEventListener("click", (event) => {
  if (event.target === guidePopup) {
    dismissGuide();
  }
});
citationModal.addEventListener("click", (event) => {
  if (event.target === citationModal) {
    closeCitationModal();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && guidePopup.classList.contains("show")) {
    dismissGuide();
  }
  if (event.key === "Escape" && citationModal.classList.contains("show")) {
    closeCitationModal();
  }
});
document.addEventListener("click", (event) => {
  if (event.target.closest(".history-tech-trigger")) {
    return;
  }

  document.querySelectorAll(".history-item.show-tech").forEach((node) => {
    node.classList.remove("show-tech");
  });
});
languageSelect.addEventListener("change", handleLanguageChange);
filesInput.addEventListener("change", handleFileSelectionChange);
selectedFilesList.addEventListener("click", handleSelectedFilesListClick);
clearSelectedFilesButton.addEventListener("click", clearAllSelectedFiles);

suggestionButtons.forEach((button) => {
  button.addEventListener("click", fillSuggestedQuestion);
});

// Toggle re-ranker options visibility
function toggleRerankingOptions() {
  const isEnabled = useRerankingCheckbox.checked;
  rerankerModelLabel.style.display = isEnabled ? "" : "none";
  rerankTopNLabel.style.display = isEnabled ? "" : "none";
}

useRerankingCheckbox.addEventListener("change", toggleRerankingOptions);

// Toggle Self-RAG options visibility
function toggleSelfRagOptions() {
  const isEnabled = useSelfRagCheckbox.checked;
  queryRewritingLabel.style.display = isEnabled ? "" : "none";
  multiHopLabel.style.display = isEnabled ? "" : "none";
  maxHopsLabel.style.display = isEnabled ? "" : "none";
  confidenceThresholdLabel.style.display = isEnabled ? "" : "none";
}

useSelfRagCheckbox.addEventListener("change", toggleSelfRagOptions);

languageSelect.value = currentLanguage;
setHistoryCollapsed(isHistoryCollapsed);
applyTranslations();
setGuideVisible(!isGuideDismissed);
toggleRerankingOptions(); // Initialize visibility
toggleSelfRagOptions(); // Initialize visibility
buildForm.addEventListener("submit", buildIndex);
askForm.addEventListener("submit", askQuestion);
setAskStatus("", false);
updateSelectedFilesSummary();
syncHealth();
loadHistory();
