const buildForm = document.getElementById("build-form");
const askForm = document.getElementById("ask-form");
const chat = document.getElementById("chat");

const buildStatus = document.getElementById("build-status");
const askStatus = document.getElementById("ask-status");
const metricLanguage = document.getElementById("metric-language");
const metricChunks = document.getElementById("metric-chunks");

function addMessage(role, text, sources = []) {
  const item = document.createElement("div");
  item.className = `message ${role}`;
  item.textContent = text;

  if (role === "assistant" && sources.length > 0) {
    const sourceBox = document.createElement("div");
    sourceBox.className = "sources";
    sourceBox.textContent = sources
      .map((s) => `Chunk ${s.chunk_id} (${s.source}): ${s.preview}`)
      .join("\n");
    item.appendChild(sourceBox);
  }

  chat.appendChild(item);
  chat.scrollTop = chat.scrollHeight;
}

async function buildIndex(event) {
  event.preventDefault();
  buildStatus.textContent = "Building RAG index...";

  const filesInput = document.getElementById("pdf-files");
  const ollamaModel = document.getElementById("ollama-model").value;
  const embeddingModel = document.getElementById("embedding-model").value;
  const temperature = document.getElementById("temperature").value;

  if (!filesInput.files.length) {
    buildStatus.textContent = "Please upload at least one PDF file.";
    return;
  }

  const formData = new FormData();
  for (const file of filesInput.files) {
    formData.append("files", file);
  }
  formData.append("ollama_model", ollamaModel);
  formData.append("embedding_model", embeddingModel);
  formData.append("temperature", temperature);

  try {
    const response = await fetch("/api/build-index", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Cannot build index.");
    }

    metricLanguage.textContent = data.doc_language;
    metricChunks.textContent = String(data.chunk_count);
    buildStatus.textContent = "Index built successfully.";
    askStatus.textContent = "Now you can ask questions.";
    chat.innerHTML = "";
  } catch (error) {
    buildStatus.textContent = error.message;
  }
}

async function askQuestion(event) {
  event.preventDefault();
  const questionInput = document.getElementById("question");
  const question = questionInput.value.trim();

  if (!question) {
    return;
  }

  addMessage("user", question);
  askStatus.textContent = "Generating answer...";
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
      throw new Error(data.detail || "Cannot generate answer.");
    }

    addMessage("assistant", `${data.answer}\n\nResponse time: ${data.response_time}s`, data.sources);
    askStatus.textContent = "Done.";
  } catch (error) {
    addMessage("assistant", `Error: ${error.message}`);
    askStatus.textContent = "Failed.";
  }
}

buildForm.addEventListener("submit", buildIndex);
askForm.addEventListener("submit", askQuestion);
