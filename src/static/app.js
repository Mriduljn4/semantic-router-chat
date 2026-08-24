const conversation = document.querySelector("#conversation");
const form = document.querySelector("#chat-form");
const query = document.querySelector("#query");
const clearButton = document.querySelector("#clear-chat");

function scrollToLatest() {
  conversation.scrollTop = conversation.scrollHeight;
}

function appendMessage(text, role, metadata = "") {
  const article = document.createElement("article");
  article.className = `message ${role}-message`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "assistant" ? "✦" : "You";
  const content = document.createElement("div");
  content.className = "message-content";
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  content.append(paragraph);
  if (metadata) {
    const meta = document.createElement("span");
    meta.className = "agent-meta";
    meta.textContent = metadata;
    content.append(meta);
  }
  article.append(avatar, content);
  conversation.append(article);
  scrollToLatest();
  return article;
}

function setLoading(isLoading) {
  const button = form.querySelector("button");
  button.disabled = isLoading;
  button.querySelector("span").textContent = isLoading ? "Thinking" : "Send";
}

async function submitQuery(value) {
  const text = value.trim();
  if (!text) return;
  appendMessage(text, "user");
  query.value = "";
  query.style.height = "auto";
  setLoading(true);
  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: text }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Unable to process your request.");
    appendMessage(payload.answer, "assistant", `${payload.routed_agent} · ${payload.llm_provider_used}`);
  } catch (error) {
    appendMessage(error.message || "Something went wrong. Please try again.", "assistant", "System");
  } finally {
    setLoading(false);
    query.focus();
  }
}

form.addEventListener("submit", (event) => { event.preventDefault(); submitQuery(query.value); });
query.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); } });
query.addEventListener("input", () => { query.style.height = "auto"; query.style.height = `${Math.min(query.scrollHeight, 150)}px`; });
document.querySelectorAll("[data-query]").forEach((button) => button.addEventListener("click", () => submitQuery(button.dataset.query)));
clearButton.addEventListener("click", () => { conversation.innerHTML = ""; appendMessage("Hi! I route questions to Research, Coding, or Data specialists. What would you like to explore?", "assistant"); });