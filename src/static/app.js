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
  avatar.innerHTML = role === "assistant"
    ? '<i data-lucide="bot" aria-label="Assistant"></i>'
    : '<i data-lucide="user-round" aria-label="You"></i>';
  const content = document.createElement("div");
  content.className = "message-content";
  const paragraph = document.createElement("p");
  renderMessage(paragraph, text, role);
  content.append(paragraph);
  if (metadata) {
    const meta = document.createElement("span");
    meta.className = "agent-meta";
    meta.textContent = metadata;
    content.append(meta);
  }
  article.append(avatar, content);
  conversation.append(article);
  window.lucide?.createIcons({ attrs: { "stroke-width": 2 } });
  scrollToLatest();
  return article;
}

function updateMessage(article, text, metadata = "") {
  renderMessage(article.querySelector("p"), text, article.classList.contains("assistant-message") ? "assistant" : "user");
  if (metadata) {
    const meta = document.createElement("span");
    meta.className = "agent-meta";
    meta.textContent = metadata;
    article.querySelector(".message-content").append(meta);
  }
  article.classList.remove("streaming-message");
  scrollToLatest();
}

function renderMessage(element, text, role) {
  if (role !== "assistant" || !window.marked || !window.DOMPurify) {
    element.textContent = text;
    return;
  }
  element.innerHTML = window.DOMPurify.sanitize(window.marked.parse(text, { breaks: true }));
  element.querySelectorAll("pre code").forEach((block) => window.hljs?.highlightElement(block));
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
  const pending = appendMessage("Understanding your intent…", "assistant");
  pending.classList.add("streaming-message");
  try {
    const response = await fetch("/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: text }),
    });
    if (!response.ok || !response.body) throw new Error("Unable to start a response stream.");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let streamedAnswer = "";
    while (true) {
      const { value: chunk, done } = await reader.read();
      buffer += decoder.decode(chunk || new Uint8Array(), { stream: !done });
      const events = buffer.split("\n\n");
      buffer = events.pop();
      for (const rawEvent of events) {
        const eventName = rawEvent.match(/^event: (.+)$/m)?.[1];
        const rawData = rawEvent.match(/^data: (.+)$/m)?.[1];
        if (!eventName || !rawData) continue;
        const payload = JSON.parse(rawData);
        if (eventName === "status") updateMessage(pending, payload.message);
        if (eventName === "answer_start") {
          const scores = Object.entries(payload.router_scores).map(([agent, score]) => `${agent} ${Math.round(score * 100)}%`).join(" · ");
          const toolStatus = payload.tools_used?.length ? ` · tools: ${payload.tools_used.join(", ")}` : "";
          updateMessage(pending, "", `${payload.routed_agent} · ${payload.intent_classifier} intent · ${payload.llm_provider_used} · ${scores}${toolStatus}`);
          pending.classList.add("answer-streaming");
        }
        if (eventName === "answer_chunk") {
          streamedAnswer += payload.text;
          renderMessage(pending.querySelector("p"), streamedAnswer, "assistant");
          scrollToLatest();
        }
        if (eventName === "answer_complete") {
          pending.classList.remove("answer-streaming");
        }
        if (eventName === "error") {
          const attempts = Object.entries(payload.attempts || {}).map(([provider, reason]) => `${provider}: ${reason}`).join("; ");
          updateMessage(pending, `${payload.message} (${payload.reason || "provider error"})${attempts ? ` — ${attempts}` : ""}`, "System");
        }
      }
      if (done) break;
    }
  } catch (error) {
    updateMessage(pending, error.message || "Something went wrong. Please try again.", "System");
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