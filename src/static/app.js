const conversation = document.querySelector("#conversation");
const form = document.querySelector("#chat-form");
const query = document.querySelector("#query");

const CONVERSATION_STORAGE_KEY = "agents_ai_conversation_id";

let conversationId = sessionStorage.getItem(CONVERSATION_STORAGE_KEY);

if (!conversationId) {
  conversationId = crypto.randomUUID();
  sessionStorage.setItem(
    CONVERSATION_STORAGE_KEY,
    conversationId,
  );
}


function saveConversationId(value) {
  if (!value) {
    return;
  }

  conversationId = value;

  sessionStorage.setItem(
    CONVERSATION_STORAGE_KEY,
    conversationId,
  );
}


function startNewConversation() {
  conversationId = crypto.randomUUID();

  sessionStorage.setItem(
    CONVERSATION_STORAGE_KEY,
    conversationId,
  );

  conversation.innerHTML = "";
  query.focus();
}


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

  window.lucide?.createIcons({
    attrs: {
      "stroke-width": 2,
    },
  });

  scrollToLatest();

  return article;
}


function updateMessage(article, text, metadata = "") {
  const role = article.classList.contains("assistant-message")
    ? "assistant"
    : "user";

  const paragraph = article.querySelector("p");

  if (paragraph) {
    renderMessage(paragraph, text, role);
  }

  if (metadata) {
    const existingMetadata = article.querySelector(".agent-meta");

    if (existingMetadata) {
      existingMetadata.textContent = metadata;
    } else {
      const meta = document.createElement("span");
      meta.className = "agent-meta";
      meta.textContent = metadata;

      article.querySelector(".message-content")?.append(meta);
    }
  }

  article.classList.remove("streaming-message");

  scrollToLatest();
}


function renderMessage(element, text, role) {
  if (role !== "assistant" || !window.marked || !window.DOMPurify) {
    element.textContent = text;
    return;
  }

  element.innerHTML = window.DOMPurify.sanitize(
    window.marked.parse(text, {
      breaks: true,
    }),
  );

  element.querySelectorAll("pre code").forEach((block) => {
    window.hljs?.highlightElement(block);
  });
}


function setLoading(isLoading) {
  const button = form.querySelector("button");

  if (!button) {
    return;
  }

  button.disabled = isLoading;

  const buttonLabel = button.querySelector("span");

  if (buttonLabel) {
    buttonLabel.textContent = isLoading
      ? "Thinking"
      : "Send";
  }
}


function parseSseEvents(buffer) {
  const events = buffer.split(/\r?\n\r?\n/);
  const remainingBuffer = events.pop() || "";

  const parsedEvents = [];

  for (const rawEvent of events) {
    const eventName = rawEvent.match(
      /^event:\s*(.+)$/m,
    )?.[1];

    const rawData = rawEvent
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n");

    if (!eventName || !rawData) {
      continue;
    }

    try {
      parsedEvents.push({
        name: eventName,
        payload: JSON.parse(rawData),
      });
    } catch (error) {
      console.error(
        "Unable to parse SSE payload:",
        rawData,
        error,
      );
    }
  }

  return {
    events: parsedEvents,
    remainingBuffer,
  };
}


function buildMetadata(payload) {
  const scores = Object.entries(
    payload.router_scores || {},
  )
    .map(
      ([agent, score]) =>
        `${agent} ${Math.round(Number(score) * 100)}%`,
    )
    .join(" · ");

  const toolStatus = payload.tools_used?.length
    ? ` · tools: ${payload.tools_used.join(", ")}`
    : "";

  const metadata = [
    payload.routed_agent,
    payload.intent_classifier
      ? `${payload.intent_classifier} intent`
      : "",
    payload.llm_provider_used,
    scores,
  ]
    .filter(Boolean)
    .join(" · ");

  return `${metadata}${toolStatus}`;
}


function renderStreamEvent(
  eventName,
  payload,
  pending,
  streamState,
) {
  if (payload.conversation_id) {
    saveConversationId(payload.conversation_id);
  }

  if (eventName === "status") {
    updateMessage(
      pending,
      payload.message || "Working…",
    );

    return;
  }

  if (eventName === "answer_start") {
    updateMessage(
      pending,
      "",
      buildMetadata(payload),
    );

    pending.classList.add("answer-streaming");

    return;
  }

  if (eventName === "answer_chunk") {
    streamState.answer += payload.text || "";

    const paragraph = pending.querySelector("p");

    if (paragraph) {
      renderMessage(
        paragraph,
        streamState.answer,
        "assistant",
      );
    }

    scrollToLatest();

    return;
  }

  if (eventName === "answer_complete") {
    pending.classList.remove("answer-streaming");
    pending.classList.remove("streaming-message");

    return;
  }

  if (eventName === "error") {
    const attempts = Object.entries(payload.attempts || {})
      .map(
        ([provider, reason]) =>
          `${provider}: ${reason}`,
      )
      .join("; ");

    const message = `${payload.message || "Request failed"} (${
      payload.reason || "provider error"
    })${attempts ? ` — ${attempts}` : ""}`;

    updateMessage(pending, message);
  }
}


async function submitQuery(value) {
  const text = value.trim();

  if (!text) {
    return;
  }

  appendMessage(text, "user");

  query.value = "";
  query.style.height = "auto";

  setLoading(true);

  const pending = appendMessage(
    "Understanding your intent…",
    "assistant",
  );

  pending.classList.add("streaming-message");

  try {
    const response = await fetch("/query/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: text,
        conversation_id: conversationId,
      }),
    });

    if (!response.ok || !response.body) {
      throw new Error(
        "Unable to start a response stream.",
      );
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let buffer = "";

    const streamState = {
      answer: "",
    };

    while (true) {
      const { value: chunk, done } = await reader.read();

      buffer += decoder.decode(
        chunk || new Uint8Array(),
        {
          stream: !done,
        },
      );

      const parsed = parseSseEvents(buffer);
      buffer = parsed.remainingBuffer;

      for (const streamEvent of parsed.events) {
        renderStreamEvent(
          streamEvent.name,
          streamEvent.payload,
          pending,
          streamState,
        );
      }

      if (done) {
        if (buffer.trim()) {
          console.warn(
            "SSE stream ended with an incomplete event:",
            buffer,
          );
        }

        break;
      }
    }

    if (!streamState.answer.trim()) {
      updateMessage(
        pending,
        "The server completed the request without returning answer text.",
      );
    }
  } catch (error) {
    const message = error instanceof Error
      ? error.message
      : "Something went wrong. Please try again.";

    updateMessage(pending, message);
  } finally {
    setLoading(false);
    query.focus();
  }
}


form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuery(query.value);
});


query.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});


query.addEventListener("input", () => {
  query.style.height = "auto";

  query.style.height = `${
    Math.min(query.scrollHeight, 150)
  }px`;
});


document.querySelectorAll("[data-query]").forEach((button) => {
  button.addEventListener("click", () => {
    submitQuery(button.dataset.query || "");
  });
});