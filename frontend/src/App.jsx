import React, { useMemo, useRef, useState } from "react";
import { Bot, FileUp, Send, Sparkles, UserRound } from "lucide-react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE = import.meta.env.VITE_CHATBOT_API_URL || "/api";

async function apiPost(path, body) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Yêu cầu không thành công");
  }
  return response.json();
}

const routeLabels = {
  base_model: "Model chính",
  base_model_no_rag_match: "Model chính",
  rag: "PDF RAG",
  vector_rag: "PDF RAG",
  error: "Lỗi",
};

function routeLabel(route) {
  if (!route || route === "welcome") return "";
  return routeLabels[route] || route;
}

function renderInline(text) {
  const parts = [];
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g;
  let lastIndex = 0;
  let match;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }

    const token = match[0];
    const key = `${match.index}-${token}`;
    if (token.startsWith("**")) {
      parts.push(<strong key={key}>{token.slice(2, -2)}</strong>);
    } else if (token.startsWith("`")) {
      parts.push(<code key={key}>{token.slice(1, -1)}</code>);
    } else {
      parts.push(<em key={key}>{token.slice(1, -1)}</em>);
    }
    lastIndex = pattern.lastIndex;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts;
}

function renderMarkdown(content) {
  const lines = content.split(/\r?\n/);
  const blocks = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index].trim();

    if (!line) {
      index += 1;
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      const Tag = `h${Math.min(heading[1].length, 4)}`;
      blocks.push(<Tag key={`heading-${index}`}>{renderInline(heading[2])}</Tag>);
      index += 1;
      continue;
    }

    if (/^[-*]\s+/.test(line)) {
      const items = [];
      while (index < lines.length && /^[-*]\s+/.test(lines[index].trim())) {
        items.push(lines[index].trim().replace(/^[-*]\s+/, ""));
        index += 1;
      }
      blocks.push(
        <ul key={`list-${index}`}>
          {items.map((item, itemIndex) => (
            <li key={itemIndex}>{renderInline(item)}</li>
          ))}
        </ul>
      );
      continue;
    }

    if (/^\d+[.)]\s+/.test(line)) {
      const items = [];
      while (index < lines.length && /^\d+[.)]\s+/.test(lines[index].trim())) {
        items.push(lines[index].trim().replace(/^\d+[.)]\s+/, ""));
        index += 1;
      }
      blocks.push(
        <ol key={`ordered-${index}`}>
          {items.map((item, itemIndex) => (
            <li key={itemIndex}>{renderInline(item)}</li>
          ))}
        </ol>
      );
      continue;
    }

    const paragraph = [];
    while (
      index < lines.length &&
      lines[index].trim() &&
      !/^(#{1,4})\s+/.test(lines[index].trim()) &&
      !/^[-*]\s+/.test(lines[index].trim()) &&
      !/^\d+[.)]\s+/.test(lines[index].trim())
    ) {
      paragraph.push(lines[index].trim());
      index += 1;
    }
    blocks.push(<p key={`paragraph-${index}`}>{renderInline(paragraph.join(" "))}</p>);
  }

  return blocks;
}

function MessageContent({ message }) {
  if (message.role === "user") {
    return <div className="messageText plainText">{message.content}</div>;
  }

  return <div className="messageText markdownText">{renderMarkdown(message.content)}</div>;
}

function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Xin chào, mình là trợ lý IELTS của bạn. Bạn có thể hỏi về Reading, Listening, Writing, Speaking hoặc tải PDF lên để mình hỗ trợ phân tích nội dung tài liệu.",
      route_used: "welcome",
    },
  ]);
  const [input, setInput] = useState("");
  const [useRag, setUseRag] = useState(import.meta.env.VITE_CHATBOT_DEFAULT_RAG !== "false");
  const [isSending, setIsSending] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const fileInputRef = useRef(null);

  const history = useMemo(
    () =>
      messages
        .filter((message) => message.role === "user" || message.role === "assistant")
        .slice(-6)
        .map(({ role, content }) => ({ role, content })),
    [messages]
  );

  async function sendMessage(event) {
    event?.preventDefault();
    const text = input.trim();
    if (!text || isSending) return;

    setInput("");
    setIsSending(true);
    setMessages((current) => [...current, { role: "user", content: text }]);

    try {
      const data = await apiPost("/chat", {
        message: text,
        use_rag: useRag,
        conversation_history: history,
      });
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: data.response,
          route_used: data.route_used,
          sources: data.sources || [],
        },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: error.message,
          route_used: "error",
        },
      ]);
    } finally {
      setIsSending(false);
    }
  }

  async function uploadPdf(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadStatus("");
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE}/rag/upload-pdf`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || "Tải PDF không thành công");
      }
      const data = await response.json();
      setUploadStatus(`Đã xử lý ${data.chunks_processed} đoạn từ ${data.file_name}`);
      setUseRag(true);
    } catch (error) {
      setUploadStatus(error.message);
    } finally {
      setIsUploading(false);
      event.target.value = "";
    }
  }

  return (
    <main className="appShell">
      <section className="chatPanel">
        <header className="toolbar">
          <div className="brand">
            <span className="brandIcon">
              <Bot size={22} />
            </span>
            <div>
              <h1>IELTS Chatbot</h1>
              <p>Trợ lý luyện IELTS chạy bằng Ollama, có hỗ trợ hỏi đáp theo PDF</p>
            </div>
          </div>

          <div className="actions">
            <label className="toggle">
              <input type="checkbox" checked={useRag} onChange={(event) => setUseRag(event.target.checked)} />
              <span>RAG</span>
            </label>
            <button className="iconButton" type="button" onClick={() => fileInputRef.current?.click()} disabled={isUploading}>
              <FileUp size={18} />
              {isUploading ? "Đang tải" : "PDF"}
            </button>
            <input ref={fileInputRef} className="hiddenInput" type="file" accept="application/pdf" onChange={uploadPdf} />
          </div>
        </header>

        {uploadStatus && <div className="statusLine">{uploadStatus}</div>}

        <div className="messages">
          {messages.map((message, index) => (
            <article key={`${message.role}-${index}`} className={`message ${message.role}`}>
              <div className="avatar">{message.role === "user" ? <UserRound size={17} /> : <Sparkles size={17} />}</div>
              <div className="bubble">
                <MessageContent message={message} />
                {routeLabel(message.route_used) && <div className="route">{routeLabel(message.route_used)}</div>}
                {message.sources?.length > 0 && (
                  <div className="sources">
                    {message.sources.map((source, sourceIndex) => (
                      <details key={`${source.source_file}-${sourceIndex}`}>
                        <summary>
                          {source.source_file} · score {Number(source.score).toFixed(2)}
                        </summary>
                        <p>{source.text}</p>
                      </details>
                    ))}
                  </div>
                )}
              </div>
            </article>
          ))}
        </div>

        <form className="composer" onSubmit={sendMessage}>
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                sendMessage(event);
              }
            }}
            placeholder="Nhập câu hỏi IELTS..."
            rows={1}
          />
          <button className="sendButton" type="submit" disabled={isSending || !input.trim()}>
            <Send size={18} />
            {isSending ? "Đang gửi" : "Gửi"}
          </button>
        </form>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
