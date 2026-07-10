import React, { useEffect, useMemo, useRef, useState } from "react";
import { Bot, CheckCircle2, FileText, FileUp, Send, Sparkles, UserRound, XCircle } from "lucide-react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
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
  rag: "Tài liệu RAG",
  vector_rag: "Tài liệu RAG",
  upload: "Tài liệu",
  error: "Lỗi",
};

function routeLabel(route) {
  if (!route || route === "welcome") return "";
  return routeLabels[route] || route;
}

function normalizeMarkdown(content) {
  return content.replace(/<br\s*\/?>/gi, "\n");
}

function MessageContent({ message }) {
  if (message.role === "user") {
    return (
      <>
        {message.content && <div className="messageText plainText">{message.content}</div>}
        {message.attachment && <AttachmentCard attachment={message.attachment} />}
      </>
    );
  }

  return (
    <div className="messageText markdownText">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: ({ children }) => (
            <div className="tableScroll">
              <table>{children}</table>
            </div>
          ),
        }}
      >
        {normalizeMarkdown(message.content)}
      </ReactMarkdown>
    </div>
  );
}

function AttachmentCard({ attachment }) {
  const statusText = {
    uploading: "Đang tải lên...",
    ready: `${attachment.chunks || 0} đoạn đã được lập chỉ mục`,
    error: attachment.error || "Không thể tải tệp",
  }[attachment.status];

  return (
    <div className={`attachmentCard ${attachment.status}`}>
      <span className="attachmentIcon">
        <FileText size={20} />
      </span>
      <div className="attachmentMeta">
        <strong>{attachment.name}</strong>
        <span>{statusText}</span>
      </div>
      {attachment.status === "ready" && <CheckCircle2 className="attachmentState" size={18} />}
      {attachment.status === "error" && <XCircle className="attachmentState" size={18} />}
    </div>
  );
}

function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Xin chào, mình là trợ lý IELTS của bạn. Bạn có thể hỏi về Reading, Listening, Writing, Speaking hoặc tải tài liệu lên để mình hỗ trợ phân tích nội dung.",
      route_used: "welcome",
    },
  ]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  const history = useMemo(
    () =>
      messages
        .filter((message) => message.role === "user" || message.role === "assistant")
        .filter((message) => message.content?.trim())
        .slice(-6)
        .map(({ role, content }) => ({ role, content })),
    [messages]
  );

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isSending]);

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

  async function uploadDocument(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    const uploadId = `upload-${Date.now()}-${file.name}`;
    setIsUploading(true);
    setMessages((current) => [
      ...current,
      {
        id: uploadId,
        role: "user",
        content: "",
        attachment: {
          name: file.name,
          status: "uploading",
        },
      },
    ]);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE}/documents/upload`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || "Tải tài liệu không thành công");
      }
      const data = await response.json();
      setMessages((current) =>
        current.map((message) =>
          message.id === uploadId
            ? {
                ...message,
                attachment: {
                  name: data.file_name,
                  status: "ready",
                  chunks: data.chunks_processed,
                },
              }
            : message
        )
      );
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: `Mình đã đọc xong **${data.file_name}**. Bạn có thể hỏi nội dung trong tài liệu này, mình sẽ ưu tiên dùng nguồn đã tải lên để trả lời.`,
          route_used: "upload",
        },
      ]);
    } catch (error) {
      setMessages((current) =>
        current.map((message) =>
          message.id === uploadId
            ? {
                ...message,
                attachment: {
                  name: file.name,
                  status: "error",
                  error: error.message,
                },
              }
            : message
        )
      );
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: error.message,
          route_used: "error",
        },
      ]);
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
              <p>Trợ lý luyện IELTS chạy bằng Ollama, có hỗ trợ hỏi đáp theo tài liệu</p>
            </div>
          </div>
        </header>

        <div className="messages">
          {messages.map((message, index) => (
            <article key={message.id || `${message.role}-${index}`} className={`message ${message.role}`}>
              <div className="avatar">{message.role === "user" ? <UserRound size={17} /> : <Sparkles size={17} />}</div>
              <div className="bubble">
                <MessageContent message={message} />
                {routeLabel(message.route_used) && <div className="route">{routeLabel(message.route_used)}</div>}
                {message.sources?.length > 0 && (
                  <div className="sources">
                    {message.sources.map((source, sourceIndex) => (
                      <details key={`${source.source_file}-${sourceIndex}`}>
                        <summary>
                          {source.source_file}
                          {source.pages?.length ? ` · trang ${source.pages.join(", ")}` : ""} · score{" "}
                          {Number(source.score).toFixed(2)}
                        </summary>
                        <p>{source.text}</p>
                      </details>
                    ))}
                  </div>
                )}
              </div>
            </article>
          ))}
          {isSending && (
            <article className="message assistant">
              <div className="avatar">
                <Sparkles size={17} />
              </div>
              <div className="bubble loadingBubble" aria-live="polite">
                <span className="typingDots" aria-label="Đang trả lời">
                  <span />
                  <span />
                  <span />
                </span>
                <span className="loadingText">Đang suy nghĩ và soạn câu trả lời...</span>
              </div>
            </article>
          )}
          <div ref={messagesEndRef} />
        </div>

        <form className="composer" onSubmit={sendMessage}>
          <button
            className="composerIconButton"
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            title="Tải tài liệu"
          >
            <FileUp size={19} />
            <span>{isUploading ? "Đang tải" : "Tệp"}</span>
          </button>
          <input
            ref={fileInputRef}
            className="hiddenInput"
            type="file"
            accept=".txt,.md,.pdf,.docx,image/png,image/jpeg,image/webp"
            onChange={uploadDocument}
          />
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
