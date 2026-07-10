import { useEffect, useMemo, useRef, useState } from "react";
import { Bot, Bug, CheckCircle2, FileText, FileUp, Send, Sparkles, UserRound, XCircle } from "lucide-react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./styles.css";

const API_BASE = import.meta.env.VITE_CHATBOT_API_URL || "/api";

const routeLabels = {
  base_model: "Model chính",
  vector_rag: "Tài liệu RAG",
  vector_rag_no_match: "Tài liệu RAG",
  upload: "Tài liệu",
  error: "Lỗi",
};

function routeLabel(route) {
  if (!route || route === "welcome") return "";
  return routeLabels[route] || route;
}

function normalizeMarkdown(content) {
  return content
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/li>\s*<li>/gi, "\n- ")
    .replace(/<ul>\s*<li>/gi, "- ")
    .replace(/<\/li>\s*<\/ul>/gi, "")
    .replace(/<\/?ul>/gi, "")
    .replace(/<\/?li>/gi, "");
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
      {!message.content && message.streamingStatus && (
        <span className="inlineStatus">
          <span className="typingDots compact" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
          {message.streamingStatus}
        </span>
      )}
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

function DebugPanel({ debug, sources }) {
  if (!debug) return null;

  const sourceSummary = (sources || []).map((source) => ({
    file: source.source_file,
    pages: source.pages,
    score: source.score,
    dense: source.probe_dense_score,
    keyword: source.probe_keyword_score,
    question: source.probe_question_score,
    overview: source.probe_overview_score,
    chunk_id: source.chunk_id,
    unit_type: source.metadata?.unit_type,
    chunk_reason: source.metadata?.chunk_reason,
    passage_number: source.metadata?.passage_number,
    question_range: source.metadata?.question_range,
    parent_id: source.metadata?.parent_id,
    preview: (source.display_text || source.text)?.slice(0, 220),
  }));

  return (
    <details className="debugPanel">
      <summary>
        <Bug size={14} />
        Debug pipeline
      </summary>
      <pre>{JSON.stringify({ ...debug, sources: sourceSummary }, null, 2)}</pre>
    </details>
  );
}

function sourceScoreLabel(source) {
  const question = Number(source.probe_question_score || 0);
  const keyword = Number(source.probe_keyword_score || 0);
  const overview = Number(source.probe_overview_score || source.overview_score || 0);
  const dense = Number(source.probe_dense_score || source.score || 0);

  if (question > 0) return `question ${question.toFixed(1)}`;
  if (keyword > 0) return `keyword ${keyword.toFixed(1)}`;
  if (overview > 0) return `overview ${overview.toFixed(1)}`;
  if (dense > 0) return `dense ${dense.toFixed(2)}`;
  return `score ${Number(source.score || 0).toFixed(2)}`;
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
  const hasStreamingAssistant = messages.some((message) => message.streaming);

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
    if (!text || isSending || isUploading) return;

    const assistantId = `assistant-${Date.now()}`;
    setInput("");
    setIsSending(true);
    setMessages((current) => [
      ...current,
      { role: "user", content: text },
      {
        id: assistantId,
        role: "assistant",
        content: "",
        streaming: true,
        streamingStatus: "Đang gửi câu hỏi...",
      },
    ]);

    try {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          conversation_history: history,
        }),
      });
      if (!response.ok || !response.body) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || "Yêu cầu không thành công");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;
          const eventData = JSON.parse(line);
          if (eventData.type === "status") {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId ? { ...message, streamingStatus: eventData.message } : message
              )
            );
          } else if (eventData.type === "metadata") {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? {
                      ...message,
                      route_used: eventData.route_used,
                      sources: eventData.sources || [],
                      debug: eventData.debug,
                    }
                  : message
              )
            );
          } else if (eventData.type === "token") {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? {
                      ...message,
                      content: `${message.content || ""}${eventData.token || ""}`,
                      streamingStatus: "",
                    }
                  : message
              )
            );
          } else if (eventData.type === "done") {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId ? { ...message, streaming: false, streamingStatus: "" } : message
              )
            );
          } else if (eventData.type === "error") {
            throw new Error(eventData.message || "Yêu cầu không thành công");
          }
        }
      }
    } catch (error) {
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                content: error.message,
                route_used: "error",
                streaming: false,
                streamingStatus: "",
              }
            : message
        )
      );
    } finally {
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId ? { ...message, streaming: false, streamingStatus: "" } : message
        )
      );
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
          debug: data.debug,
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
                <DebugPanel debug={message.debug} sources={message.sources} />
                {message.sources?.length > 0 && (
                  <div className="sources">
                    {message.sources.map((source, sourceIndex) => (
                      <details key={`${source.source_file}-${sourceIndex}`}>
                        <summary>
                          {source.source_file}
                          {source.pages?.length ? ` · trang ${source.pages.join(", ")}` : ""} ·{" "}
                          {sourceScoreLabel(source)}
                        </summary>
                        <p>{source.display_text || source.text}</p>
                      </details>
                    ))}
                  </div>
                )}
              </div>
            </article>
          ))}
          {isSending && !hasStreamingAssistant && (
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
            disabled={isUploading || isSending}
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
          <button className="sendButton" type="submit" disabled={isSending || isUploading || !input.trim()}>
            <Send size={18} />
            {isSending ? "Đang gửi" : "Gửi"}
          </button>
        </form>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
