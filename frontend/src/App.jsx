import { useEffect, useMemo, useRef, useState } from "react";
import {
  Bot,
  Bug,
  CheckCircle2,
  Download,
  FileText,
  Paperclip,
  Send,
  Sparkles,
  UserRound,
  X,
  XCircle,
} from "lucide-react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./styles.css";

const API_BASE = import.meta.env.VITE_CHATBOT_API_URL || "/api";

const routeLabels = {
  base_model: "Model chính",
  vector_rag: "Tài liệu RAG",
  vector_rag_static: "Tài liệu RAG",
  vector_rag_no_match: "Tài liệu RAG",
  vector_rag_ambiguous_document: "Tài liệu RAG",
  upload: "Tài liệu",
  error: "Lỗi",
};

function routeLabel(route) {
  if (!route || route === "welcome") return "";
  return routeLabels[route] || route;
}

function normalizeMarkdown(content) {
  return (content || "")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/li>\s*<li>/gi, "\n- ")
    .replace(/<ul>\s*<li>/gi, "- ")
    .replace(/<\/li>\s*<\/ul>/gi, "")
    .replace(/<\/?ul>/gi, "")
    .replace(/<\/?li>/gi, "");
}

function safeFilename(value) {
  return (value || "rag-debug")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-zA-Z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
}

function downloadJson(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function affinityFromMetadata(eventData) {
  if (!eventData.route_used?.startsWith("vector_rag")) return null;

  const sources = eventData.sources || [];
  const resolvedIds = eventData.debug?.target_resolution?.resolved_document_ids || [];
  const documentIds = [
    ...new Set([
      ...sources.map((source) => source.document_id).filter(Boolean),
      ...resolvedIds,
    ]),
  ];
  if (!documentIds.length) return null;

  const passageNumbers = [
    ...new Set(
      sources
        .map((source) => source.metadata?.passage_number)
        .filter((value) => Number.isInteger(value))
    ),
  ];
  const questionRanges = [];
  const seenRanges = new Set();
  for (const source of sources) {
    const range = source.metadata?.question_range;
    if (!Array.isArray(range) || range.length !== 2) continue;
    const key = `${range[0]}-${range[1]}`;
    if (seenRanges.has(key)) continue;
    seenRanges.add(key);
    questionRanges.push(range);
  }
  return {
    document_ids: documentIds,
    passage_numbers: passageNumbers,
    question_ranges: questionRanges,
  };
}

function MessageContent({ message }) {
  const content = message.content || "";

  if (message.role === "user") {
    const attachments = message.attachments || (message.attachment ? [message.attachment] : []);
    return (
      <>
        {content && <div className="messageText plainText">{content}</div>}
        {attachments.length > 0 && (
          <div className="messageAttachments">
            {attachments.map((attachment) => (
              <AttachmentCard key={attachment.id || attachment.name} attachment={attachment} />
            ))}
          </div>
        )}
      </>
    );
  }

  const showStatus = !content && message.streamingStatus;
  const showEmptyFallback = !content && !message.streaming && !message.streamingStatus;

  return (
    <div className="messageText markdownText">
      {showStatus && (
        <span className="inlineStatus">
          <span className="typingDots compact" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
          {message.streamingStatus}
        </span>
      )}
      {content && (
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
          {normalizeMarkdown(content)}
        </ReactMarkdown>
      )}
      {showEmptyFallback && <span className="emptyAnswer">Chưa nhận được nội dung trả lời.</span>}
    </div>
  );
}

function AttachmentCard({ attachment, onRemove }) {
  const statusText = {
    queued: "Sẵn sàng gửi",
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
      {onRemove && (
        <button
          className="attachmentRemoveButton"
          type="button"
          title={`Bỏ ${attachment.name}`}
          aria-label={`Bỏ ${attachment.name}`}
          onClick={onRemove}
        >
          <X size={16} />
        </button>
      )}
    </div>
  );
}

function DebugPanel({ debug, sources, onDownload }) {
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
        <span className="debugSummaryTitle">
          <Bug size={14} />
          Debug pipeline
        </span>
        {onDownload && (
          <button
            className="debugDownloadButton"
            type="button"
            title="Tải câu hỏi, câu trả lời và debug"
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              onDownload();
            }}
          >
            <Download size={14} />
          </button>
        )}
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
  const [pendingFiles, setPendingFiles] = useState([]);
  const [activeDocumentIds, setActiveDocumentIds] = useState([]);
  const [conversationAffinity, setConversationAffinity] = useState(null);
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

  function exportDebug(message, index) {
    const previousQuestion = [...messages.slice(0, index)]
      .reverse()
      .find((item) => item.role === "user" && item.content?.trim());
    const debug = message.debug || {};
    const queryIntent = debug.query_intent || debug.probe?.query_intent || null;
    const routeDecision = debug.route_decision || message.route_used || null;
    const payload = {
      exported_at: new Date().toISOString(),
      question: previousQuestion?.content || "",
      answer: message.content || "",
      route_used: message.route_used || null,
      route_decision: routeDecision,
      query_intent: queryIntent,
      debug,
      sources: message.sources || [],
      source_previews: (message.sources || []).map((source) => ({
        file: source.source_file,
        pages: source.pages,
        score: source.score,
        dense: source.probe_dense_score,
        keyword: source.probe_keyword_score,
        question: source.probe_question_score,
        overview: source.probe_overview_score,
        chunk_id: source.chunk_id,
        unit_type: source.metadata?.unit_type,
        passage_number: source.metadata?.passage_number,
        question_range: source.metadata?.question_range,
        text: source.display_text || source.text || "",
      })),
    };
    const suffix = safeFilename(previousQuestion?.content || message.route_used || "rag-debug");
    downloadJson(`ielts-chatbot-debug-${suffix}-${Date.now()}.json`, payload);
  }

  function selectFiles(event) {
    const selectedFiles = Array.from(event.target.files || []);
    if (!selectedFiles.length) return;

    setPendingFiles((current) => {
      const existing = new Set(current.map((item) => item.id));
      const additions = selectedFiles
        .map((file) => ({
          id: `${file.name}-${file.size}-${file.lastModified}`,
          file,
          name: file.name,
          status: "queued",
        }))
        .filter((item) => !existing.has(item.id));
      return [...current, ...additions];
    });
    event.target.value = "";
  }

  function updateAttachment(messageId, attachmentId, changes) {
    setMessages((current) =>
      current.map((message) =>
        message.id === messageId
          ? {
              ...message,
              attachments: (message.attachments || []).map((attachment) =>
                attachment.id === attachmentId ? { ...attachment, ...changes } : attachment
              ),
            }
          : message
      )
    );
  }

  async function uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetch(`${API_BASE}/documents/upload`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Tải tài liệu không thành công");
    }
    return response.json();
  }

  async function sendMessage(event) {
    event?.preventDefault();
    const text = input.trim();
    const queuedFiles = pendingFiles;
    if ((!text && !queuedFiles.length) || isSending || isUploading) return;

    const submissionId = Date.now();
    const userId = `user-${submissionId}`;
    const assistantId = `assistant-${submissionId}`;
    setInput("");
    setPendingFiles([]);
    setIsSending(true);
    setMessages((current) => [
      ...current,
      {
        id: userId,
        role: "user",
        content: text,
        attachments: queuedFiles.map(({ id, name }) => ({ id, name, status: "queued" })),
      },
      {
        id: assistantId,
        role: "assistant",
        content: "",
        streaming: true,
        streamingStatus: queuedFiles.length ? "Đang chuẩn bị tài liệu..." : "Đang gửi câu hỏi...",
      },
    ]);

    try {
      const uploadedFiles = [];
      const failedFiles = [];

      if (queuedFiles.length) {
        setIsUploading(true);
        for (const [index, item] of queuedFiles.entries()) {
          updateAttachment(userId, item.id, { status: "uploading" });
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? {
                    ...message,
                    streamingStatus: `Đang xử lý tài liệu ${index + 1}/${queuedFiles.length}: ${item.name}`,
                  }
                : message
            )
          );
          try {
            const data = await uploadFile(item.file);
            uploadedFiles.push(data);
            updateAttachment(userId, item.id, {
              name: data.file_name,
              status: "ready",
              chunks: data.chunks_processed,
              documentId: data.document_id,
            });
          } catch (error) {
            failedFiles.push({ name: item.name, error: error.message });
            updateAttachment(userId, item.id, {
              status: "error",
              error: error.message,
            });
          }
        }
        setIsUploading(false);
        if (uploadedFiles.length) {
          setActiveDocumentIds((current) => [
            ...new Set([...current, ...uploadedFiles.map((data) => data.document_id)]),
          ]);
          setConversationAffinity(
            uploadedFiles.length === 1
              ? {
                  document_ids: [uploadedFiles[0].document_id],
                  passage_numbers: [],
                  question_ranges: [],
                }
              : null
          );
        }
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId
              ? {
                  ...message,
                  debug: {
                    ...(message.debug || {}),
                    uploads: {
                      succeeded: uploadedFiles.map((data) => ({
                        file_name: data.file_name,
                        document_id: data.document_id,
                        chunks_processed: data.chunks_processed,
                        debug: data.debug,
                      })),
                      failed: failedFiles,
                    },
                  },
                }
              : message
          )
        );
      }

      if (!text) {
        const readyNames = uploadedFiles.map((data) => `**${data.file_name}**`).join(", ");
        const failedNames = failedFiles.map((item) => `**${item.name}**`).join(", ");
        const parts = [];
        if (readyNames) parts.push(`Đã xử lý xong ${uploadedFiles.length} tài liệu: ${readyNames}.`);
        if (failedNames) parts.push(`Không thể xử lý ${failedFiles.length} tài liệu: ${failedNames}.`);
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId
              ? {
                  ...message,
                  content: parts.join("\n\n"),
                  route_used: uploadedFiles.length ? "upload" : "error",
                  streaming: false,
                  streamingStatus: "",
                }
              : message
          )
        );
        return;
      }

      if (failedFiles.length) {
        throw new Error("Chưa gửi câu hỏi vì chưa xử lý thành công toàn bộ tài liệu đính kèm.");
      }

      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId ? { ...message, streamingStatus: "Đang gửi câu hỏi..." } : message
        )
      );
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          conversation_history: history,
          document_ids: uploadedFiles.length
            ? uploadedFiles.map((data) => data.document_id)
            : activeDocumentIds,
          affinity: uploadedFiles.length ? null : conversationAffinity,
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
          let eventData;
          try {
            eventData = JSON.parse(line);
          } catch {
            throw new Error("Dữ liệu stream từ backend không hợp lệ");
          }
          if (eventData.type === "status") {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId ? { ...message, streamingStatus: eventData.message } : message
              )
            );
          } else if (eventData.type === "metadata") {
            const nextAffinity = affinityFromMetadata(eventData);
            if (nextAffinity) setConversationAffinity(nextAffinity);
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? {
                      ...message,
                      route_used: eventData.route_used,
                      sources: eventData.sources || [],
                      debug: { ...(message.debug || {}), ...(eventData.debug || {}) },
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
                message.id === assistantId
                  ? {
                      ...message,
                      content: message.content || "Mình chưa nhận được nội dung trả lời. Vui lòng thử lại.",
                      streaming: false,
                      streamingStatus: "",
                    }
                  : message
              )
            );
          } else if (eventData.type === "error") {
            if (eventData.detail) {
              setMessages((current) =>
                current.map((message) =>
                  message.id === assistantId
                    ? {
                        ...message,
                        debug: { ...(message.debug || {}), generation_error: eventData.detail },
                      }
                    : message
                )
              );
            }
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
      setIsUploading(false);
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
                <DebugPanel
                  debug={message.debug}
                  sources={message.sources}
                  onDownload={message.debug ? () => exportDebug(message, index) : null}
                />
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
          {pendingFiles.length > 0 && (
            <div className="pendingAttachments" aria-label="Tệp đính kèm đang chờ gửi">
              {pendingFiles.map((item) => (
                <AttachmentCard
                  key={item.id}
                  attachment={item}
                  onRemove={() => setPendingFiles((current) => current.filter((file) => file.id !== item.id))}
                />
              ))}
            </div>
          )}
          <div className="composerControls">
            <button
              className="composerIconButton"
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading || isSending}
              title="Đính kèm tệp"
              aria-label="Đính kèm tệp"
            >
              <Paperclip size={19} />
            </button>
            <input
              ref={fileInputRef}
              className="hiddenInput"
              type="file"
              multiple
              accept=".txt,.md,.pdf,.docx,image/png,image/jpeg,image/webp"
              onChange={selectFiles}
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
            <button
              className="sendButton"
              type="submit"
              disabled={isSending || isUploading || (!input.trim() && !pendingFiles.length)}
              title={isSending || isUploading ? "Đang xử lý" : "Gửi"}
              aria-label={isSending || isUploading ? "Đang xử lý" : "Gửi"}
            >
              <Send size={18} />
            </button>
          </div>
        </form>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
