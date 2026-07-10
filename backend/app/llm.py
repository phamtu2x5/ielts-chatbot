import os
import re
from typing import List, Optional

import httpx

from .schemas import ChatMessage


OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "hf.co/Zkare/Chatbot_Ielts_Assistant_v2:Q4_K_M")
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1200"))


ASSISTANT_STYLE = """You are an IELTS preparation assistant for Vietnamese learners.
Default to Vietnamese unless the user clearly asks for another language or is practicing an English answer.
Write in a warm, natural, and coherent tutoring style.
Open with a brief, friendly sentence when appropriate, then answer directly and explain the reasoning or steps in a logical order.
Avoid robotic, abrupt, or overly terse phrasing.
Use simple Markdown only when it improves readability: short headings, numbered lists, or bullet points.
Use Markdown tables when the user asks for a schedule, comparison, rubric, or other structured information.
HTML line breaks and a small number of helpful emojis are acceptable when they make the answer easier to read."""


def format_history(history: Optional[List[ChatMessage]]) -> str:
    if not history:
        return ""

    lines = []
    for msg in history[-6:]:
        role = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def clean_response(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


async def query_ollama(prompt: str, temperature: float = 0.7, num_predict: Optional[int] = None) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 40,
            "num_ctx": 4096,
            "num_predict": num_predict or OLLAMA_NUM_PREDICT,
            "repeat_penalty": 1.1,
        },
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        data = response.json()

    text = data.get("response") or data.get("thinking") or ""
    text = clean_response(text)
    if not text:
        return "Mình sẵn sàng hỗ trợ bạn luyện IELTS. Bạn có thể hỏi cụ thể về Reading, Listening, Writing hoặc Speaking nhé."
    return text


def direct_prompt(message: str, history: Optional[List[ChatMessage]] = None) -> str:
    history_text = format_history(history)
    if history_text:
        return f"""{ASSISTANT_STYLE}

If the user asks about uploaded files, documents, PDFs, page content, question numbers, tables, flow charts, or "nội dung trong tài liệu", do not invent document content. Ask the backend/user to use the uploaded document context instead.

Previous conversation:
{history_text}

Current question:
{message}

Answer naturally and keep the conversation context in mind."""

    return f"""{ASSISTANT_STYLE}

Help students with IELTS Reading, Listening, Writing, and Speaking.
If the user asks about uploaded files, documents, PDFs, page content, question numbers, tables, flow charts, or "nội dung trong tài liệu", do not invent document content. Say that the answer needs the uploaded document context.

Question:
{message}

Answer naturally and clearly."""


def route_prompt(
    message: str,
    history: Optional[List[ChatMessage]] = None,
    document_context: str = "",
) -> str:
    history_text = format_history(history)
    context = f"\nPrevious conversation:\n{history_text}\n" if history_text else ""
    docs = f"\nUploaded document context and retrieval probe:\n{document_context}\n" if document_context else ""
    return f"""You are a strict router for an IELTS chatbot.

Decide whether the assistant should answer directly or use the uploaded document/vector knowledge base.

Choose "rag" when the user asks about:
- the uploaded file, document, PDF, DOCX, image, material, source, lesson, or text
- a summary, explanation, extraction, comparison, or question based on uploaded material
- "dựa vào tài liệu", "trong file", "PDF", "DOCX", "ảnh", "nội dung trên", or similar references

Choose "direct" for general IELTS advice, greetings, study plans, grammar explanations, writing/speaking tips, or anything that does not need uploaded material.

Use the uploaded document context and retrieval probe carefully:
- If uploaded documents exist and the current question asks for file content, question numbers, page content, passages, tables, flow charts, summaries, explanations, or answers based on the material, choose "rag".
- If the retrieval probe strength is strong or the probe hits clearly mention the requested content, choose "rag".
- If the retrieval probe is weak_or_none and the question is clearly general IELTS advice, choose "direct".
- Choose "direct" only when the question is clearly independent from uploaded material.
{context}
{docs}
Current user message:
{message}

Return exactly one word: direct or rag."""


async def classify_route(
    message: str,
    history: Optional[List[ChatMessage]] = None,
    document_context: str = "",
) -> str:
    decision = await query_ollama(route_prompt(message, history, document_context), temperature=0.0, num_predict=8)
    decision = decision.strip().lower()
    if "rag" in decision and "direct" not in decision:
        return "rag"
    return "direct"


def rag_prompt(message: str, context: str, history: Optional[List[ChatMessage]] = None) -> str:
    history_text = format_history(history)
    parts = [
        ASSISTANT_STYLE,
        "You must answer using only the study material context below.",
        "Do not invent passages, questions, people, dates, examples, answer options, or explanations that are not present in the context.",
        "If the context does not contain the requested content, say in Vietnamese that you cannot find it in the uploaded material. Do not give a generic IELTS answer.",
        "Always cite the source file name and page marker when answering from context.",
        "",
        f"Study material context:\n{context}",
    ]
    if history_text:
        parts.append(f"Previous conversation:\n{history_text}")
    parts.append(f"Question:\n{message}")
    parts.append("Answer naturally and clearly, but stay strictly grounded in the provided context.")
    return "\n\n".join(parts)
