import os
import re
from typing import List, Optional

import httpx

from .schemas import ChatMessage


OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "hf.co/Zkare/Chatbot_Ielts_Assistant_v2:Q4_K_M")
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "700"))


ASSISTANT_STYLE = """You are an IELTS preparation assistant for Vietnamese learners.
Default to Vietnamese unless the user clearly asks for another language or is practicing an English answer.
Write in a warm, natural, and coherent tutoring style.
Open with a brief, friendly sentence when appropriate, then answer directly and explain the reasoning or steps in a logical order.
Avoid robotic, abrupt, or overly terse phrasing.
Use simple Markdown only when it improves readability: short headings, numbered lists, or bullet points.
Do not expose raw Markdown syntax awkwardly, do not overuse emojis, and avoid decorative symbols such as check marks."""


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
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


async def query_ollama(prompt: str, temperature: float = 0.7) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 40,
            "num_ctx": 4096,
            "num_predict": OLLAMA_NUM_PREDICT,
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

Previous conversation:
{history_text}

Current question:
{message}

Answer naturally and keep the conversation context in mind."""

    return f"""{ASSISTANT_STYLE}

Help students with IELTS Reading, Listening, Writing, and Speaking.

Question:
{message}

Answer naturally and clearly."""


def rag_prompt(message: str, context: str, history: Optional[List[ChatMessage]] = None) -> str:
    history_text = format_history(history)
    parts = [
        ASSISTANT_STYLE,
        "Use the study material context below when it is relevant.",
        "If the context does not contain enough information, say so briefly in Vietnamese and then give general IELTS guidance.",
        "",
        f"Study material context:\n{context}",
    ]
    if history_text:
        parts.append(f"Previous conversation:\n{history_text}")
    parts.append(f"Question:\n{message}")
    parts.append("Answer naturally and clearly. Cite source file names when useful.")
    return "\n\n".join(parts)
