import os
import re
from typing import List, Optional

import httpx

from .schemas import ChatMessage


OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "hf.co/Zkare/Chatbot_Ielts_Assistant_v2:Q4_K_M")


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
            "num_predict": 1200,
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
        return "I am ready to help you with IELTS preparation. Please ask a specific question."
    return text


def direct_prompt(message: str, history: Optional[List[ChatMessage]] = None) -> str:
    history_text = format_history(history)
    if history_text:
        return f"""You are an IELTS preparation assistant.

Previous conversation:
{history_text}

Current question:
{message}

Answer clearly and helpfully. Keep the conversation context in mind."""

    return f"""You are an IELTS preparation assistant. Help students with IELTS Reading, Listening, Writing, and Speaking.

Question:
{message}

Answer clearly and helpfully."""


def rag_prompt(message: str, context: str, history: Optional[List[ChatMessage]] = None) -> str:
    history_text = format_history(history)
    parts = [
        "You are an IELTS preparation assistant.",
        "Use the study material context below when it is relevant.",
        "If the context does not contain enough information, say so briefly and then give general IELTS guidance.",
        "",
        f"Study material context:\n{context}",
    ]
    if history_text:
        parts.append(f"Previous conversation:\n{history_text}")
    parts.append(f"Question:\n{message}")
    parts.append("Answer clearly and cite source file names when useful.")
    return "\n\n".join(parts)
