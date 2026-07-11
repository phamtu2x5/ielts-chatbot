import json
import re
from collections.abc import AsyncIterator
from typing import List, Optional

import httpx

from .config import settings
from .schemas import ChatMessage


OLLAMA_API_URL = settings.ollama_api_url
OLLAMA_MODEL = settings.ollama_model
OLLAMA_NUM_PREDICT = settings.ollama_num_predict


ASSISTANT_STYLE = """You are an IELTS preparation assistant for Vietnamese learners.
Default to Vietnamese unless the user clearly asks for another language or is practicing an English answer.
Write in a warm, natural, and coherent tutoring style.
Open with a brief, friendly sentence when appropriate, then answer directly and explain the reasoning or steps in a logical order.
Avoid robotic, abrupt, or overly terse phrasing.
Use simple Markdown only when it improves readability: short headings, numbered lists, or bullet points.
Use Markdown tables when the user asks for a schedule, comparison, rubric, or other structured information.
Keep Markdown tables simple: no nested bullet lists, no HTML, and no multi-paragraph content inside table cells.
Never output raw HTML tags such as <ul>, <li>, <br>, or <table>; use Markdown instead.
A small number of helpful emojis are acceptable when they make the answer easier to read."""


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


def looks_like_prompt_echo(text: str, prompt: str) -> bool:
    cleaned_text = " ".join((text or "").split()).lower()
    cleaned_prompt = " ".join((prompt or "").split()).lower()
    if not cleaned_text:
        return False
    if cleaned_prompt and cleaned_text.startswith(cleaned_prompt[:180]):
        return True
    prompt_markers = [
        "you must answer using only the study material context below",
        "study material context:",
        "generation policy:",
        "previous conversation:",
        "answer naturally and clearly, but stay strictly grounded",
    ]
    return sum(1 for marker in prompt_markers if marker in cleaned_text) >= 2


def _ollama_payload(
    prompt: str,
    stream: bool,
    temperature: float,
    num_predict: Optional[int],
) -> dict:
    return {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 40,
            "num_ctx": settings.ollama_num_ctx,
            "num_predict": num_predict or OLLAMA_NUM_PREDICT,
            "repeat_penalty": 1.1,
        },
    }


async def query_ollama(prompt: str, temperature: float = 0.7, num_predict: Optional[int] = None) -> str:
    payload = _ollama_payload(prompt, stream=False, temperature=temperature, num_predict=num_predict)

    async with httpx.AsyncClient(timeout=settings.ollama_timeout_seconds) as client:
        response = await client.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        data = response.json()

    text = data.get("response") or data.get("thinking") or ""
    text = clean_response(text)
    if looks_like_prompt_echo(text, prompt):
        return ""
    if not text:
        return "Mình sẵn sàng hỗ trợ bạn luyện IELTS. Bạn có thể hỏi cụ thể về Reading, Listening, Writing hoặc Speaking nhé."
    return text


async def stream_ollama(
    prompt: str,
    temperature: float = 0.7,
    num_predict: Optional[int] = None,
) -> AsyncIterator[str]:
    payload = _ollama_payload(prompt, stream=True, temperature=temperature, num_predict=num_predict)
    prompt_prefix = " ".join(prompt.split()).lower()
    guard_buffer = ""
    guard_released = False

    async with httpx.AsyncClient(timeout=settings.ollama_timeout_seconds) as client:
        async with client.stream("POST", OLLAMA_API_URL, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise RuntimeError(str(data["error"]))
                token = data.get("response") or data.get("message", {}).get("content") or ""
                token = re.sub(r"<br\s*/?>", "\n", token, flags=re.IGNORECASE)
                if not token:
                    continue

                if guard_released:
                    yield token
                    continue

                guard_buffer += token
                buffer_prefix = " ".join(guard_buffer.split()).lower()
                if looks_like_prompt_echo(guard_buffer, prompt):
                    return
                if buffer_prefix and not prompt_prefix.startswith(buffer_prefix):
                    guard_released = True
                    yield guard_buffer
                    guard_buffer = ""
                elif len(buffer_prefix) >= 220:
                    return

            if guard_buffer and not looks_like_prompt_echo(guard_buffer, prompt):
                yield guard_buffer


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


def rag_prompt(
    message: str,
    context: str,
    history: Optional[List[ChatMessage]] = None,
    query_intent: str = "semantic_qa",
) -> str:
    history_text = format_history(history)
    if query_intent in {"show_questions", "translate_questions"}:
        history_text = ""
    parts = [
        ASSISTANT_STYLE,
        "You must answer using only the study material context below.",
        "Do not invent passages, questions, people, dates, examples, answer options, or explanations that are not present in the context.",
        "If the context does not contain the requested content, say in Vietnamese that you cannot find it in the uploaded material. Do not give a generic IELTS answer.",
        "If the user asks what the whole document contains, summarize all distinct passages or sections visible in the context. Do not focus on only one passage when multiple passages are present.",
        "Question statements are prompts to be answered; they are not evidence from the passage.",
        "Always cite the source file name and page marker when answering from context.",
        "",
        f"Study material context:\n{context}",
    ]
    if query_intent == "show_questions":
        parts.extend(
            [
                "Generation policy:",
                "- Only list the requested question instructions and question statements.",
                "- Do not mention passage evidence, do not evaluate the statements, and do not explain why any statement is true or false.",
                "- Do not provide TRUE/FALSE/NOT GIVEN labels or answer choices.",
                "- A short Vietnamese meaning for each statement is allowed, but keep it separate from answers.",
            ]
        )
    elif query_intent == "translate_questions":
        parts.extend(
            [
                "Generation policy:",
                "- Translate only the requested question instructions and question statements.",
                "- Do not mention passage evidence, do not evaluate the statements, and do not solve.",
                "- Do not provide TRUE/FALSE/NOT GIVEN labels or answer choices.",
            ]
        )
    elif query_intent == "explain_questions":
        parts.extend(
            [
                "Generation policy:",
                "- Present or explain the requested questions only.",
                "- You may explain the task type, instructions, vocabulary, and Vietnamese meaning.",
                "- Do not solve the questions, do not provide True/False/Not Given labels, do not choose A/B/C/D, and do not infer answers.",
                "- Do not treat the question statements themselves as passage evidence.",
            ]
        )
    elif query_intent == "solve_questions":
        parts.extend(
            [
                "Generation policy:",
                "- The user is asking to solve questions.",
                "- Use passage evidence from the context before giving an answer.",
                "- For True/False/Not Given questions, compare each statement against the passage evidence, then give TRUE, FALSE, or NOT GIVEN with a short reason.",
                "- If the context only contains question text and lacks passage evidence, say that there is not enough passage evidence to solve reliably.",
            ]
        )
    elif query_intent == "document_overview":
        parts.extend(
            [
                "Generation policy:",
                "- Summarize the document from the outline and passage context.",
                "- Mention passage titles, page ranges, and question groups when available.",
                "- Do not answer individual questions or invent answer keys.",
            ]
        )
    if history_text:
        parts.append(f"Previous conversation:\n{history_text}")
    parts.append(f"Question:\n{message}")
    parts.append("Answer naturally and clearly, but stay strictly grounded in the provided context.")
    return "\n\n".join(parts)
