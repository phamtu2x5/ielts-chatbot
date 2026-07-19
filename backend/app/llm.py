import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import List, Optional

import httpx

from .config import settings
from .schemas import ChatMessage


OLLAMA_API_URL = settings.ollama_api_url
OLLAMA_MODEL = settings.ollama_model
OLLAMA_NUM_PREDICT = settings.ollama_num_predict
RAG_ROUTE_SENTINEL = "[[USE_RAG]]"


ASSISTANT_STYLE = """You are an IELTS preparation assistant for Vietnamese learners.
Default to Vietnamese unless the user clearly asks for another language or is practicing an English answer.
Write in a concise, neutral, and coherent tutoring style.
Lead with the requested answer or result. Add evidence and brief reasoning only after it.
Do not restate the user's question unless needed for clarity.
Use at most one short introductory sentence, and omit it when the answer can start directly.
Do not repeat the same conclusion at the end.
Avoid robotic, abrupt, or overly terse phrasing.
Use simple Markdown only when it improves readability: short headings, numbered lists, or bullet points.
Use Markdown tables when the user asks for a schedule, comparison, rubric, or other structured information.
Keep Markdown tables simple: no nested bullet lists, no HTML, and no multi-paragraph content inside table cells.
Never output raw HTML tags such as <ul>, <li>, <br>, or <table>; use Markdown instead.
Do not add emojis, generic encouragement, or invitations to ask another question."""


DECORATIVE_ICON_RE = re.compile(
    "[ \\t]*[\u2600-\u27bf\U0001f300-\U0001faff]+[\ufe0f\u200d]*[ \\t]*"
)
VIETNAMESE_CHARACTER_RE = re.compile(
    r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩị"
    r"óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
    re.IGNORECASE,
)
WORD_RANGE_RE = re.compile(
    r"\b(\d{2,4})\s*(?:-|–|—|to|đến|tới)\s*(\d{2,4})\s*(?:words?|từ)\b",
    re.IGNORECASE,
)
WRITING_META_RE = re.compile(
    r"^\s*(?:here(?:'s|\s+is)\s+(?:the|a)\s+(?:revised\s+)?(?:answer|essay|report)|"
    r"below\s+is\s+(?:the|a)\s+(?:revised\s+)?(?:answer|essay|report)|"
    r"đây\s+là\s+(?:bài|bản|đoạn)|dưới\s+đây\s+là\s+(?:bài|bản|đoạn)|"
    r"(?:word\s+count|số\s+từ)\s*[:=-])",
    re.IGNORECASE,
)
EXPLICIT_ENGLISH_RE = re.compile(
    r"(?:bằng|sang|viết|trả\s+lời)\s+(?:ra\s+)?tiếng\s+anh|in\s+english|translate\s+(?:it\s+)?(?:into|to)\s+english",
    re.IGNORECASE,
)
EXPLICIT_VIETNAMESE_RE = re.compile(
    r"(?:bằng|sang|viết|trả\s+lời|dịch)\s+(?:ra\s+)?tiếng\s+việt|in\s+vietnamese|translate\s+(?:it\s+)?(?:into|to)\s+vietnamese",
    re.IGNORECASE,
)
QUESTION_RANGE_RE = re.compile(
    r"\b(?:questions?|câu(?:\s+hỏi)?)\s*(\d{1,3})\s*(?:-|\u2013|\u2014|to|đến|tới)\s*(\d{1,3})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WritingOutputContract:
    language: str
    min_words: int | None
    max_words: int | None
    target_words: tuple[int, int] | None
    single_paragraph: bool
    overview_only: bool = False

    def prompt_lines(self) -> list[str]:
        lines = [f"- Output language: {self.language}."]
        if self.min_words is not None and self.max_words is not None:
            lines.append(f"- Required length: {self.min_words}-{self.max_words} words.")
            lines.append(
                "- Silently verify the final word count before returning; do not stop below the minimum or exceed the maximum."
            )
        if self.target_words is not None:
            lines.append(
                f"- Aim for {self.target_words[0]}-{self.target_words[1]} words so the final response stays safely within the required range."
            )
        if self.single_paragraph:
            lines.append("- Output exactly one paragraph without a heading.")
        if self.overview_only:
            lines.append("- Write only the overview. Do not add an introduction or body details.")
        lines.extend(
            [
                "- Return only the final Writing content. Begin directly with the response.",
                "- Do not add a heading, preface, word-count statement, revision note, or commentary about these instructions.",
            ]
        )
        return lines


@dataclass(frozen=True)
class ResponseOutputContract:
    language: str | None
    forbid_solution: bool
    required_question_numbers: tuple[int, ...] = ()

    def prompt_lines(self) -> list[str]:
        lines: list[str] = []
        if self.language:
            lines.append(f"- Output language: {self.language}.")
        if self.required_question_numbers:
            numbers = ", ".join(str(number) for number in self.required_question_numbers)
            lines.append(f"- Preserve and answer every requested question number: {numbers}.")
        if self.forbid_solution:
            lines.append(
                "- Do not select, infer, eliminate, or hint at any answer. Explain or translate only."
            )
        lines.append("- Return only the requested content, without a generic introduction or invitation.")
        return lines


class OllamaRequestError(RuntimeError):
    def __init__(
        self,
        kind: str,
        message: str,
        *,
        status_code: int | None = None,
        response_body: str | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code
        self.response_body = response_body
        self.attempts = attempts

    def debug_detail(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "message": str(self),
            "status_code": self.status_code,
            "response_body": self.response_body,
            "attempts": self.attempts,
        }


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
    text = DECORATIVE_ICON_RE.sub(
        lambda match: "" if match.start() == 0 or text[match.start() - 1] == "\n" else " ",
        text,
    )
    text = re.sub(r"\[Source\s+\d+\s*:\s*([^\]]+)\]", r"\1", text, flags=re.IGNORECASE)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def writing_output_contract(message: str) -> WritingOutputContract:
    lowered = message.lower()
    requests_vietnamese = bool(
        re.search(
            r"(?:dịch|bằng|sang|viết|trả\s+lời)\s+(?:ra\s+)?tiếng\s+việt|in\s+vietnamese",
            lowered,
        )
    )
    range_match = WORD_RANGE_RE.search(message)
    min_words = int(range_match.group(1)) if range_match else None
    max_words = int(range_match.group(2)) if range_match else None
    overview_only = "overview" in lowered and any(marker in lowered for marker in ["viết", "write"])
    single_paragraph = overview_only or any(
        marker in lowered
        for marker in [
            "viết đoạn",
            "một đoạn",
            "write a paragraph",
            "write an introduction",
            "write a body paragraph",
        ]
    )
    if overview_only and min_words is None:
        min_words, max_words = 40, 80
    target_words = _writing_target_range(min_words, max_words)
    return WritingOutputContract(
        language="Vietnamese" if requests_vietnamese else "English",
        min_words=min_words,
        max_words=max_words,
        target_words=target_words,
        single_paragraph=single_paragraph,
        overview_only=overview_only,
    )


def response_output_contract(
    message: str,
    query_intent: str,
    *,
    allow_solution: bool,
    writing_context: bool = False,
) -> ResponseOutputContract:
    if query_intent == "translate_questions":
        language = "English" if EXPLICIT_ENGLISH_RE.search(message) else "Vietnamese"
    elif writing_context:
        language = writing_output_contract(message).language
    elif EXPLICIT_ENGLISH_RE.search(message):
        language = "English"
    elif EXPLICIT_VIETNAMESE_RE.search(message) or VIETNAMESE_CHARACTER_RE.search(message):
        language = "Vietnamese"
    else:
        language = "English"

    required_numbers: tuple[int, ...] = ()
    if query_intent == "translate_questions":
        match = QUESTION_RANGE_RE.search(message)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            if start <= end and end - start <= 100:
                required_numbers = tuple(range(start, end + 1))

    return ResponseOutputContract(
        language=language,
        forbid_solution=not allow_solution
        and query_intent in {"show_questions", "translate_questions", "explain_questions"},
        required_question_numbers=required_numbers,
    )


def response_output_issues(text: str, contract: ResponseOutputContract) -> list[str]:
    issues: list[str] = []
    letters = re.findall(r"[^\W\d_]", text, flags=re.UNICODE)
    vietnamese_characters = VIETNAMESE_CHARACTER_RE.findall(text)
    if (
        contract.language == "English"
        and len(vietnamese_characters) >= 5
        and len(vietnamese_characters) / max(1, len(letters)) >= 0.02
    ):
        issues.append("The response is not written in English.")
    if contract.language == "Vietnamese" and len(vietnamese_characters) < 2:
        issues.append("The response is not written in Vietnamese.")
    if contract.forbid_solution and likely_contains_solution(text):
        issues.append("The response reveals or narrows an answer despite the no-solution constraint.")
    if contract.required_question_numbers:
        present = {
            int(value)
            for value in re.findall(
                r"(?im)^\s*(?:câu(?:\s+hỏi)?\s*)?(\d{1,3})\s*[.):]",
                text,
            )
        }
        missing = [number for number in contract.required_question_numbers if number not in present]
        if missing:
            issues.append(f"The response is missing question numbers: {missing}.")
    return issues


def response_retry_prompt(original_prompt: str, contract: ResponseOutputContract) -> str:
    contract_text = "\n".join(contract.prompt_lines())
    return f"""{original_prompt}

Generate a fresh response from the original study material context. Do not refer to an earlier draft, validation, or correction.

Final output contract:
{contract_text}

Begin the final response now."""


def response_output_penalty(text: str, contract: ResponseOutputContract) -> tuple[int, int, int]:
    issues = response_output_issues(text, contract)
    return (
        int(any("reveals or narrows" in issue for issue in issues)),
        int(any("not written in" in issue for issue in issues)),
        len(issues),
    )


def _writing_target_range(
    min_words: int | None,
    max_words: int | None,
) -> tuple[int, int] | None:
    if min_words is None or max_words is None or max_words <= min_words:
        return None
    span = max_words - min_words
    target_min = min_words + max(1, round(span * 0.4))
    target_max = max_words - max(1, round(span * 0.3))
    return (target_min, target_max) if target_min <= target_max else (min_words, max_words)


def writing_output_issues(text: str, contract: WritingOutputContract) -> list[str]:
    issues: list[str] = []
    words = re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)
    letters = re.findall(r"[^\W\d_]", text, flags=re.UNICODE)
    vietnamese_characters = VIETNAMESE_CHARACTER_RE.findall(text)
    if (
        contract.language == "English"
        and len(vietnamese_characters) >= 5
        and len(vietnamese_characters) / max(1, len(letters)) >= 0.02
    ):
        issues.append("The response is not written in English.")
    if contract.language == "Vietnamese" and len(vietnamese_characters) < 2:
        issues.append("The response is not written in Vietnamese.")
    if contract.min_words is not None and len(words) < contract.min_words:
        issues.append(f"The response has {len(words)} words, below {contract.min_words}.")
    if contract.max_words is not None and len(words) > contract.max_words:
        issues.append(f"The response has {len(words)} words, above {contract.max_words}.")
    if contract.single_paragraph and len(re.split(r"\n\s*\n", text.strip())) != 1:
        issues.append("The response is not exactly one paragraph.")
    if WRITING_META_RE.search(text):
        issues.append("The response contains meta commentary instead of starting with the Writing content.")
    return issues


def writing_retry_prompt(
    original_prompt: str,
    contract: WritingOutputContract,
) -> str:
    contract_text = "\n".join(contract.prompt_lines())
    return f"""{original_prompt}

Generate a fresh response from the original study material context. Do not refer to any earlier draft, validation, correction, or word count.

Final output contract:
{contract_text}

Begin the final response now."""


def writing_output_penalty(text: str, contract: WritingOutputContract) -> tuple[int, int, int, int]:
    issues = writing_output_issues(text, contract)
    word_count = len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))
    if contract.min_words is not None and word_count < contract.min_words:
        word_distance = contract.min_words - word_count
    elif contract.max_words is not None and word_count > contract.max_words:
        word_distance = word_count - contract.max_words
    else:
        word_distance = 0
    return (
        int(any("meta commentary" in issue for issue in issues)),
        int(any("not written in" in issue for issue in issues)),
        int(any("paragraph" in issue for issue in issues)),
        word_distance,
    )


def select_best_writing_output(
    first: str,
    second: str,
    contract: WritingOutputContract,
) -> str:
    return min((first, second), key=lambda text: writing_output_penalty(text, contract))


def likely_contains_solution(text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in ["đáp án là", "đáp án đúng", "answer is", "correct answer"]):
        return True
    return bool(
        re.search(
            r"(?im)^\s*(?:câu\s*)?\d{1,2}\s*[:=-]\s*(?:[a-h]\b|true\b|false\b|not\s+given\b|\S.{0,40}$)",
            text,
        )
        or re.search(r"(?im)^\s*(?:câu(?:\s+hỏi)?\s*)?\d{1,2}\s*(?:→|->|=>)\s*\S+", text)
        or re.search(r"(?:→|->)\s*(?:[a-h]\b|true\b|false\b|not\s+given\b)", lowered)
        or re.search(
            r"\b(?:loại(?:\s+trừ)?\s+(?:phương\s+án\s+)?[a-h]|(?:không\s+thể|khó)\s+là\s+[a-h]|"
            r"chỉ\s+còn\s+(?:phương\s+án\s+)?[a-h]|phù\s+hợp\s+với\s+(?:phương\s+án\s+)?[a-h]|"
            r"(?:phù\s+hợp|khả\s+năng)\s+(?:nhất\s+)?(?:là\s+)?[a-h])\b",
            lowered,
        )
        or re.search(
            r"(?is)(?:câu(?:\s+hỏi)?\s*)?\d{1,2}.{0,180}?"
            r"(?:không\s+thể\s+(?:xác\s+định|phân\s+loại)|không\s+đủ\s+thông\s+tin)",
            lowered,
        )
    )


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

    data: dict = {}
    async with httpx.AsyncClient(timeout=settings.ollama_timeout_seconds) as client:
        for attempt in range(1, 3):
            try:
                response = await client.post(OLLAMA_API_URL, json=payload)
                response.raise_for_status()
                data = response.json()
                break
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                body = exc.response.text[:500] or None
                if status_code >= 500 and attempt == 1:
                    continue
                raise OllamaRequestError(
                    "http_status",
                    f"Ollama returned HTTP {status_code}.",
                    status_code=status_code,
                    response_body=body,
                    attempts=attempt,
                ) from exc
            except httpx.RequestError as exc:
                if attempt == 1:
                    continue
                raise OllamaRequestError(
                    "transport",
                    f"{type(exc).__name__}: {exc}",
                    attempts=attempt,
                ) from exc
            except (json.JSONDecodeError, ValueError) as exc:
                raise OllamaRequestError(
                    "invalid_json",
                    f"Ollama returned invalid JSON: {exc}",
                    response_body=response.text[:500] or None,
                    attempts=attempt,
                ) from exc

    text = data.get("response") or ""
    text = clean_response(text)
    if looks_like_prompt_echo(text, prompt):
        raise OllamaRequestError("prompt_echo", "Ollama echoed the prompt instead of answering.")
    if not text:
        raise OllamaRequestError("empty_response", "Ollama returned an empty response.")
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
        try:
            stream_context = client.stream("POST", OLLAMA_API_URL, json=payload)
            async with stream_context as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    body = (await response.aread()).decode(errors="replace")[:500] or None
                    raise OllamaRequestError(
                        "http_status",
                        f"Ollama returned HTTP {response.status_code} while streaming.",
                        status_code=response.status_code,
                        response_body=body,
                    ) from exc
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise OllamaRequestError("stream_error", str(data["error"]))
                    token = data.get("response") or data.get("message", {}).get("content") or ""
                    token = re.sub(r"<br\s*/?>", "\n", token, flags=re.IGNORECASE)
                    token = DECORATIVE_ICON_RE.sub("", token)
                    if not token:
                        continue

                    if guard_released:
                        yield token
                        continue

                    guard_buffer += token
                    buffer_prefix = " ".join(guard_buffer.split()).lower()
                    if looks_like_prompt_echo(guard_buffer, prompt):
                        raise OllamaRequestError("prompt_echo", "Ollama echoed the prompt while streaming.")
                    if buffer_prefix and not prompt_prefix.startswith(buffer_prefix):
                        guard_released = True
                        yield guard_buffer
                        guard_buffer = ""
                    elif len(buffer_prefix) >= 220:
                        raise OllamaRequestError("prompt_echo", "Ollama echoed the prompt while streaming.")

                if guard_buffer and not looks_like_prompt_echo(guard_buffer, prompt):
                    yield guard_buffer
        except httpx.RequestError as exc:
            raise OllamaRequestError("transport", f"{type(exc).__name__}: {exc}") from exc


def route_or_answer_prompt(
    message: str,
    history: Optional[List[ChatMessage]] = None,
    document_context: str = "",
) -> str:
    history_text = format_history(history)
    parts = [
        ASSISTANT_STYLE,
        "You are the first-stage gateway for an IELTS chatbot that may have uploaded study materials.",
        "If the current request requires any fact, passage, question, table, image, answer, or evidence from uploaded material, output exactly [[USE_RAG]] and nothing else.",
        "If the request is independent of uploaded material, answer it fully now. The text you return will be shown directly to the user, so do not output a route label.",
        "General greetings, study advice, grammar help, and IELTS strategy questions are independent unless the user explicitly ties them to uploaded material.",
        "Document metadata and retrieval previews below are routing hints only. Never answer a document-grounded request from those previews.",
    ]
    if document_context:
        parts.append(f"Uploaded material routing context:\n{document_context}")
    if history_text:
        parts.append(f"Previous conversation:\n{history_text}")
    parts.extend(
        [
            f"Current user message:\n{message}",
            "Either return the complete direct answer, or return exactly [[USE_RAG]].",
        ]
    )
    return "\n\n".join(parts)


async def route_or_answer(
    message: str,
    history: Optional[List[ChatMessage]] = None,
    document_context: str = "",
) -> tuple[str, str | None]:
    prompt = route_or_answer_prompt(message, history, document_context)
    for attempt in range(2):
        try:
            response = await query_ollama(prompt, temperature=0.2)
            break
        except OllamaRequestError as exc:
            if attempt == 0 and exc.kind in {"empty_response", "prompt_echo"}:
                continue
            raise
    normalized = re.sub(r"^assistant\s*", "", response.strip(), flags=re.IGNORECASE).strip()
    if RAG_ROUTE_SENTINEL in normalized.upper():
        return "rag", None
    return "direct", response


def rag_prompt(
    message: str,
    context: str,
    history: Optional[List[ChatMessage]] = None,
    query_intent: str = "semantic_qa",
    allow_solution: bool = False,
    writing_context: bool = False,
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
    ]
    if query_intent != "writing_generation":
        parts.append("Always cite the source file name and page marker when answering from context.")
    parts.extend(["", f"Study material context:\n{context}"])
    if writing_context:
        parts.extend(
            [
                "Writing response language policy:",
                "- Answer in English by default because the selected material is an IELTS Writing task or sample answer.",
                "- Use Vietnamese only when the user explicitly asks for Vietnamese or requests a translation into Vietnamese.",
            ]
        )
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
                "- Name the task type only when it is explicitly supported by the question instructions in the context. Otherwise describe the instruction without guessing a type.",
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
                "- For multiple-choice questions, compare every supplied option with explicit passage evidence. Do not treat indirect preference, popularity, or possibility as proof of an option.",
                "- If the question refers to a list or answer choices that are missing from context, do not invent a replacement answer or title.",
                "- For True/False/Not Given questions, first classify the relationship between the statement and passage evidence as supports, contradicts, or absent.",
                "- Mapping is strict: supports -> TRUE; contradicts -> FALSE; absent -> NOT GIVEN.",
                "- Do not mark FALSE just because the passage lacks a reason, cause, date, comparison, or detail. If the required detail is absent, the answer is NOT GIVEN.",
                "- If the context only contains question text and lacks passage evidence, say that there is not enough passage evidence to solve reliably.",
                "- For each answer, output the answer first, followed by one short evidence quote and its relationship to the answer.",
                "- Do not use a second conclusion or unsupported elimination. Keep the evidence check concise.",
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
    elif query_intent == "writing_generation":
        parts.extend(
            [
                "Generation policy:",
                "- The user explicitly requested a Writing response based on the supplied prompt or structured visual data.",
                "- Write IELTS Writing content in English by default. Use Vietnamese only when the user explicitly requests Vietnamese or a translation into Vietnamese.",
                "- Use only values, labels, periods, categories, and instructions present in the context.",
                "- Do not substitute a different chart, topic, country, date, or measurement.",
                "- Treat deterministic table facts as authoritative calculations derived from the table.",
                "- Distinguish the highest final value from the largest increase. They may belong to different categories.",
                "- If the user requests only an overview, write only one concise overview paragraph without an introduction or body details.",
            ]
        )
    elif not allow_solution:
        parts.extend(
            [
                "Generation policy:",
                "- Do not provide an answer key or solve exercise questions unless the user explicitly requested it.",
            ]
        )
    if history_text:
        parts.append(f"Previous conversation:\n{history_text}")
    parts.append(f"Question:\n{message}")
    if query_intent == "writing_generation":
        contract = writing_output_contract(message)
        parts.append("Final output contract:\n" + "\n".join(contract.prompt_lines()))
        parts.append("Begin the final Writing response immediately. Output nothing before or after it.")
    else:
        contract = response_output_contract(
            message,
            query_intent,
            allow_solution=allow_solution,
            writing_context=writing_context,
        )
        parts.append("Final output contract:\n" + "\n".join(contract.prompt_lines()))
        parts.append("Answer naturally and clearly, but stay strictly grounded in the provided context.")
    return "\n\n".join(parts)
