import re
import unicodedata


ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\ufeff]")
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
LINEBREAK_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = ZERO_WIDTH_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WHITESPACE_RE.sub(" ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r" *\n *", "\n", text)
    text = LINEBREAK_RE.sub("\n\n", text)
    return text.strip()


def normalize_inline_text(text: str) -> str:
    return " ".join(normalize_text(text).split())
