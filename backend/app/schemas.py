from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    use_rag: bool = True
    conversation_history: Optional[List[ChatMessage]] = None


class ChatResponse(BaseModel):
    response: str
    route_used: str
    sources: Optional[List[Dict[str, Any]]] = None


class UploadResponse(BaseModel):
    message: str
    file_name: str
    chunks_processed: int
    collection_stats: Dict[str, Any]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]


class StatsResponse(BaseModel):
    documents: int
    chunks: int
    embedding_model: str
