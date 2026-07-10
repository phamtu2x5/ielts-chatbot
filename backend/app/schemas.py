from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8_000)
    conversation_history: Optional[List[ChatMessage]] = Field(default=None, max_length=20)


class ChatResponse(BaseModel):
    response: str
    route_used: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    debug: Optional[Dict[str, Any]] = None


class UploadResponse(BaseModel):
    message: str
    file_name: str
    document_id: str
    document_type: str
    chunks_processed: int
    collection_stats: Dict[str, Any]
    debug: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]


class StatsResponse(BaseModel):
    documents: int
    chunks: int
    embedding_model: str
