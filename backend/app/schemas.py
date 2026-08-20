from typing import List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class ChatUserFact(BaseModel):
    key: str = Field(min_length=1, max_length=80)
    value: str = Field(min_length=1, max_length=500)
    evidence: str = Field(min_length=1, max_length=1_000)


class ChatConversationState(BaseModel):
    last_route: Literal["direct"] = "direct"
    last_intent: str = "direct"
    user_facts: List[ChatUserFact] = Field(default_factory=list, max_length=12)


class ChatRequest(BaseModel):
    session_id: UUID
    message: str = Field(min_length=1, max_length=8_000)
    conversation_history: Optional[List[ChatMessage]] = Field(default=None, max_length=20)
    conversation_state: Optional[ChatConversationState] = None


class SessionDeleteResponse(BaseModel):
    session_id: UUID
    deleted: bool


class SessionExpireResponse(BaseModel):
    session_id: UUID
    scheduled: bool
    expires_in_seconds: int
