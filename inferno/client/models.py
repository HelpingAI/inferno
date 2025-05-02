"""
Data models for the Inferno client.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    """A message in a chat conversation."""
    role: str
    content: str
    name: Optional[str] = None


@dataclass
class CompletionRequest:
    """Request model for text completions."""
    model: str
    prompt: Union[str, List[str]]
    suffix: Optional[str] = None
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    n: int = 1
    stream: bool = False
    logprobs: Optional[int] = None
    echo: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    best_of: int = 1
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


@dataclass
class ChatCompletionRequest:
    """Request model for chat completions."""
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    top_p: float = 0.95
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: int = 256
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


@dataclass
class EmbeddingRequest:
    """Request model for embeddings."""
    model: str
    input: Union[str, List[str]]
    user: Optional[str] = None


@dataclass
class CompletionChoice:
    """A completion choice returned by the API."""
    text: str
    index: int
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


@dataclass
class CompletionResponse:
    """Response model for text completions."""
    id: str
    object: str = "text_completion"
    created: int = field(default_factory=lambda: int(datetime.now().timestamp()))
    model: str = ""
    choices: List[CompletionChoice] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)


@dataclass
class ChatCompletionMessage:
    """A message in a chat completion response."""
    role: str
    content: str
    name: Optional[str] = None


@dataclass
class ChatCompletionChoice:
    """A chat completion choice returned by the API."""
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[str] = None


@dataclass
class ChatCompletionResponse:
    """Response model for chat completions."""
    id: str
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(datetime.now().timestamp()))
    model: str = ""
    choices: List[ChatCompletionChoice] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)


@dataclass
class EmbeddingData:
    """Embedding data returned by the API."""
    embedding: List[float]
    index: int
    object: str = "embedding"


@dataclass
class EmbeddingResponse:
    """Response model for embeddings."""
    object: str = "list"
    data: List[EmbeddingData] = field(default_factory=list)
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)


@dataclass
class ModelData:
    """Model data returned by the API."""
    id: str
    object: str = "model"
    created: int = field(default_factory=lambda: int(datetime.now().timestamp()))
    owned_by: str = "inferno"


@dataclass
class ModelListResponse:
    """Response model for listing models."""
    object: str = "list"
    data: List[ModelData] = field(default_factory=list)
