"""
Data models for the Inferno client.
"""

import re
from typing import List, Dict, Any, Optional, Union, Literal
from dataclasses import dataclass, field
from datetime import datetime

# --- Multimodal Content Parts ---

@dataclass
class TextContentPart:
    """Text content part for multimodal messages."""
    type: Literal["text"] = "text"
    text: str

@dataclass
class ImageUrl:
    """Image URL detail for multimodal messages."""
    url: str
    detail: Optional[Literal["low", "high", "auto"]] = "auto"

    def __post_init__(self):
        # Validation for data URI or standard HTTP/HTTPS URL
        if self.url.startswith("data:image"):
            if not re.match(r"data:image/(?:jpeg|png|gif|webp);base64,", self.url):
                # Allow common image types, adjust regex if more are needed
                raise ValueError("Invalid image data URI format. Expected data:image/[jpeg|png|gif|webp];base64,...")
        elif not self.url.startswith(("http://", "https://")):
             raise ValueError("Invalid image URL format. Expected data URI or http/https URL.")

@dataclass
class ImageUrlContentPart:
    """Image URL content part for multimodal messages."""
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


# --- Core Message Structure ---

@dataclass
class Message:
    """
    A message in a chat conversation.
    Content can be a simple string or a list of multimodal content parts.
    """
    role: str
    content: Union[str, List[Union[TextContentPart, ImageUrlContentPart, Dict[str, Any]]]] # Dict for flexibility
    name: Optional[str] = None


# --- Tool Calling Structures ---

@dataclass
class Function:
    """Function definition for tool calling."""
    name: str
    parameters: Dict[str, Any] # JSON Schema object
    description: Optional[str] = None

@dataclass
class Tool:
    """Tool definition for tool calling."""
    type: Literal["function"] = "function"
    function: Function

@dataclass
class ToolChoiceFunction:
    """Specific function choice for tool_choice."""
    name: str

# Type alias for tool_choice, matching OpenAI's options
ToolChoice = Union[Literal["none", "auto"], ToolChoiceFunction, Dict[str, Any]] # Dict for flexibility


# --- Request Models ---

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
    messages: List[Message] # Updated to use the Message dataclass
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
    tools: Optional[List[Tool]] = None # Added tools
    tool_choice: Optional[ToolChoice] = None # Added tool_choice


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
class FunctionCall:
    """Function call details in a response message."""
    name: str
    arguments: str # JSON string arguments

@dataclass
class ToolCall:
    """Tool call details in a response message."""
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall

@dataclass
class ChatCompletionMessage:
    """A message in a chat completion response."""
    role: str
    content: Optional[str] = None # Content is optional if tool_calls are present
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None # Added tool_calls


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
