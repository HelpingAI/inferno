"""
Inferno Client - OpenAI-compatible Python client for Inferno

This package provides a Python client for interacting with Inferno's API,
with an interface compatible with the OpenAI Python client.
"""

from .client import (
    InfernoClient,
    Completion,
    ChatCompletion,
    Embedding,
    Model
)
from .exceptions import (
    InfernoError,
    InfernoAPIError,
    InfernoConnectionError,
    InfernoTimeoutError
)
from .models import (
    CompletionRequest,
    ChatCompletionRequest,
    EmbeddingRequest,
    CompletionResponse,
    ChatCompletionResponse,
    EmbeddingResponse,
    ModelListResponse
)

__all__ = [
    # Client classes
    "InfernoClient",
    "Completion",
    "ChatCompletion",
    "Embedding",
    "Model",
    
    # Exceptions
    "InfernoError",
    "InfernoAPIError",
    "InfernoConnectionError",
    "InfernoTimeoutError",
    
    # Models
    "CompletionRequest",
    "ChatCompletionRequest",
    "EmbeddingRequest",
    "CompletionResponse",
    "ChatCompletionResponse",
    "EmbeddingResponse",
    "ModelListResponse"
]
