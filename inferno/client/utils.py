"""
Utility functions for the Inferno client.
"""

import json
import uuid
from typing import Dict, Any, List, Union, Optional, Generator
from datetime import datetime
from dataclasses import asdict, is_dataclass

from .exceptions import InfernoAPIError
from .models import Message, TextContentPart, ImageUrlContentPart # Import necessary models


# Type alias for input messages
MessagesInput = Union[List[Message], List[Dict[str, Any]]]


def generate_request_id() -> str:
    """
    Generate a unique request ID.
    
    Returns:
        str: A unique request ID
    """
    return f"req_{uuid.uuid4().hex}"


def current_timestamp() -> int:
    """
    Get the current timestamp in seconds.
    
    Returns:
        int: Current timestamp in seconds
    """
    return int(datetime.now().timestamp())


def parse_stream_response(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a line from a streaming response.
    
    Args:
        line: Line from the streaming response
        
    Returns:
        Optional[Dict[str, Any]]: Parsed response or None if the line is empty or "[DONE]"
    """
    if not line or line.strip() == "" or line.strip() == "[DONE]":
        return None
    
    # Remove "data: " prefix if present
    if line.startswith("data: "):
        line = line[6:]
    
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def handle_error_response(response: Dict[str, Any]) -> None:
    """
    Handle an error response from the API.
    
    Args:
        response: Response from the API
        
    Raises:
        InfernoAPIError: If the response contains an error
    """
    if "error" in response:
        error = response["error"]
        message = error.get("message", "Unknown error")
        code = error.get("code", "unknown_error")
        status = error.get("status", 500)
        raise InfernoAPIError(status, f"{code}: {message}", response)


# def prepare_messages(messages: MessagesInput) -> List[Dict[str, Any]]:
#     """
#     Prepare messages for a chat completion request, handling multimodal content.
#     Accepts either a list of Message dataclasses or a list of dictionaries.

#     Args:
#         messages: List of Message objects or dictionaries.

#     Returns:
#         List[Dict[str, Any]]: Prepared messages in the dictionary format expected by the API.
#     """
#     prepared_messages = []
#     for message in messages:
#         if isinstance(message, Message):
#             # Handle Message dataclass input
#             prepared_message: Dict[str, Any] = {"role": message.role}
#             if message.name:
#                 prepared_message["name"] = message.name

#             # Handle content (string or list of parts)
#             if isinstance(message.content, str):
#                 prepared_message["content"] = message.content
#             elif isinstance(message.content, list):
#                 prepared_content = []
#                 for part in message.content:
#                     if isinstance(part, (TextContentPart, ImageUrlContentPart)):
#                         # Convert dataclass part to dict
#                         prepared_content.append(asdict(part))
#                     elif isinstance(part, dict):
#                         # Assume it's already in the correct dict format
#                         prepared_content.append(part)
#                     else:
#                         # Fallback for unexpected content part types
#                         prepared_content.append({"type": "text", "text": str(part)})
#                 prepared_message["content"] = prepared_content
#             else:
#                 # Fallback if content is neither string nor list
#                 prepared_message["content"] = str(message.content)

#             prepared_messages.append(prepared_message)

#         elif isinstance(message, dict):
#             # Handle dictionary input - assume it's already in the correct format
#             # Perform minimal validation/normalization if needed, but for OpenAI
#             # compatibility, we mostly trust the input dictionary structure.
#             if "role" not in message or "content" not in message:
#                  # Basic check, could be more robust
#                 raise ValueError("Input dictionary message must contain 'role' and 'content' keys.")
#             # Ensure content parts within a dict message are also dicts (as expected by API)
#             if isinstance(message["content"], list):
#                 message["content"] = [
#                     part if isinstance(part, dict) else asdict(part) if is_dataclass(part) else {"type": "text", "text": str(part)}
#                     for part in message["content"]
#                 ]
#             prepared_messages.append(message)
#         else:
#             raise TypeError(f"Invalid message type: {type(message)}. Expected Message object or dict.")

#     return prepared_messages


def prepare_stop_sequences(stop: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
    """
    Prepare stop sequences for a completion request.
    
    Args:
        stop: Stop sequence(s)
        
    Returns:
        Optional[List[str]]: Prepared stop sequences
    """
    if stop is None:
        return None
    
    if isinstance(stop, str):
        return [stop]
    
    return stop


def calculate_usage(prompt_tokens: int, completion_tokens: int) -> Dict[str, int]:
    """
    Calculate token usage.
    
    Args:
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        
    Returns:
        Dict[str, int]: Token usage
    """
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    }
