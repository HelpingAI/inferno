"""
Utility functions for Inferno
"""

import base64
import logging

logger = logging.getLogger(__name__)

def parse_duration(duration_str: str) -> float:
    """
    Parse a duration string into seconds.
    
    Args:
        duration_str (str): Duration string (e.g., '5m', '1h', '30s', '500ms', '0').
    Returns:
        float: Duration in seconds.
    """
    if not duration_str:
        return 300.0  # Default 5 minutes
    if duration_str.endswith("ms"):
        return int(duration_str[:-2]) / 1000.0
    elif duration_str.endswith("s"):
        return int(duration_str[:-1])
    elif duration_str.endswith("m"):
        return int(duration_str[:-1]) * 60
    elif duration_str.endswith("h"):
        return int(duration_str[:-1]) * 3600
    elif duration_str == "0":
        return 0.0
    else:
        try:
            return float(duration_str)
        except ValueError:
            return 300.0  # Default 5 minutes

def format_duration(seconds: float) -> str:
    """
    Format seconds into a human-readable duration string.
    Args:
        seconds (float): Duration in seconds.
    Returns:
        str: Human-readable duration string.
    """
    if seconds < 1:
        return f"{int(seconds * 1000)}ms"
    elif seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds / 60)}m"
    else:
        return f"{int(seconds / 3600)}h"

def decode_image(image_str: str) -> bytes:
    """
    Decode a base64 image string to bytes.
    Args:
        image_str (str): Base64-encoded image string (optionally with data URI prefix).
    Returns:
        bytes: Decoded image bytes.
    """
    if image_str.startswith("data:"):
        image_str = image_str.split(",", 1)[1]
    return base64.b64decode(image_str)

def encode_image(image_bytes: bytes, mime_type: str = "image/png") -> str:
    """
    Encode image bytes to a base64 data URI.
    Args:
        image_bytes (bytes): Image data.
        mime_type (str): MIME type for the image.
    Returns:
        str: Base64-encoded data URI string.
    """
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"

def get_file_size_str(size_bytes: int) -> str:
    """
    Convert file size in bytes to a human-readable string.
    Args:
        size_bytes (int): File size in bytes.
    Returns:
        str: Human-readable file size string.
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
