"""
Core functionality for Inferno
"""

from .llm import LLMInterface
from .model_manager import ModelManager
from .ram_estimator import (
    estimate_gguf_ram_requirements,
    get_ram_requirement_string,
    get_hardware_suggestion,
    suggest_hardware,
    get_system_ram,
    extract_max_context_from_gguf,
    estimate_from_huggingface_repo,
    detect_quantization_from_filename
)

__all__ = [
    "LLMInterface",
    "ModelManager",
    "estimate_gguf_ram_requirements",
    "get_ram_requirement_string",
    "get_hardware_suggestion",
    "suggest_hardware",
    "get_system_ram",
    "extract_max_context_from_gguf",
    "estimate_from_huggingface_repo",
    "detect_quantization_from_filename"
]
