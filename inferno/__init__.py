"""
Inferno - A llama-cpp-python based LLM serving tool with Ollama-compatible API
"""

# Import version
from .version import __version__

# Import main components for easier access
try:
    from .core import (
        LLMInterface,
        ModelManager,
        estimate_gguf_ram_requirements,
        get_ram_requirement_string,
        get_hardware_suggestion,
        suggest_hardware,
        get_system_ram,
        extract_max_context_from_gguf,
        estimate_from_huggingface_repo,
        detect_quantization_from_filename
    )
except ImportError:
    # This allows the package to be installed without llama-cpp-python
    # which is required to be installed separately with hardware acceleration
    pass
from .api import start_server

# Define what's available when using `from inferno import *`
__all__ = [
    "LLMInterface",
    "ModelManager",
    "start_server",
    "estimate_gguf_ram_requirements",
    "get_ram_requirement_string",
    "get_hardware_suggestion",
    "suggest_hardware",
    "get_system_ram",
    "extract_max_context_from_gguf",
    "estimate_from_huggingface_repo",
    "detect_quantization_from_filename"
]

