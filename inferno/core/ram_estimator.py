"""
RAM requirement estimation for GGUF models.
"""

import json
import os
import requests
import math
import re
from .gguf_reader import simple_gguf_info
from pathlib import Path
from typing import Any, Dict, Optional, List

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Default context lengths (can be overridden by function arguments)
DEFAULT_CONTEXT_LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]

def estimate_gguf_ram_requirements(model_path: str, verbose: bool = False, context_lengths: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Estimate RAM requirements to run a GGUF model.

    Args:
        model_path: Path to the GGUF model file or URL
        verbose: Whether to print detailed information (deprecated, kept for compatibility)

    Returns:
        Dictionary with RAM requirements for different quantization levels
    """
    # Get model size in bytes
    file_size_bytes = get_model_size(model_path)
    if file_size_bytes is None:
        return {}

    file_size_gb = file_size_bytes / (1024**3)  # Convert to GB

    # Estimate RAM requirements based on model size and quantization
    # These multipliers are based on empirical observations
    ram_requirements = {}

    # Comprehensive list of all GGUF quantization levels and their typical RAM multipliers
    # From lowest precision (Q2) to highest (F16/FP16)
    quantization_multipliers = {
        # 2-bit quantization
        "Q2_K": 1.15,       # Q2_K (2-bit quantization with K-quants)
        "Q2_K_S": 1.18,     # Q2_K_S (2-bit quantization with K-quants, small)

        # 3-bit quantization
        "Q3_K_S": 1.25,     # Q3_K_S (3-bit quantization with K-quants, small)
        "Q3_K_M": 1.28,     # Q3_K_M (3-bit quantization with K-quants, medium)
        "Q3_K_L": 1.30,     # Q3_K_L (3-bit quantization with K-quants, large)

        # 4-bit quantization
        "Q4_0": 1.33,       # Q4_0 (4-bit quantization, version 0)
        "Q4_1": 1.35,       # Q4_1 (4-bit quantization, version 1)
        "Q4_K_S": 1.38,     # Q4_K_S (4-bit quantization with K-quants, small)
        "Q4_K_M": 1.40,     # Q4_K_M (4-bit quantization with K-quants, medium)
        "Q4_K_L": 1.43,     # Q4_K_L (4-bit quantization with K-quants, large)

        # 5-bit quantization
        "Q5_0": 1.50,       # Q5_0 (5-bit quantization, version 0)
        "Q5_1": 1.55,       # Q5_1 (5-bit quantization, version 1)
        "Q5_K_S": 1.60,     # Q5_K_S (5-bit quantization with K-quants, small)
        "Q5_K_M": 1.65,     # Q5_K_M (5-bit quantization with K-quants, medium)
        "Q5_K_L": 1.70,     # Q5_K_L (5-bit quantization with K-quants, large)

        # 6-bit quantization
        "Q6_K": 1.80,       # Q6_K (6-bit quantization with K-quants)

        # 8-bit quantization
        "Q8_0": 2.00,       # Q8_0 (8-bit quantization, version 0)
        "Q8_K": 2.10,       # Q8_K (8-bit quantization with K-quants)

        # Floating point formats
        "F16": 2.80,        # F16 (16-bit float, same as FP16)
        "FP16": 2.80,       # FP16 (16-bit float)
    }

    # Calculate RAM requirements for each quantization level
    for quant_name, multiplier in quantization_multipliers.items():
        ram_requirements[quant_name] = file_size_gb * multiplier

    # Use simple_gguf_info to get context length from GGUF metadata
    context_length = None
    try:
        info = simple_gguf_info(model_path)
        # Guard against simple_gguf_info returning None or unexpected types
        if not isinstance(info, dict):
            metadata = {}
        else:
            metadata = info.get("metadata") or {}

        # Ensure metadata is a dict before attempting to iterate it
        if isinstance(metadata, dict):
            # Use regex to find any key ending with .context_length
            context_length_keys = [k for k in metadata.keys() if re.search(r'\.context_length$', k)]
            if context_length_keys:
                # Use the first valid integer value found
                for key in sorted(context_length_keys):
                    try:
                        context_length = int(metadata[key])
                        break
                    except (ValueError, TypeError):
                        continue
    except Exception:
        context_length = None

    # If context_length is found, use it as the only context length for RAM estimation
    if context_length is not None:
        context_lengths = [context_length]
    elif context_lengths is None:
        context_lengths = DEFAULT_CONTEXT_LENGTHS

    # For context generation, add additional overhead based on context length
    context_ram = {}
    model_params_billions = estimate_params_from_file_size(file_size_bytes, quant="Q4_K_M")
    for ctx_len in context_lengths:
        # KV cache formula: 2 (K&V) * num_layers * hidden_dim * context_length * bytes_per_token
        # We estimate based on model parameters
        estimated_layers = min(max(int(model_params_billions * 0.8), 24), 80)  # Estimate number of layers
        estimated_hidden_dim = min(max(int(model_params_billions * 30), 1024), 8192)  # Estimate hidden dimension
        bytes_per_token = 2  # 2 bytes for half-precision (FP16) KV cache

        kv_cache_size_gb = (2 * estimated_layers * estimated_hidden_dim * ctx_len * bytes_per_token) / (1024**3)
        context_ram[f"Context {ctx_len}"] = kv_cache_size_gb

    ram_requirements["context_overhead"] = context_ram
    ram_requirements["model_params_billions"] = model_params_billions

    return ram_requirements

def estimate_params_from_file_size(file_size_bytes: int, quant: str = "Q4_K_M") -> float:
    """
    Estimate the number of parameters (in billions) from model file size.

    Args:
        file_size_bytes: Size of the model file in bytes
        quant: Quantization type

    Returns:
        Estimated number of parameters in billions
    """
    # Bits per parameter for different quantization types
    bits_per_param = {
        "Q2_K": 2.5,      # ~2-2.5 bits per param
        "Q3_K_M": 3.5,    # ~3-3.5 bits per param
        "Q4_K_M": 4.5,    # ~4-4.5 bits per param
        "Q5_K_M": 5.5,    # ~5-5.5 bits per param
        "Q6_K": 6.5,      # ~6-6.5 bits per param
        "Q8_0": 8.5,      # ~8-8.5 bits per param
        "F16": 16.0,      # 16 bits per param
    }

    # Default to Q4_K_M if the specified quant is not in the dictionary
    bits = bits_per_param.get(quant, 4.5)

    # Convert bits to bytes for calculation
    bytes_per_param = bits / 8

    # Calculate number of parameters
    params = file_size_bytes / bytes_per_param

    # Convert to billions
    params_billions = params / 1e9

    return params_billions

def get_model_size(model_path: str) -> Optional[int]:
    """
    Get the size of a model file in bytes.
    Works for both local files and remote URLs.

    Args:
        model_path: Path to the model file or URL

    Returns:
        Size in bytes or None if size can't be determined
    """
    if os.path.exists(model_path):
        # Local file
        return os.path.getsize(model_path)

    elif model_path.startswith(('http://', 'https://')):
        # Remote file - try to get size from HTTP headers
        try:
            response = requests.head(model_path, allow_redirects=True)
            if response.status_code == 200 and 'content-length' in response.headers:
                return int(response.headers['content-length'])
            return None
        except Exception:
            return None
    else:
        return None

def suggest_hardware(ram_required: float) -> str:
    """
    Suggest hardware based on RAM requirements.

    Args:
        ram_required: RAM required in GB

    Returns:
        Hardware recommendation
    """
    if ram_required <= 4:
        return "Entry-level desktop/laptop with 8GB RAM should work"
    elif ram_required <= 8:
        return "Standard desktop/laptop with 16GB RAM recommended"
    elif ram_required <= 16:
        return "High-end desktop/laptop with 32GB RAM recommended"
    elif ram_required <= 32:
        return "Workstation with 64GB RAM recommended"
    elif ram_required <= 64:
        return "High-end workstation with 128GB RAM recommended"
    else:
        return f"Server-grade hardware with at least {math.ceil(ram_required*1.5)}GB RAM recommended"

def detect_gpu_vram() -> Any:
    """
    Detect available GPU VRAM if possible.
    Requires optional dependencies (nvidia-ml-py or pynvml).

    Returns:
        Dict mapping GPU index to available VRAM in GB, or empty dict if detection fails
    """
    try:
        import pynvml # type: ignore[import]
        pynvml.nvmlInit()

        vram_info = {}
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_total_gb = info.total / (1024**3)
            vram_free_gb = info.free / (1024**3)
            vram_info[i] = {
                "total": vram_total_gb,
                "free": vram_free_gb,
                "name": pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            }

        pynvml.nvmlShutdown()
        return vram_info

    except ImportError:
        print("GPU VRAM detection requires pynvml. Install with: pip install nvidia-ml-py")
        return {}
    except Exception as e:
        print(f"Error detecting GPUs: {e}")
        return {}


def detect_quantization_from_filename(filename: str) -> Optional[str]:
    """
    Try to detect the quantization type from the filename.

    Args:
        filename: Name of the model file

    Returns:
        Detected quantization type or None if not detected
    """
    filename = filename.lower()

    # Common quantization naming patterns
    patterns = [
        ('q2k', 'Q2_K'),
        ('q2_k', 'Q2_K'),
        ('q3k', 'Q3_K_M'),
        ('q3_k', 'Q3_K_M'),
        ('q3_k_m', 'Q3_K_M'),
        ('q3_k_s', 'Q3_K_S'),
        ('q3_k_l', 'Q3_K_L'),
        ('q4_0', 'Q4_0'),
        ('q4_1', 'Q4_1'),
        ('q4k', 'Q4_K_M'),
        ('q4_k', 'Q4_K_M'),
        ('q4_k_m', 'Q4_K_M'),
        ('q4_k_s', 'Q4_K_S'),
        ('q4_k_l', 'Q4_K_L'),
        ('q5_0', 'Q5_0'),
        ('q5_1', 'Q5_1'),
        ('q5k', 'Q5_K_M'),
        ('q5_k', 'Q5_K_M'),
        ('q5_k_m', 'Q5_K_M'),
        ('q5_k_s', 'Q5_K_S'),
        ('q5_k_l', 'Q5_K_L'),
        ('q6k', 'Q6_K'),
        ('q6_k', 'Q6_K'),
        ('q8_0', 'Q8_0'),
        ('q8k', 'Q8_K'),
        ('q8_k', 'Q8_K'),
        ('f16', 'F16'),
        ('fp16', 'FP16')
    ]

    for pattern, quant_type in patterns:
        if pattern in filename:
            return quant_type

    return None

def estimate_from_huggingface_repo(repo_id: str, branch: str = "main", context_lengths: Optional[List[int]] = None) -> Dict[str, float]:
    """
    Estimate RAM requirements for a model from a Hugging Face repository.

    Args:
        repo_id: Hugging Face repository ID (e.g., 'TheBloke/Llama-2-7B-GGUF')
        branch: Repository branch

    Returns:
        Dictionary with RAM requirements
    """
    api_url = f"https://huggingface.co/api/models/{repo_id}/tree/{branch}"
    try:
        response = requests.get(api_url)
        if response.status_code != 200:
            return {}

        files = response.json()
        gguf_files = [f for f in files if f.get('path', '').endswith('.gguf')]

        if not gguf_files:
            return {}

        # Group files by quantization type
        quant_groups = {}
        all_files_info = {}

        for file in gguf_files:
            file_path = file.get('path', '')
            filename = os.path.basename(file_path)
            size_bytes = file.get('size', 0)
            size_gb = size_bytes / (1024**3)

            # Store info for all files
            all_files_info[filename] = {
                'path': file_path,
                'size_bytes': size_bytes,
                'size_gb': size_gb,
                'max_context': None  # Will be estimated later
            }

            quant_type = detect_quantization_from_filename(filename)
            if quant_type:
                if quant_type not in quant_groups:
                    quant_groups[quant_type] = []
                quant_groups[quant_type].append((filename, size_gb, size_bytes))

        # Find a representative file for RAM estimation
        # Prefer Q4_K_M as it's common, or pick the largest file
        if not quant_groups:
            # If quantization detection failed, just use the largest file
            largest_file = max(gguf_files, key=lambda x: x.get('size', 0))
            size_bytes = largest_file.get('size', 0)
            file_path = largest_file.get('path', '')
            return estimate_gguf_ram_requirements(file_path, verbose=False)

        # Choose a representative model
        chosen_quant = None
        chosen_file = None

        # Preference order
        preferred_quants = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]

        for quant in preferred_quants:
            if quant in quant_groups:
                chosen_quant = quant
                # Choose the latest version if multiple files with same quant
                chosen_file = max(quant_groups[quant], key=lambda x: x[1])  # Sort by size
                break

        if not chosen_quant:
            # Just choose the first available quantization
            chosen_quant = list(quant_groups.keys())[0]
            chosen_file = quant_groups[chosen_quant][0]

        # Ensure we actually have a chosen file
        if chosen_file is None:
            return {}

        filename, size_gb, size_bytes = chosen_file

        # Create RAM estimation using the file size
        ram_requirements = {}

        # Use the same estimation logic as the main function
        # Comprehensive list of all GGUF quantization levels
        quantization_multipliers = {
            # 2-bit quantization
            "Q2_K": 1.15,
            "Q2_K_S": 1.18,

            # 3-bit quantization
            "Q3_K_S": 1.25,
            "Q3_K_M": 1.28,
            "Q3_K_L": 1.30,

            # 4-bit quantization
            "Q4_0": 1.33,
            "Q4_1": 1.35,
            "Q4_K_S": 1.38,
            "Q4_K_M": 1.40,
            "Q4_K_L": 1.43,

            # 5-bit quantization
            "Q5_0": 1.50,
            "Q5_1": 1.55,
            "Q5_K_S": 1.60,
            "Q5_K_M": 1.65,
            "Q5_K_L": 1.70,

            # 6-bit quantization
            "Q6_K": 1.80,

            # 8-bit quantization
            "Q8_0": 2.00,
            "Q8_K": 2.10,

            # Floating point formats
            "F16": 2.80,
            "FP16": 2.80,
        }

        # Estimate base size from the chosen quantization
        base_size_gb = size_bytes / (1024**3)
        model_params_billions = estimate_params_from_file_size(size_bytes, chosen_quant)

        # Calculate RAM estimates for all quantizations by scaling from the chosen one
        chosen_multiplier = quantization_multipliers[chosen_quant]
        base_model_size = base_size_gb / chosen_multiplier  # Theoretical unquantized size

        for quant_name, multiplier in quantization_multipliers.items():
            ram_requirements[quant_name] = base_model_size * multiplier

        # For context generation, add additional overhead
        if context_lengths is None:
            context_lengths = DEFAULT_CONTEXT_LENGTHS
        context_ram = {}

        # KV cache formula
        estimated_layers = min(max(int(model_params_billions * 0.8), 24), 80)
        estimated_hidden_dim = min(max(int(model_params_billions * 30), 1024), 8192)
        bytes_per_token = 2  # 2 bytes for half-precision KV cache

        for ctx_len in context_lengths:
            kv_cache_size_gb = (2 * estimated_layers * estimated_hidden_dim * ctx_len * bytes_per_token) / (1024**3)
            context_ram[f"Context {ctx_len}"] = kv_cache_size_gb

        ram_requirements["context_overhead"] = context_ram
        ram_requirements["model_params_billions"] = model_params_billions

        # Estimate max context length for each file based on model size
        for filename, file_info in all_files_info.items():
            # Larger models typically support longer context
            # This is just an estimate since we can't directly read the GGUF file
            params_billions = estimate_params_from_file_size(file_info['size_bytes'],
                                                           detect_quantization_from_filename(filename) or "Q4_K_M")

            # Estimate max context based on model size
            # These are rough estimates - actual values may vary
            max_ctx = None  # Remove hardcoded fallback
            # Optionally, could use context_lengths[-1] or None
            all_files_info[filename]['max_context'] = max_ctx
            all_files_info[filename]['params_billions'] = params_billions

        ram_requirements["all_files"] = all_files_info

        return ram_requirements

    except Exception:
        return {}

def print_gpu_compatibility(ram_requirements: Dict[str, Any], vram_info: Dict, context_lengths: Optional[List[int]] = None):
    """
    Print GPU compatibility information based on RAM requirements.

    Args:
        ram_requirements: Dictionary with RAM requirements (values may include nested dicts)
        vram_info: Dictionary with GPU VRAM information
    """
    if not vram_info:
        print("\nNo GPU information available")
        return

    print("\n=== GPU Compatibility Analysis ===")

    # Context lengths to analyze
    if context_lengths is None:
        context_lengths = DEFAULT_CONTEXT_LENGTHS[:4]  # Default to first 4 for display
    else:
        # Coerce values to ints where possible
        try:
            context_lengths = [int(x) for x in context_lengths]
        except Exception:
            context_lengths = DEFAULT_CONTEXT_LENGTHS[:4]

    # Quantization levels to check (arranged from most efficient to highest quality)
    quant_levels = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]

    # Get context RAM overhead and validate it's a dict
    context_ram_obj = ram_requirements.get("context_overhead", {})
    context_ram = context_ram_obj if isinstance(context_ram_obj, dict) else {}

    # For each GPU
    for gpu_idx, gpu_data in vram_info.items():
        gpu_name = gpu_data.get("name", f"GPU {gpu_idx}")
        vram_total = gpu_data.get("total", 0)
        vram_free = gpu_data.get("free", 0)

        print(f"\n{gpu_name}: {vram_total:.2f} GB total VRAM, {vram_free:.2f} GB free")

        # Check compatibility for each combination
        print("\nCompatibility matrix (✓: fits, ✗: doesn't fit):")

        # Print header row with context lengths
        header = "Quantization | "
        for ctx_len in context_lengths:
            header += f"{int(ctx_len):6d} | "
        print(header)
        print("-" * len(header))

        # Print compatibility for each quantization level
        for quant in quant_levels:
            if quant not in ram_requirements:
                continue

            # Ensure base_ram is numeric
            try:
                base_ram = float(ram_requirements[quant])
            except Exception:
                # Skip non-numeric entries
                continue

            row = f"{quant:11s} | "

            for ctx_len in context_lengths:
                ctx_key = f"Context {int(ctx_len)}"
                if ctx_key in context_ram:
                    ctx_overhead = context_ram[ctx_key]
                    try:
                        total_ram = base_ram + float(ctx_overhead)
                    except Exception:
                        # If ctx_overhead is not numeric, treat as zero
                        total_ram = base_ram

                    # Check if it fits in VRAM
                    fits = total_ram <= vram_free
                    row += f"{'✓':6s} | " if fits else f"{'✗':6s} | "

            print(row)

def get_ram_requirement_string(ram_gb: float, colorize: bool = False) -> str:
    """
    Format RAM requirement as a string.

    Args:
        ram_gb: RAM requirement in GB
        colorize: Whether to add color formatting based on RAM size

    Returns:
        Formatted string with optional color formatting
    """
    if ram_gb < 1:
        ram_str = f"{ram_gb * 1024:.0f} MB"
    else:
        ram_str = f"{ram_gb:.1f} GB"

    if colorize:
        if ram_gb < 4:
            return f"[green]{ram_str}[/green]"
        elif ram_gb < 8:
            return f"[cyan]{ram_str}[/cyan]"
        elif ram_gb < 16:
            return f"[blue]{ram_str}[/blue]"
        elif ram_gb < 32:
            return f"[yellow]{ram_str}[/yellow]"
        elif ram_gb < 64:
            return f"[orange]{ram_str}[/orange]"
        else:
            return f"[red]{ram_str}[/red]"
    else:
        return ram_str

def get_hardware_suggestion(ram_gb: float) -> str:
    """
    Get hardware suggestion based on RAM requirement.

    Args:
        ram_gb: RAM requirement in GB

    Returns:
        Hardware suggestion string
    """
    return suggest_hardware(ram_gb)

def get_system_ram() -> float:
    """
    Get total system RAM in GB.

    Returns:
        Total RAM in GB or 0 if detection fails
    """
    if PSUTIL_AVAILABLE:
        try:
            return psutil.virtual_memory().total / (1024**3)
        except Exception:
            pass
    return 0

def extract_max_context_from_gguf(model_path: str, debug: bool = False) -> Optional[int]:
    """
    Extract maximum context length from a GGUF file using simple_gguf_info and regex.

    Args:
        model_path: Path to the GGUF file
        debug: Whether to print debug information

    Returns:
        Maximum context length or None if not found
    """
    try:
        # Get model filename for debugging
        filename = os.path.basename(model_path)
        if debug:
            print(f"Analyzing context length for: {filename}")

        # Read GGUF info using the robust simple_gguf_info
        info = simple_gguf_info(model_path)
        if debug:
            print(f"GGUF info keys: {list(info.keys())}")
            if "metadata" in info:
                print(f"Metadata keys found: {len(info['metadata'])}")

        # Check if metadata exists
        if "metadata" in info:
            metadata = info["metadata"]
            context_length_keys = []

            # Find all keys ending with .context_length using regex
            for key in metadata.keys():
                if re.search(r'\.context_length$', key):
                    context_length_keys.append(key)

            if debug:
                print(f"Found keys ending with '.context_length': {context_length_keys}")

            # Sort keys alphabetically for consistent selection
            if isinstance(context_length_keys, list):
                context_length_keys.sort()

                # Try to extract and validate the context length from the found keys
                for key in context_length_keys:
                    try:
                        value = int(metadata[key])
                        if debug:
                            print(f"Found valid context length in metadata key '{key}': {value}")
                        return value
                    except (ValueError, TypeError, OverflowError):
                        if debug:
                            print(f"Value for key '{key}' ('{metadata[key]}') is not a valid integer.")
                        continue # Try the next key

        # If no valid context length found in metadata
        if debug:
            if "error" in info:
                print(f"simple_gguf_info reported an error: {info['error']}")
            elif "context_length_error" in info:
                 print(f"simple_gguf_info reported context length error: {info['context_length_error']}")
            else:
                print("Could not find a valid '.context_length' key in metadata.")

        return None
    except Exception as e:
        print(f"Error extracting context length: {e}")
        if debug:
            import traceback
            print(traceback.format_exc())
        return None

def estimate_ram_for_all_models(models_dir: str, context_lengths: Optional[List[int]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Estimate RAM requirements for all models in a directory.

    Args:
        models_dir: Directory containing models

    Returns:
        Dictionary mapping model names to RAM requirements
    """
    results = {}
    models_path = Path(os.path.expanduser(models_dir))

    if not models_path.exists():
        return results

    for model_dir in models_path.iterdir():
        if not model_dir.is_dir():
            continue

        info_file = model_dir / "info.json"
        if not info_file.exists():
            continue

        try:
            with open(info_file, "r") as f:
                info = json.load(f)

            model_path = info.get("path")
            if not model_path or not os.path.exists(model_path):
                continue

            # Estimate RAM requirements
            ram_req = estimate_gguf_ram_requirements(model_path, context_lengths=context_lengths)
            if not ram_req:
                continue

            # Extract context length if possible
            ctx_len = extract_max_context_from_gguf(model_path)

            # Add to results
            results[model_dir.name] = {
                "info": info,
                "ram_requirements": ram_req,
                "max_context": ctx_len
            }
        except Exception:
            continue

    return results

if __name__ == "__main__":
    path = r"""C:\Users\koula\.inferno\models\Qwen2-0.5B-Instruct-GGUF\qwen2-0_5b-instruct-q2_k.gguf"""
    ram_requirements = extract_max_context_from_gguf(path)
    print(ram_requirements)
