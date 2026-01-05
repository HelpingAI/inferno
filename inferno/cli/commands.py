"""
Command-line interface for Inferno
"""

import os
import typer
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.box import SIMPLE
from rich.panel import Panel
from rich.text import Text
from typing import Optional, List
from pathlib import Path
import datetime
from ..core.model_manager import ModelManager
from ..core.quantizer import QuantizationMethod
from ..core.llm import LLMInterface
from ..api.server import start_server
from ..core.ram_estimator import (
    estimate_gguf_ram_requirements,
    get_ram_requirement_string,
    get_hardware_suggestion,
    get_system_ram,
    detect_quantization_from_filename
)
from ..core.gguf_reader import simple_gguf_info, debug_gguf_context_length

app: typer.Typer = typer.Typer(help="Inferno - Run Llama 3.3, DeepSeek-R1, Phi-4, Gemma 3, Mistral Small 3.1, and other state-of-the-art language models locally with scorching-fast performance. Inferno provides an intuitive CLI and an OpenAI/Ollama-compatible API, putting the inferno of AI innovation directly in your hands.")
console: Console = Console()

model_manager: ModelManager = ModelManager()

# Fallback RAM requirements for different model sizes (used when estimation fails)
FALLBACK_RAM_REQUIREMENTS = {
    "1B": "2 GB",
    "3B": "4 GB",
    "7B": "8 GB",
    "13B": "16 GB",
    "33B": "32 GB",
    "70B": "64 GB",
}

@app.command("serve")
def run_model(
    model_string: str = typer.Argument(..., help="Model to run (format: 'name', 'repo_id', 'repo_id:filename', 'hf:repo_id', or 'hf:repo_id:quantization')"),
    host: Optional[str] = typer.Option(None, help="Host to bind the server to"),
    port: Optional[int] = typer.Option(None, help="Port to bind the server to"),
    n_gpu_layers: Optional[int] = typer.Option(None, help="Number of layers to offload to GPU (-1 for all)"),
    n_ctx: Optional[int] = typer.Option(None, help="Context window size"),
    n_threads: Optional[int] = typer.Option(None, help="Number of threads to use for inference"),
    use_mlock: bool = typer.Option(False, help="Lock model in memory"),
) -> None:
    """
    Start a model server (downloads if needed).
    
    Supports multiple model string formats:
    - 'name' - Name of a locally downloaded model
    - 'repo_id' - Standard HuggingFace repository ID
    - 'repo_id:filename' - Repository ID with specific filename
    - 'hf:repo_id' - HuggingFace prefix with repository ID
    - 'hf:repo_id:quantization' - HuggingFace prefix with repository ID and quantization type (e.g., Q2_K, Q4_K_M)
    
    Example: hf:mradermacher/DAN-Qwen3-1.7B-GGUF:Q2_K
    """
    # First, ensure the model string is treated as a str for type checkers
    model_string = str(model_string)

    # First check if this is a filename that already exists
    model_path = model_manager.get_model_path(model_string)
    if model_path:
        # This is a filename that exists, find the model name
        for model_info in model_manager.list_models():
            if isinstance(model_info, dict) and (model_info.get("filename") == model_string or model_info.get("path") == model_path):
                model_name = model_info.get("name")
                break
        else:
            # Fallback to using the string as model name
            model_name = model_string
    else:
        # Parse the model string to see if it's a repo_id:filename format, hf:repo_id or hf:repo_id:quantization
        original_model_string = model_string
        
        # Handle hf: prefix for model name
        if model_string.startswith("hf:"):
            model_string = model_string[3:]  # Remove hf: prefix for model name derivation
        
        repo_id, _ = model_manager.parse_model_string(original_model_string)
        model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id

        # Check if model exists, if not try to download it
        if not model_manager.get_model_path(model_name):
            console.print(f"[yellow]Model {model_name} not found locally. Attempting to download...[/yellow]")
            try:
                # Download the model with the original string (including hf: prefix if it exists)
                model_name, _ = model_manager.download_model(original_model_string)
                console.print(f"[bold green]Model {model_name} downloaded successfully[/bold green]")
            except Exception as e:
                console.print(f"[bold red]Error downloading model: {str(e)}[/bold red]")
                return

    # Check RAM requirements
    # Ensure model_name is a str for type checking
    model_name = str(model_name)
    model_path = model_manager.get_model_path(model_name)
    ram_requirement = "Unknown"
    ram_reqs = None
    quant_type = None

    if model_path and os.path.exists(model_path):
        try:
            # Try to detect quantization from filename
            path_str = str(model_path)
            filename = os.path.basename(path_str)
            quant_type = detect_quantization_from_filename(filename)

            # Try to estimate RAM requirements from the model file (pass string path)
            ram_reqs = estimate_gguf_ram_requirements(path_str, verbose=False)
            if ram_reqs:
                # Use detected quantization or fall back to Q4_K_M
                quant_to_use = quant_type if quant_type and quant_type in ram_reqs else "Q4_K_M"
                if quant_to_use in ram_reqs:
                    ram_gb = ram_reqs[quant_to_use]
                    ram_requirement = get_ram_requirement_string(ram_gb, colorize=True)
                    hardware_suggestion = get_hardware_suggestion(ram_gb)

                    console.print(f"[yellow]Model quantization: [bold]{quant_type or 'Unknown'}[/bold][/yellow]")
                    # Clarify RAM estimation context
                    console.print(f"[yellow]Estimated RAM requirement ({quant_to_use}, base): {ram_requirement}[/yellow]")
                    console.print(f"[yellow]Hardware suggestion: {hardware_suggestion}[/yellow]")
        except Exception as e:
            console.print(f"[dim]Error estimating RAM requirements: {str(e)}[/dim]")

    # Fall back to size-based estimation if needed
    if ram_requirement == "Unknown":
        for size, ram in FALLBACK_RAM_REQUIREMENTS.items():
            if isinstance(model_name, str) and size in model_name:
                ram_requirement = ram
                console.print(f"[yellow]This model requires approximately {ram_requirement} of RAM (estimated from model name)[/yellow]")
                # console.print(f"[yellow]This model requires approximately {ram_requirement} of RAM (estimated from model name)[/yellow]")
                break

    # Check if we have enough RAM
    if ram_reqs:
        system_ram = get_system_ram()
        if system_ram > 0:
            # Use detected quantization or fall back to Q4_K_M
            quant_to_use = quant_type if quant_type and quant_type in ram_reqs else "Q4_K_M"
            if quant_to_use in ram_reqs:
                # Get RAM requirement with default context
                base_ram = ram_reqs[quant_to_use]
                context_overhead = ram_reqs.get("context_overhead", {})
                ctx_ram = context_overhead.get("Context 4096", 0) if isinstance(context_overhead, dict) else 0
                total_ram = base_ram + ctx_ram

                if total_ram > system_ram:
                    from rich.panel import Panel
                    from rich.text import Text

                    warning_text = Text()
                    # Clarify context used for warning
                    warning_text.append(f"WARNING: This model requires ~{total_ram:.2f} GB RAM (with 4096 context), but only {system_ram:.2f} GB is available!\n", style="bold red")
                    warning_text.append("The model may not load or could cause system instability.\n", style="bold red")
                    warning_text.append("\nConsider using a lower quantization level like Q3_K or Q2_K if available.", style="yellow")

                    console.print(Panel(
                        warning_text,
                        title="⚠️ INSUFFICIENT RAM ⚠️",
                        border_style="red"
                    ))

    # Try to detect max context length from the model file
    detected_max_context = None # Rename variable to avoid confusion with loop variable
    if model_path and os.path.exists(model_path):
        try:
            # Always use extract_max_context_from_gguf for max context detection
            from ..core.ram_estimator import extract_max_context_from_gguf
            detected_max_context = extract_max_context_from_gguf(model_path)
            if detected_max_context:
                console.print(f"[cyan]Detected maximum context length: {detected_max_context:,}, but we will use 4096[/cyan]")
            else:
                # Handle case where function returns None without raising an error
                console.print("[yellow]Could not detect maximum context length, using default (4096)[/yellow]")
                detected_max_context = 4096
        except Exception as e:
            console.print(f"[yellow]Error detecting context length: {str(e)}. Using default (4096)[/yellow]")
            detected_max_context = 4096

    # Load the model with provided options
    try:
        # Ensure we pass a concrete str to the LLM interface
        llm = LLMInterface(str(model_name))
        # Use provided context length or default to 4096
        n_ctx_to_load = n_ctx or 4096
        llm.load_model(
            verbose=False,
            n_ctx=n_ctx_to_load,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            use_mlock=use_mlock
        )
    except Exception as e:
        console.print(f"[bold red]Error loading model: {str(e)}[/bold red]")
        return

    # Start the server
    console.print(f"[bold blue]Starting Inferno server with model {model_name}...[/bold blue]")
    # Create options dictionary for model configuration
    model_options = {
        "n_gpu_layers": n_gpu_layers,
        "n_ctx": n_ctx,
        "n_threads": n_threads,
        "use_mlock": use_mlock,
    }

    # Start server with model options. Provide sensible defaults if host/port are None to satisfy type checkers.
    start_server(host=host or "127.0.0.1", port=port or 8080, model_options=model_options)

@app.command("pull")
def pull_model(
    model_string: str = typer.Argument(..., help="Model to download (format: 'repo_id', 'repo_id:filename', 'hf:repo_id', or 'hf:repo_id:quantization')"),
) -> None:
    """
    Download a model from Hugging Face without running it.
    
    Supports multiple formats:
    - 'repo_id' - Standard HuggingFace repository ID
    - 'repo_id:filename' - Repository ID with specific filename
    - 'hf:repo_id' - HuggingFace prefix with repository ID
    - 'hf:repo_id:quantization' - HuggingFace prefix with repository ID and quantization type (e.g., Q2_K, Q4_K_M)
    """
    import traceback # Import traceback for detailed error reporting

    try:
        console.print(f"Attempting to download model: {model_string}") # Add logging
        result = model_manager.download_model(model_string)

        # Add check for unexpected return types (though exceptions are better)
        if not isinstance(result, tuple) or len(result) != 2:
            console.print(f"[bold red]Error: Download function returned unexpected result: {result}[/bold red]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return

        model_name, model_path = result
        console.print(f"[bold green]Model {model_name} downloaded successfully to {model_path}[/bold green]")

        # Always use extract_max_context_from_gguf for max context detection
        max_context = None
        if model_path and os.path.exists(model_path):
            try:
                from ..core.ram_estimator import extract_max_context_from_gguf
                path_str = str(model_path)
                max_context = extract_max_context_from_gguf(path_str)
                if max_context:
                    console.print(f"[cyan]Detected maximum context length: {max_context:,}, but we will use 4096 by default[/cyan]")
                else:
                    # Handle case where function returns None without raising an error
                    console.print("[yellow]Could not detect maximum context length, using default (4096)[/yellow]")
                    max_context = 4096
            except Exception as e:
                console.print(f"[yellow]Error detecting context length: {str(e)}. Using default (4096)[/yellow]")
                max_context = 4096

            console.print(f"[yellow]Analyzing downloaded model file: {os.path.basename(model_path)}[/yellow]")
            # console.print(f"[yellow]Analyzing downloaded model file: {os.path.basename(model_path)}[/yellow]")
            try:
                # Use simple_gguf_info for more comprehensive details post-download
                path_str = str(model_path)
                info = simple_gguf_info(path_str)
                metadata = info.get("metadata", {})
                filename = os.path.basename(path_str)

                # Detect quantization from filename (fallback) or metadata
                quant_type = info.get("quantization_type") or detect_quantization_from_filename(filename)
                if quant_type:
                    console.print(f"[yellow]Detected quantization: [bold]{quant_type}[/bold][/yellow]")
                else:
                    console.print("[yellow]Could not detect quantization type.[/yellow]")
                # if quant_type:
                #     console.print(f"[yellow]Detected quantization: [bold]{quant_type}[/bold][/yellow]")
                # else:
                #     console.print("[yellow]Could not detect quantization type.[/yellow]")

                # Debug print for context length detection (already printed above)
                # console.print("[yellow]Attempting to detect maximum context length...[/yellow]")

                # Try to estimate RAM requirements using estimate_gguf_ram_requirements
                ram_reqs = estimate_gguf_ram_requirements(str(model_path), verbose=False)
                if ram_reqs:
                    # Use detected quant_type if available and present in ram_reqs, else fallback
                    quant_to_use = quant_type if quant_type and quant_type in ram_reqs else "Q4_K_M"
                    if quant_to_use in ram_reqs:
                        ram_gb = ram_reqs[quant_to_use]
                        ram_requirement = get_ram_requirement_string(ram_gb, colorize=True)
                        hardware_suggestion = get_hardware_suggestion(ram_gb)

                        # Clarify RAM estimation context
                        console.print(f"[yellow]Estimated RAM requirement ({quant_to_use}, base): {ram_requirement}[/yellow]")
                        console.print(f"[yellow]Hardware suggestion: {hardware_suggestion}[/yellow]")
                    else:
                        console.print(f"[yellow]Could not estimate RAM for quantization '{quant_to_use}'.[/yellow]")
                else:
                    console.print("[yellow]Could not estimate RAM requirements.[/yellow]")
                # if ram_reqs:
                #     # Use detected quant_type if available and present in ram_reqs, else fallback
                #     quant_to_use = quant_type if quant_type and quant_type in ram_reqs else "Q4_K_M"
                #     if quant_to_use in ram_reqs:
                #         ram_gb = ram_reqs[quant_to_use]
                #         ram_requirement = get_ram_requirement_string(ram_gb, colorize=True)
                #         hardware_suggestion = get_hardware_suggestion(ram_gb)

                #         # Clarify RAM estimation context
                #         console.print(f"[yellow]Estimated RAM requirement ({quant_to_use}, base): {ram_requirement}[/yellow]")
                #         console.print(f"[yellow]Hardware suggestion: {hardware_suggestion}[/yellow]")
                #     else:
                #         console.print(f"[yellow]Could not estimate RAM for quantization '{quant_to_use}'.[/yellow]")
                # else:
                #     console.print("[yellow]Could not estimate RAM requirements.[/yellow]")

                # Add context length information panel
                from rich.panel import Panel
                from rich.text import Text

                context_text = Text()
                # Clarify when context is detected
                context_text.append("Maximum context length is detected *after* download.\n", style="dim")
                if max_context and max_context != 4096: # Only show if detected and not default
                    context_text.append(f"Detected maximum context length: {max_context:,}\n", style="bold green")
                elif max_context == 4096:
                    context_text.append(f"Using default/detected maximum context length: {max_context:,}\n", style="yellow")
                else:
                    context_text.append("Could not determine the maximum supported context length for this model.\n", style="yellow")
                context_text.append("Larger context allows for longer conversations but requires more RAM.\n", style="cyan")
                context_text.append("You can manually set context length using the '/set context <size>' command in chat mode.", style="cyan") # Adjusted wording

                console.print(Panel(
                    context_text,
                    title="Context Length Information",
                    border_style="blue"
                ))
                # console.print(Panel(
                #     context_text,
                #     title="Context Length Information",
                #     border_style="blue"
                # ))
            except Exception as e:
                console.print(f"[yellow]Error analyzing downloaded model file: {str(e)}[/yellow]")
                # Print traceback if debug mode is enabled
                if os.environ.get("INFERNO_DEBUG", "0") == "1":
                    console.print(f"[dim]{traceback.format_exc()}[/dim]")
    except AttributeError as e:
        # Provide more context for AttributeError
        console.print(f"[bold red]Attribute Error during download: {str(e)}[/bold red]")
        console.print("[yellow]This might indicate an issue finding or processing the model file on Hugging Face Hub, or an internal error.[/yellow]")
        console.print(f"[dim]{traceback.format_exc()}[/dim]") # Print traceback for debugging
    except Exception as e:
        console.print(f"[bold red]Error downloading model: {str(e)}[/bold red]")
        console.print(f"[dim]{traceback.format_exc()}[/dim]") # Print traceback for debugging

def list_models_logic() -> None:
    """
    List downloaded models.
    """
    # Get system RAM for comparison
    system_ram = get_system_ram()

    # Get list of models
    models = model_manager.list_models()

    if not models:
        console.print("[yellow]No models found.[/yellow]")
        console.print("[cyan]Use 'inferno pull <repo_id>' to download a model.[/cyan]")
        return

    # Create main table for models
    table = Table(
        title="Downloaded Models",
        box=SIMPLE,
        show_header=True,
        header_style="bold cyan",
        expand=True
    )

    # Only include the essential columns
    table.add_column("Name", style="cyan")
    table.add_column("Size", style="magenta", justify="right")
    table.add_column("Quantization", style="yellow")
    table.add_column("RAM Usage", style="red", justify="right")
    table.add_column("Max Context", style="blue", justify="right")
    table.add_column("Downloaded", style="dim")

    for model in models:
        # Ensure model entry is a dict
        if not isinstance(model, dict):
            continue

        # Get file path and size
        file_path = model.get("path")
        file_size = "Unknown"
        size_bytes = 0

        if file_path:
            file_path = str(file_path)
        if file_path and os.path.exists(file_path):
            try:
                size_bytes = os.path.getsize(file_path)
                # Convert to human-readable format
                for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                    if size_bytes < 1024.0 or unit == 'TB':
                        file_size = f"{size_bytes:.2f} {unit}"
                        break
                    size_bytes /= 1024.0
            except Exception:
                pass

        # Format downloaded date
        downloaded_at = model.get("downloaded_at", "Unknown")
        if downloaded_at != "Unknown":
            try:
                dt = datetime.datetime.fromisoformat(downloaded_at)
                downloaded_at = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass

        # Get quantization type
        filename = model.get("filename", "")
        quant_type = detect_quantization_from_filename(filename) or "Unknown"

        # Get RAM usage and Max Context
        ram_usage = "Unknown"
        ram_color = "white"
        max_context = "Unknown"

        if file_path and os.path.exists(file_path):
            try:
                # Use simple_gguf_info to get details including context length
                gguf_info = simple_gguf_info(file_path)
                ctx_len = gguf_info.get("context_length")
                if ctx_len:
                    max_context = f"{ctx_len:,}"

                # Get RAM requirements
                ram_reqs = estimate_gguf_ram_requirements(str(file_path), verbose=False)
                if ram_reqs:
                    # Use detected quantization or fall back to Q4_K_M
                    fallback_quant = "Q4_K_M"
                    quant_to_use = quant_type if quant_type and quant_type in ram_reqs else fallback_quant
                    
                    if quant_to_use in ram_reqs:
                        ram_gb = ram_reqs[quant_to_use]
                        ram_usage = f"{ram_gb:.1f} GB"
                        
                        # Color code based on system RAM
                        if system_ram > 0:
                            if ram_gb > system_ram:
                                ram_color = "bold red"  # Exceeds available RAM
                            elif ram_gb > system_ram * 0.8:
                                ram_color = "bold yellow"  # Close to available RAM
                            else:
                                ram_color = "bold green"  # Well within available RAM
                    else:
                        # Handle case where even fallback quant isn't in ram_reqs
                        ram_usage = "N/A"
                        ram_color = "dim"

            except Exception as e:
                # Keep defaults if analysis fails, maybe log error in debug mode
                if os.environ.get("INFERNO_DEBUG", "0") == "1":
                    console.print(f"[dim]Error analyzing {filename} for list: {e}[/dim]")
                pass

        # Add row with only the essential columns
        table.add_row(
            model["name"],
            file_size,
            quant_type,
            f"[{ram_color}]{ram_usage}[/{ram_color}]",
            max_context,
            downloaded_at,
        )

    console.print(table)

    # Add a RAM usage comparison panel if we have models
    if models:
        # Import Panel and Text from rich if not already imported
        from rich.panel import Panel
        from rich.text import Text

        # Create a quantization comparison table
        quant_table = Table(
            title="RAM Usage by Quantization Type",
            show_header=True,
            header_style="bold cyan",
            box=SIMPLE
        )

        quant_table.add_column("Quantization", style="yellow")
        quant_table.add_column("Bits/Param", style="blue", justify="right")
        quant_table.add_column("RAM Multiplier", style="magenta", justify="right")
        quant_table.add_column("Description", style="green")

        # Quantization info
        quant_info = [
            ("Q2_K", "~2.5", "1.15×", "2-bit quantization (lowest quality, smallest size)"),
            ("Q3_K_M", "~3.5", "1.28×", "3-bit quantization (medium)"),
            ("Q4_K_M", "~4.5", "1.40×", "4-bit quantization (balanced quality/size)"),
            ("Q5_K_M", "~5.5", "1.65×", "5-bit quantization (better quality)"),
            ("Q6_K", "~6.5", "1.80×", "6-bit quantization (high quality)"),
            ("Q8_0", "~8.5", "2.00×", "8-bit quantization (very high quality)"),
            ("F16", "16.0", "2.80×", "16-bit float (highest quality, largest size)")
        ]

        for quant, bits, multiplier, desc in quant_info:
            quant_table.add_row(quant, bits, multiplier, desc)

        # Only show the system RAM info if we have it
        if system_ram > 0:
            console.print(Panel(
                Text(f"Your system has {system_ram:.1f} GB of RAM available", style="bold cyan"),
                title="System RAM",
                border_style="blue"
            ))

        console.print(quant_table)

    # Remove the duplicate quantization table
    # The following duplicate code block has been removed:
    # if models:
    #     # Create a quantization comparison table
    #     quant_table = Table(...)
    #     ...
    #     console.print(quant_table)

@app.command("list")
def list_models() -> None:
    """
    List downloaded models.
    """
    list_models_logic()

@app.command("ls", hidden=True)
def ls_models() -> None:
    """
    Alias for 'list'.
    """
    list_models_logic()

@app.command(name="remove", help="Remove a downloaded model")
def remove_model(
    model_string: str = typer.Argument(..., help="Name or filename of the model to remove"),
    force: bool = typer.Option(False, "--force", "-f", help="Force removal without confirmation"),
) -> None:
    """
    Remove a downloaded model.
    """
    # First check if this is a model name
    model_info = model_manager.get_model_info(str(model_string))

    # If not found by name, check if it's a filename
    if not model_info:
        for info in model_manager.list_models():
            if isinstance(info, dict) and info.get("filename") == model_string:
                model_info = info
                model_string = info.get("name", model_string)
                break

    if not model_info:
        console.print(f"[yellow]Model {model_string} not found.[/yellow]")
        return

    if not force:
        confirm = Prompt.ask(
            f"Are you sure you want to remove model {model_string}?",
            choices=["y", "n"],
            default="n",
        )

        if confirm.lower() != "y":
            console.print("[yellow]Operation cancelled.[/yellow]")
            return

    if model_manager.remove_model(model_string):
        console.print(f"[bold green]Model {model_string} removed successfully[/bold green]")
    else:
        console.print(f"[bold red]Error removing model {model_string}[/bold red]")

@app.command(name="rm", hidden=True)
def rm_model(
    model_string: str = typer.Argument(..., help="Name or filename of the model to remove"),
    force: bool = typer.Option(False, "--force", "-f", help="Force removal without confirmation"),
) -> None:
    """
    Alias for 'remove'.
    """
    remove_model(model_string, force)

@app.command("copy")
def copy_model(
    source: str = typer.Argument(..., help="Name of the source model"),
    destination: str = typer.Argument(..., help="Name for the destination model"),
) -> None:
    """
    Copy a model to a new name.
    """
    try:
        if model_manager.copy_model(source, destination):
            console.print(f"[bold green]Model {source} copied to {destination} successfully[/bold green]")
        else:
            console.print(f"[bold red]Failed to copy model {source} to {destination}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error copying model: {str(e)}[/bold red]")

@app.command("debug-context")
def debug_context(
    model_name: str = typer.Argument(..., help="Name of the model to debug context length for"),
) -> None:
    """
    Debug context length detection for a model.
    """
    # First check if this is a model name
    model_info = model_manager.get_model_info(str(model_name))

    # If not found by name, check if it's a filename
    if not model_info:
        for info in model_manager.list_models():
            if info.get("filename") == model_name:
                model_info = info
                model_name = info["name"]
                break

    if not model_info:
        console.print(f"[yellow]Model {model_name} not found.[/yellow]")
        return

    # Get file path
    file_path = model_info.get("path")
    if not file_path or not os.path.exists(file_path):
        console.print(f"[yellow]Model file not found at {file_path}.[/yellow]")
        return

    # Run the debug function
    try:
        debug_gguf_context_length(file_path)
    except Exception as e:
        console.print(f"[red]Error debugging context length: {str(e)}[/red]")
        if os.environ.get("INFERNO_DEBUG"):
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")


@app.command("show")
def show_model(
    model_name: str = typer.Argument(..., help="Name of the model to show information for"),
) -> None:
    """
    Show detailed information about a model.
    """
    # First check if this is a model name
    model_info = model_manager.get_model_info(str(model_name))

    # If not found by name, check if it's a filename
    if not model_info:
        for info in model_manager.list_models():
            if isinstance(info, dict) and info.get("filename") == model_name:
                model_info = info
                model_name = info.get("name", model_name)
                break

    if not model_info:
        console.print(f"[yellow]Model {model_name} not found.[/yellow]")
        return

    # Get file path
    file_path = model_info.get("path")
    if not file_path or not os.path.exists(file_path):
        console.print(f"[yellow]Model file not found at {file_path}.[/yellow]")
        return

    # Get file size
    try:
        size_bytes = os.path.getsize(file_path)
        # Convert to human-readable format
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0 or unit == 'TB':
                file_size = f"{size_bytes:.2f} {unit}"
                break
            size_bytes /= 1024.0
    except Exception:
        file_size = "Unknown"

    # Format downloaded date
    downloaded_at = model_info.get("downloaded_at", "Unknown")
    if downloaded_at != "Unknown":
        try:
            import datetime
            dt = datetime.datetime.fromisoformat(downloaded_at)
            downloaded_at = dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass

    # Basic information
    console.print(f"[bold cyan]Model: {model_name}[/bold cyan]")
    console.print(f"[cyan]Repository: {model_info.get('repo_id', 'Unknown')}[/cyan]")
    console.print(f"[cyan]Filename: {model_info.get('filename', 'Unknown')}[/cyan]")
    console.print(f"[cyan]Size: {file_size}[/cyan]")
    console.print(f"[cyan]Downloaded: {downloaded_at}[/cyan]")
    console.print(f"[cyan]Path: {file_path}[/cyan]")

    # Try to get detailed model information from the GGUF file
    try:
        # Use the simple GGUF reader (imported at the top of the file)
        info = simple_gguf_info(file_path)

        # Check if there was an error in the reader
        if "error" in info or len(info.get("metadata", {})) < 5:  # If we have very few metadata entries
            console.print("[yellow]Warning: GGUF reader encountered issues.[/yellow]")

            # If there's a traceback, print it in debug mode
            if "traceback" in info and os.environ.get("INFERNO_DEBUG"):
                console.print(f"[dim]{info['traceback']}[/dim]")

        # Create a table for model information
        console.print("\n[bold cyan]Model Information:[/bold cyan]")

        # Display GGUF version if available
        if "version" in info:
            console.print(f"[cyan]GGUF Version:[/cyan] {info['version']}")

        # Display file size
        if "file_size_gb" in info:
            console.print(f"[cyan]File Size:[/cyan] {info['file_size_gb']:.2f} GB ({info['file_size_bytes']:,} bytes)")

        # Display tensor count if available
        if "tensor_count" in info:
            console.print(f"[cyan]Tensor Count:[/cyan] {info['tensor_count']:,}")

        # Display metadata count if available
        if "kv_count" in info:
            console.print(f"[cyan]Metadata Key-Value Pairs:[/cyan] {info['kv_count']:,}")

        # Display architecture and name if available
        if "architecture" in info and info["architecture"]:
            console.print(f"[cyan]Architecture:[/cyan] {info['architecture']}")

        if "name" in info and info["name"]:
            console.print(f"[cyan]Model Name:[/cyan] {info['name']}")

        # Create a table for model parameters
        console.print("\n[bold green]Model Parameters:[/bold green]")

        # Display context length if available
        if "context_length" in info and info["context_length"]:
            if "context_length_source" in info:
                console.print(f"[green]Context Length:[/green] {info['context_length']:,} (from {info['context_length_source']})")
            else:
                console.print(f"[green]Context Length:[/green] {info['context_length']:,}")
        elif "context_length_error" in info:
            console.print(f"[yellow]Context Length Error:[/yellow] {info['context_length_error']}")
        elif "all_metadata_keys" in info:
            console.print("[yellow]No context length found. Available metadata keys:[/yellow]")
            for key in sorted(info["all_metadata_keys"]):
                console.print(f"  [dim]{key}[/dim]")

        # Display embedding length if available
        if "embedding_length" in info and info["embedding_length"]:
            console.print(f"[green]Embedding Length:[/green] {info['embedding_length']:,}")

        # Display block count if available
        if "block_count" in info and info["block_count"]:
            console.print(f"[green]Block Count:[/green] {info['block_count']:,}")

        # Display attention parameters if available
        if "head_count" in info and info["head_count"]:
            console.print(f"[green]Attention Head Count:[/green] {info['head_count']}")

        if "head_count_kv" in info and info["head_count_kv"]:
            console.print(f"[green]KV Head Count:[/green] {info['head_count_kv']}")

        # Display feed forward length if available
        if "feed_forward_length" in info and info["feed_forward_length"]:
            console.print(f"[green]Feed Forward Length:[/green] {info['feed_forward_length']:,}")

        # Create a table for quantization information
        console.print("\n[bold yellow]Quantization Information:[/bold yellow]")

        # Display quantization type if available
        if "quantization_type" in info and info["quantization_type"]:
            console.print(f"[yellow]Quantization Type:[/yellow] {info['quantization_type']}")

        # Create a table for RoPE parameters
        has_rope_params = False
        if "rope_freq_base" in info and info["rope_freq_base"]:
            if not has_rope_params:
                console.print("\n[bold magenta]RoPE Parameters:[/bold magenta]")
                has_rope_params = True
            console.print(f"[magenta]Frequency Base:[/magenta] {info['rope_freq_base']}")

        if "rope_freq_scale" in info and info["rope_freq_scale"]:
            if not has_rope_params:
                console.print("\n[bold magenta]RoPE Parameters:[/bold magenta]")
                has_rope_params = True
            console.print(f"[magenta]Frequency Scale:[/magenta] {info['rope_freq_scale']}")

        # Display all metadata if available
        if "metadata" in info and info["metadata"]:
            metadata = info["metadata"]

            # Find keys with specific suffixes
            suffix_groups = {
                "context_length": [],
                "embedding_length": [],
                "hidden_size": [],
                "block_count": [],
                "n_layer": [],
                "num_layers": [],
                "head_count": [],
                "num_heads": [],
                "rope.freq_base": [],
                "rope.freq_scale": [],
                "rope_freq_base": [],
                "rope_freq_scale": [],
                "feed_forward_length": [],
                "intermediate_size": []
            }

            # Collect keys by suffix
            for key in metadata.keys():
                for suffix in suffix_groups:
                    if key.endswith(f".{suffix}"):
                        suffix_groups[suffix].append(key)

            # Display keys by suffix groups
            has_important_keys = False
            for suffix, keys in suffix_groups.items():
                if keys:
                    if not has_important_keys:
                        console.print("\n[bold cyan]Important Metadata Keys:[/bold cyan]")
                        has_important_keys = True

                    # Choose color based on suffix type
                    if suffix in ["context_length"]:
                        color = "green"
                    elif suffix in ["rope.freq_base", "rope.freq_scale", "rope_freq_base", "rope_freq_scale"]:
                        color = "magenta"
                    elif suffix in ["head_count", "num_heads", "head_count_kv", "num_kv_heads"]:
                        color = "yellow"
                    else:
                        color = "cyan"

                    # Display keys with this suffix
                    for key in sorted(keys):
                        value = metadata[key]
                        # Format value based on type
                        if isinstance(value, (list, tuple)) and len(value) > 5:
                            value_str = f"[{', '.join(str(v) for v in value[:3])}, ..., {value[-1]}]"
                        elif isinstance(value, str) and len(value) > 100:
                            value_str = value[:97] + "..."
                        else:
                            value_str = str(value)

                        console.print(f"  [{color}]{key}:[/{color}] {value_str}")

            # Display tokenizer info if available
            tokenizer_keys = [k for k in metadata.keys() if k.startswith('tokenizer.') and
                            (k.endswith('_token_id') or k == 'tokenizer.ggml.model' or k == 'tokenizer.ggml.pre')]
            if tokenizer_keys:
                console.print("\n[bold blue]Tokenizer Information:[/bold blue]")
                for k in sorted(tokenizer_keys):
                    value = metadata[k]
                    console.print(f"  [blue]{k}:[/blue] {value}")

            # Option to show all metadata
            if os.environ.get("INFERNO_SHOW_ALL_METADATA"):
                # Group keys by prefix for better organization
                prefixes = {}
                for key in sorted(metadata.keys()):
                    if key.startswith('ERROR_'):
                        continue  # Skip error keys for now

                    prefix = key.split('.')[0] if '.' in key else 'other'
                    if prefix not in prefixes:
                        prefixes[prefix] = []
                    prefixes[prefix].append(key)

                console.print("\n[bold dim]All Metadata:[/bold dim]")

                # Show all metadata grouped by prefix
                for prefix in sorted(prefixes.keys()):
                    console.print(f"\n[bold dim][{prefix}][/bold dim]")
                    for key in sorted(prefixes[prefix]):
                        value = metadata[key]

                        # Format value based on type
                        if isinstance(value, (list, tuple)) and len(value) > 5:
                            value_str = f"[{', '.join(str(v) for v in value[:3])}, ..., {value[-1]}]"
                        elif isinstance(value, str) and len(value) > 100:
                            value_str = value[:97] + "..."
                        else:
                            value_str = str(value)

                        console.print(f"  [dim]{key}: {value_str}[/dim]")

            # Display any error keys
            error_keys = [k for k in metadata.keys() if k.startswith('ERROR_')]
            if error_keys:
                console.print("\n[yellow]Metadata parsing errors:[/yellow]")
                for k in sorted(error_keys):
                    console.print(f"  [dim yellow]{k}: {metadata[k]}[/dim yellow]")

    except Exception as e:
        console.print(f"[yellow]Could not extract detailed model information: {str(e)}[/yellow]")

        if os.environ.get("INFERNO_DEBUG"):
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")

        # Fallback to just extracting context length
        try:
            # Try again with the simple GGUF reader (imported at the top of the file)
            info = simple_gguf_info(file_path)

            # Check for context length in the top-level info
            if "context_length" in info and info["context_length"]:
                if "context_length_source" in info:
                    console.print(f"[cyan]Maximum context length: {info['context_length']:,} (from {info['context_length_source']})[/cyan]")
                else:
                    console.print(f"[cyan]Maximum context length: {info['context_length']:,}[/cyan]")
            elif "context_length_error" in info:
                console.print(f"[yellow]Context Length Error: {info['context_length_error']}[/yellow]")
            elif "all_metadata_keys" in info:
                console.print("[yellow]No context length found. Available metadata keys:[/yellow]")
                for key in sorted(info["all_metadata_keys"]):
                    if os.environ.get("INFERNO_DEBUG"):
                        console.print(f"  [dim]{key}[/dim]")
            # If not found, check the metadata for any keys ending with .context_length
            elif "metadata" in info:
                metadata = info["metadata"]
                # Check for qwen2.context_length specifically
                qwen2_key = 'qwen2.context_length'
                if qwen2_key in metadata:
                    try:
                        ctx_len = int(metadata[qwen2_key])
                        console.print(f"[cyan]Maximum context length from {qwen2_key}: {ctx_len:,}[/cyan]")
                    except (ValueError, TypeError):
                        if os.environ.get("INFERNO_DEBUG"):
                            console.print(f"[yellow]Found context length key {qwen2_key} but value is not a valid integer.[/yellow]")
                else:
                    # Look for any key ending with .context_length
                    context_length_keys = [k for k in metadata.keys() if k.endswith('.context_length')]
                    if context_length_keys:
                        # Use the first found context length
                        key = context_length_keys[0]
                        try:
                            ctx_len = int(metadata[key])
                            console.print(f"[cyan]Maximum context length from {key}: {ctx_len:,}[/cyan]")
                        except (ValueError, TypeError):
                            if os.environ.get("INFERNO_DEBUG"):
                                console.print(f"[yellow]Found context length key {key} but value is not a valid integer.[/yellow]")
                    else:
                        if os.environ.get("INFERNO_DEBUG"):
                            console.print("[yellow]No context length keys found in metadata.[/yellow]")
                            # Print all keys for debugging
                            console.print("[yellow]Available metadata keys:[/yellow]")
                            for key in sorted(metadata.keys()):
                                console.print(f"  [dim]{key}[/dim]")
        except Exception as e:
            if os.environ.get("INFERNO_DEBUG"):
                console.print(f"[yellow]Error extracting context length: {str(e)}[/yellow]")

    # Show RAM requirements
    try:
        ram_reqs = estimate_gguf_ram_requirements(str(file_path), verbose=False)
        if ram_reqs:
            # Get system RAM for comparison
            system_ram = get_system_ram()

            # Show parameter count
            if "model_params_billions" in ram_reqs:
                params = ram_reqs["model_params_billions"]
                console.print(f"[bold cyan]Parameters: {params:.2f} billion[/bold cyan]")

            # Try to detect quantization from filename
            quant_type = detect_quantization_from_filename(model_info.get("filename", "")) or "Q4_K_M"

            # Create a RAM requirements table with better spacing and alignment
            ram_table = Table(
                title="RAM Requirements by Quantization Type",
                show_header=True,
                header_style="bold cyan",
                box=SIMPLE,
                expand=True,
                padding=(0, 2)  # Add horizontal padding for better readability
            )

            ram_table.add_column("Quantization", style="yellow", no_wrap=True)
            ram_table.add_column("RAM Usage", style="magenta", justify="right", no_wrap=True)
            ram_table.add_column("Status", style="green", justify="center", no_wrap=True)
            ram_table.add_column("Description", style="dim")

            # Quantization descriptions
            quant_desc = {
                "Q2_K": "2-bit quantization (lowest quality, smallest size)",
                "Q2_K_S": "2-bit quantization with small block size",
                "Q3_K_S": "3-bit quantization with small block size",
                "Q3_K_M": "3-bit quantization with medium block size",
                "Q3_K_L": "3-bit quantization with large block size",
                "Q4_0": "4-bit quantization (legacy)",
                "Q4_1": "4-bit quantization with improved accuracy",
                "Q4_K_S": "4-bit quantization with small block size",
                "Q4_K_M": "4-bit quantization with medium block size (balanced)",
                "Q4_K_L": "4-bit quantization with large block size",
                "Q5_0": "5-bit quantization (legacy)",
                "Q5_1": "5-bit quantization with improved accuracy",
                "Q5_K_S": "5-bit quantization with small block size",
                "Q5_K_M": "5-bit quantization with medium block size",
                "Q5_K_L": "5-bit quantization with large block size",
                "Q6_K": "6-bit quantization (high quality)",
                "Q8_0": "8-bit quantization (very high quality)",
                "Q8_K": "8-bit quantization with K-quants",
                "F16": "16-bit float (highest quality, largest size)",
                "FP16": "16-bit float (highest quality, largest size)"
            }

            # Add rows for each quantization type
            for quant, ram in sorted(ram_reqs.items()):
                if quant not in ["context_overhead", "model_params_billions"] and isinstance(ram, (int, float)):
                    # Determine status based on system RAM
                    status = "Unknown"
                    status_color = "white"

                    if system_ram > 0:
                        if ram > system_ram:
                            status = "X Exceeds RAM"
                            status_color = "bold red"
                        elif ram > system_ram * 0.8:
                            status = "! Close to limit"
                            status_color = "bold yellow"
                        else:
                            status = "✓ Compatible"
                            status_color = "bold green"

                    # Highlight the detected quantization
                    quant_style = "bold yellow" if quant == quant_type else "yellow"

                    ram_table.add_row(
                        f"[{quant_style}]{quant}[/{quant_style}]",
                        get_ram_requirement_string(ram, colorize=True),
                        f"[{status_color}]{status}[/{status_color}]",
                        quant_desc.get(quant, "")
                    )

            console.print(ram_table)

            # Show hardware suggestion with better formatting
            hardware_suggestion = get_hardware_suggestion(ram_reqs.get(quant_type, ram_reqs.get("Q4_K_M", 0)))
            console.print(Panel(
                Text(f"Hardware suggestion: {hardware_suggestion}", style="cyan"),
                title="Recommended Hardware",
                border_style="blue",
                expand=False,
                padding=(1, 2)  # Add padding for better readability
            ))

            # Show system RAM if available with better formatting
            if system_ram > 0:
                console.print(Panel(
                    Text(f"Your system has {system_ram:.1f} GB of RAM available", style="bold cyan"),
                    title="System RAM",
                    border_style="blue",
                    expand=False,
                    padding=(1, 2)  # Add padding for better readability
                ))
    except Exception as e:
        console.print(f"[yellow]Could not estimate RAM requirements: {str(e)}[/yellow]")

    # Try to detect quantization from filename
    try:
        quant = detect_quantization_from_filename(model_info.get("filename", ""))
        if quant:
            console.print(f"[cyan]Detected quantization: {quant}[/cyan]")
    except Exception:
        pass

@app.command("ps")
def list_running_models() -> None:
    """
    List running models.
    """
    from ..api.server import loaded_models

    if not loaded_models:
        console.print("[yellow]No models currently running.[/yellow]")
        return

    table = Table(title="Running Models", box=SIMPLE)
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="green")

    for name in loaded_models.keys():
        table.add_row(name, "Running")

    console.print(table)
@app.command("run")
def chat(
    model_string: str = typer.Argument(..., help="Name or filename of the model to chat with"),
    n_gpu_layers: Optional[int] = typer.Option(None, help="Number of layers to offload to GPU (-1 for all)"),
    n_ctx: Optional[int] = typer.Option(None, help="Context window size (overrides detection)"),
    n_threads: Optional[int] = typer.Option(None, help="Number of threads to use for inference"),
    use_mlock: bool = typer.Option(False, help="Lock model in memory"),
) -> None:
    """
    Interactive chat with a model.
    """
    import base64
    import re
    try:
        from PIL import Image  # type: ignore
        import io
    except ImportError:
        console.print("[yellow]PIL not installed. Image support will be limited.[/yellow]")
        console.print("[yellow]Install with: pip install pillow[/yellow]")

    # First check if this is a filename that already exists
    model_path = model_manager.get_model_path(model_string)
    if model_path:
        # This is a filename that exists, find the model name
        # Ensure model_path is treated as a Path object if needed, or use os.path methods
        model_path_obj = Path(model_path) # Use Path object for consistency
        for model_info in model_manager.list_models():
            # Compare Path objects or normalized strings
            if model_info.get("filename") == model_string or Path(model_info.get("path", "")) == model_path_obj:
                model_name = model_info.get("name")
                break
        else:
            # Fallback to using the string as model name
            model_name = model_string
    else:
        # Use the string as model name
        model_name = model_string

        # Check if model exists, if not try to download it
        if not model_manager.get_model_path(model_name):
            console.print(f"[yellow]Model {model_name} not found locally. Attempting to download...[/yellow]")
            try:
                # Parse the model string to see if it's a repo_id:filename format
                # We don't need to use the parsed values directly as download_model handles this
                _ = model_manager.parse_model_string(model_string)  # Just to validate the format
                # Download the model
                model_name, model_path_obj = model_manager.download_model(model_string) # Get Path object back
                model_path = str(model_path_obj) # Convert back to string if needed later
                console.print(f"[bold green]Model {model_name} downloaded successfully[/bold green]")
            except Exception as e:
                console.print(f"[bold red]Error downloading model: {str(e)}[/bold red]")
                return

    # Ensure model_path is a string for os.path.exists and os.path.basename
    if isinstance(model_path, Path):
        model_path = str(model_path)

    # Initialize detected_max_context here
    detected_max_context = None
    ram_requirement = "Unknown"
    ram_reqs = None
    quant_type = None

    # Use os.path.exists correctly
    if model_path and os.path.exists(model_path):
        try:
            # Try to detect quantization from filename
            filename = os.path.basename(model_path)
            quant_type = detect_quantization_from_filename(filename)

            # Try to estimate RAM requirements from the model file
            path_str = str(model_path)
            ram_reqs = estimate_gguf_ram_requirements(path_str, verbose=False)
            if ram_reqs:
                # Use detected quantization or fall back to Q4_K_M
                quant_to_use = quant_type if quant_type and quant_type in ram_reqs else "Q4_K_M"
                if quant_to_use in ram_reqs:
                    ram_gb = ram_reqs[quant_to_use]
                    ram_requirement = get_ram_requirement_string(ram_gb, colorize=True)
                    hardware_suggestion = get_hardware_suggestion(ram_gb)

                    console.print(f"[yellow]Model quantization: [bold]{quant_type or 'Unknown'}[/bold][/yellow]")
                    # Clarify RAM estimation context
                    console.print(f"[yellow]Estimated RAM requirement ({quant_to_use}, base): {ram_requirement}[/yellow]")
                    console.print(f"[yellow]Hardware suggestion: {hardware_suggestion}[/yellow]")
        except Exception as e:
            console.print(f"[dim]Error estimating RAM requirements: {str(e)}[/dim]")

    # Fall back to size-based estimation if needed
    if ram_requirement == "Unknown":
        for size, ram in FALLBACK_RAM_REQUIREMENTS.items():
            if isinstance(model_name, str) and size in model_name:
                ram_requirement = ram
                console.print(f"[yellow]This model requires approximately {ram_requirement} of RAM (estimated from model name)[/yellow]")
                break

    # Check if we have enough RAM
    if ram_reqs:
        system_ram = get_system_ram()
        if system_ram > 0:
            # Use detected quantization or fall back to Q4_K_M
            quant_to_use = quant_type if quant_type and quant_type in ram_reqs else "Q4_K_M"
            if quant_to_use in ram_reqs:
                # Get RAM requirement with default context
                base_ram = ram_reqs[quant_to_use]
                context_overhead = ram_reqs.get("context_overhead", {})
                ctx_ram = context_overhead.get("Context 4096", 0) if isinstance(context_overhead, dict) else 0
                total_ram = base_ram + ctx_ram

                if total_ram > system_ram:
                    from rich.panel import Panel
                    from rich.text import Text

                    warning_text = Text()
                    # Clarify context used for warning
                    warning_text.append(f"WARNING: This model requires ~{total_ram:.2f} GB RAM (with 4096 context), but only {system_ram:.2f} GB is available!\n", style="bold red")
                    warning_text.append("The model may not load or could cause system instability.\n", style="bold red")
                    warning_text.append("\nConsider using a lower quantization level like Q3_K or Q2_K if available.", style="yellow")

                    console.print(Panel(
                        warning_text,
                        title="⚠️ INSUFFICIENT RAM ⚠️",
                        border_style="red"
                    ))

    # Try to detect max context length from the model file
    if model_path and os.path.exists(model_path):
        try:
            # Always use extract_max_context_from_gguf for max context detection
            from ..core.ram_estimator import extract_max_context_from_gguf
            detected_max_context = extract_max_context_from_gguf(model_path)
            if detected_max_context:
                console.print(f"[cyan]Detected maximum context length: {detected_max_context:,}, but we will use 4096[/cyan]")
            else:
                # Handle case where function returns None without raising an error
                console.print("[yellow]Could not detect maximum context length, using default (4096)[/yellow]")
                detected_max_context = 4096
        except Exception as e:
            console.print(f"[yellow]Error detecting context length: {str(e)}. Using default (4096)[/yellow]")
            detected_max_context = 4096

    # Load the model with 4096 context by default
    try:
        llm = LLMInterface(str(model_name))
        # Prioritize user-provided context, otherwise use 4096
        n_ctx_to_load = n_ctx or 4096
        llm.load_model(
            verbose=False,
            n_ctx=n_ctx_to_load,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            use_mlock=use_mlock
        )
    except Exception as e:
        console.print(f"[bold red]Error loading model: {str(e)}[/bold red]")
        return

    console.print(f"[bold green]Chat with {model_name}. Type '/help' for available commands or '/bye' to exit.[bold green]")

    # Chat history
    messages = []
    system_prompt = None

    # Initialize with empty system prompt
    messages.append({"role": "system", "content": ""})

    # Define help text
    help_text = """
    Available commands:
    /help or /? - Show this help message
    /bye - Exit the chat
    /set system <prompt> - Set the system prompt
    /set context <size> - Set context window size (reloads model)
    /show context - Show the current and maximum context window size
    /clear or /cls - Clear the terminal screen
    /reset - Reset chat history and system prompt
    /image <path> - Include an image in your next message
    
    Special syntax:
    #image:path/to/image.jpg - Include an image in your message
    """

    # Track if an image is pending for the next message
    pending_image = None

    while True:
        # Get user input
        user_input = input("\n> ")

        # Handle commands
        if user_input.startswith("/"):
            cmd_parts = user_input.split(maxsplit=2)
            cmd = cmd_parts[0].lower()

            if cmd == "/bye" or user_input.lower() in ["exit", "quit"]:
                console.print("[yellow]Goodbye![/yellow]")
                break

            elif cmd == "/help" or cmd == "/?":
                console.print(help_text)
                continue

            elif cmd == "/clear" or cmd == "/cls":
                # Do not clear history, just clear the terminal screen
                os.system('cls' if os.name == 'nt' else 'clear')
                console.print(f"[bold green]Chat with {model_name}. Type '/help' for available commands or '/bye' to exit.[/bold green]")
                console.print("[yellow]Screen cleared. Chat history preserved.[/yellow]")
                continue

            elif cmd == "/reset":
                messages = [{"role": "system", "content": ""}]
                system_prompt = None
                console.print("[yellow]All settings reset.[/yellow]")
                continue
                
            # Markdown command removed - always enabled

            elif cmd == "/show" and len(cmd_parts) >= 2 and cmd_parts[1].lower() == "context":
                # Access context size via the underlying llama object's n_ctx() method
                current_ctx = llm.llm.n_ctx() if llm.llm else None
                if current_ctx:
                    console.print(f"[cyan]Current context window size: {current_ctx:,}[/cyan]")
                    # Show the max detected context as well
                    if detected_max_context:
                        console.print(f"[cyan]Maximum detected context size for this model: {detected_max_context:,}[/cyan]")
                    else:
                        # Check info.json as a fallback if detection failed during run but might exist from download
                        model_info = model_manager.get_model_info(str(model_name))
                        saved_max_ctx = model_info.get("max_context") if isinstance(model_info, dict) else None
                        if saved_max_ctx:
                            console.print(f"[cyan]Maximum context size (from saved info): {saved_max_ctx:,}[/cyan]")
                        else:
                            console.print("[yellow]Maximum context size for this model was not detected or saved.[/yellow]")
                else:
                    console.print("[yellow]Could not determine current context size (model might not be loaded).[/yellow]")
                continue

            elif cmd == "/set" and len(cmd_parts) >= 2:
                if len(cmd_parts) < 3:
                    console.print("[red]Error: Missing value for setting[/red]")
                    continue

                setting = cmd_parts[1].lower()
                value = cmd_parts[2]

                if setting == "system":
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]

                    system_prompt = value
                    # Update system message
                    if messages and messages[0].get("role") == "system":
                        messages[0]["content"] = system_prompt
                    else:
                        # Clear messages and add system prompt
                        messages = [{"role": "system", "content": system_prompt}]

                    # Print confirmation that it's been applied
                    console.print("[yellow]System prompt set to:[/yellow]")
                    console.print(f"[cyan]\"{system_prompt}\"[/cyan]")
                    console.print("[green]System prompt applied. Next responses will follow this instruction.[/green]")
                elif setting == "context":
                    try:
                        context_size = int(value)
                        # Reload the model with new context size
                        console.print(f"[yellow]Reloading model with context size: {context_size}...[/yellow]")
                        llm.load_model(
                            n_ctx=context_size,
                            verbose=False,
                            n_gpu_layers=n_gpu_layers, # Pass existing options during reload
                            n_threads=n_threads,
                            use_mlock=use_mlock
                        )
                        console.print(f"[green]Context size set to: {context_size}[/green]")
                    except ValueError:
                        console.print(f"[red]Invalid context size: {value}. Must be an integer.[/red]")
                else:
                    console.print(f"[red]Unknown setting: {setting}[/red]")
                continue
            else:
                console.print(f"[red]Unknown command: {cmd}[/red]")
                continue

        # Check for image references in the message
        image_pattern = r'#image:(.*?)(?:\s|$)'
        image_matches = re.findall(image_pattern, user_input)

        if image_matches:
            # This is a multimodal message with images
            try:
                # Start with text content
                content_parts = []
                
                # Extract text without the image tags
                text_content = re.sub(image_pattern, '', user_input).strip()
                if text_content:
                    content_parts.append({"type": "text", "text": text_content})
                
                # Process each image reference
                for img_path in image_matches:
                    img_path = img_path.strip()
                    
                    # Handle Windows paths with escaped backslashes in the regex match
                    img_path = img_path.replace('\\\\', '\\')
                    
                    # Normalize path (handles both relative and absolute paths)
                    img_path = os.path.normpath(img_path)
                    
                    if not os.path.exists(img_path):
                        console.print(f"[red]Image file not found: {img_path}[/red]")
                        continue
                    
                    try:
                        # Validate image
                        img = Image.open(img_path)
                        img.verify()
                        
                        # Convert to base64
                        with open(img_path, "rb") as img_file:
                            img_data = img_file.read()
                            base64_img = base64.b64encode(img_data).decode('utf-8')
                        
                        # Get MIME type based on file extension
                        mime_type = "image/jpeg"  # Default
                        if img_path.lower().endswith(".png"):
                            mime_type = "image/png"
                        elif img_path.lower().endswith(".gif"):
                            mime_type = "image/gif"
                        elif img_path.lower().endswith(".webp"):
                            mime_type = "image/webp"
                        
                        # Add image part
                        content_parts.append({
                            "type": "image_url", 
                            "image_url": {"url": f"data:{mime_type};base64,{base64_img}"}
                        })
                        console.print(f"[cyan]Image attached: {img_path}[/cyan]")
                    except Exception as e:
                        console.print(f"[red]Error processing image {img_path}: {str(e)}[/red]")
                
                # Create multimodal message if we have content parts
                if content_parts:
                    messages.append({"role": "user", "content": content_parts})
                else:
                    # Fallback to text-only if all images failed
                    messages.append({"role": "user", "content": user_input})
            except Exception as e:
                console.print(f"[red]Error creating multimodal message: {str(e)}[/red]")
                # Fallback to text-only
                messages.append({"role": "user", "content": user_input})
        else:
            # Regular text message
            messages.append({"role": "user", "content": user_input})

        # Generate response
        console.print("\n")  # Add extra spacing between user input and response

        # Use a buffer to collect the response
        response_buffer = ""
       
        # Use create_chat_completion with stream=True
        stream = llm.create_chat_completion(
            messages=messages,
            stream=True,
        )

        # Process the stream with ghost text effect
        for chunk in stream:
            # Be defensive: stream items may not always be dicts
            if not isinstance(chunk, dict):
                continue

            choices = chunk.get("choices")
            if not isinstance(choices, list) or len(choices) == 0:
                continue

            first_choice = choices[0]
            if not isinstance(first_choice, dict):
                continue

            delta = first_choice.get("delta", {})
            if not isinstance(delta, dict):
                continue

            content = delta.get("content")
            if content is not None:
                token = content
                response_buffer += token

                for char in token:
                    console.print(char, end="", highlight=True)
        
        # Add a newline after streaming is complete
        console.print("")
        
        
        # Add the collected response to history
        messages.append({"role": "assistant", "content": response_buffer})

def list_quantization_methods(use_imatrix: bool = False) -> None:
    """Display available quantization methods."""
    console.print(QuantizationMethod.list_methods(use_imatrix))

@app.command(name="quantize")
def quantize_command(
    model: Optional[str] = typer.Argument(None, help="Name or HF repo of the model to quantize (format: 'name' or 'hf:org/model')"),
    output: Optional[str] = typer.Argument(None, help="Name for the quantized output (optional for HF models)"),
    method: str = typer.Option("q4_k_m", help="Quantization method (use '--method list' to see available methods)"),
    imatrix: bool = typer.Option(False, help="Use importance matrix quantization"),
    train_data: Optional[str] = typer.Option(None, help="Training data file for imatrix quantization"),
    split: bool = typer.Option(False, help="Split the model into smaller parts"),
    split_size: Optional[str] = typer.Option(None, help="Maximum size for split parts (e.g. '2G')")
):
    """Quantize a model to a different format."""
    # Handle list method without requiring model arguments
    if method == "list":
        list_quantization_methods(imatrix)
        return

    # Require model argument
    if model is None:
        console.print("[red]Error: MODEL argument is required for quantization[/red]")
        console.print("\nTo see available quantization methods, use: inferno quantize --method list")
        console.print("\nExample usage:")
        console.print("  Local model:  inferno quantize model_name output_name --method q4_k_m")
        console.print("  HuggingFace: inferno quantize hf:org/model --method q4_k_m")
        raise typer.Exit(1)

    manager = ModelManager()
    model_name = model


    # Handle HuggingFace model loading
    if model.startswith("hf:"):
        repo_id = model[3:]  # Remove 'hf:' prefix
        
        # Show available quantization methods first
        console.print("\n[bold cyan]Available Quantization Methods:[/bold cyan]")
        methods_table = QuantizationMethod.list_methods()
        console.print(methods_table)

        # Let user select method
        method = Prompt.ask(
            "\nSelect quantization method",
            choices=list(QuantizationMethod.METHODS.keys()),
            default="q4_k_m"
        )

        # Show selected method details
        method_info = QuantizationMethod.METHODS[method]
        from rich.panel import Panel
        from rich.text import Text

        details = Text()
        details.append("Selected Quantization Method:\n\n", style="bold cyan")
        details.append(f"Method: {method}\n", style="green")
        details.append(f"Description: {method_info['description']}\n", style="yellow")
        details.append(f"Bits/Param: {method_info['bits']}\n", style="blue")
        details.append(f"RAM Usage: {method_info['ram_multiplier']}\n", style="magenta")

        console.print(Panel(
            details,
            title="Quantization Configuration",
            border_style="blue"
        ))

        # Generate output name based on model and method
        model_name = repo_id.split('/')[-1]
        if output is None:
            output = f"{model_name}-{method}"

        # Ensure output path is in the models directory
        output_path = Path(manager.models_dir) / f"{output}.gguf"
        output = str(output_path)

        # Confirm before downloading
        if not Prompt.ask(f"\nProceed with downloading and quantizing {repo_id}?", choices=["y", "n"], default="y") == "y":
            console.print("[yellow]Operation cancelled.[/yellow]")
            return

        # Download model
        console.print(f"\n[yellow]Downloading raw model from {repo_id} for quantization...[/yellow]")
        temp_model_path = None
        try:
            # Download the raw model to temp directory
            model_name, temp_model_path = manager.download_raw_model(repo_id)
            console.print(f"[bold green]Model {model_name} downloaded successfully[/bold green]")
            
            # Perform quantization
            console.print(f"\n[bold blue]Quantizing model using method: {method}[/bold blue]")
            
            # Use temp_model_path directly for quantization
            output_files = manager.quantize_model(
                model_name=model_name,
                output_name=output,
                method=method,
                use_imatrix=imatrix,
                train_data=train_data,
                split_model=split,
                split_size=split_size
            )

            # Show results
            console.print("\n[bold green]Successfully quantized model to:[/bold green]")
            for f in output_files:
                console.print(f"  - {f}")

            # Show final model information
            if output_files:
                try:
                    output_path = output_files[0]  # Use first output file
                    info = simple_gguf_info(output_path)
                    quant_type = info.get("quantization_type") or detect_quantization_from_filename(output_path)
                    
                    if quant_type:
                        console.print(f"\n[cyan]Final quantization type: {quant_type}[/cyan]")

                    # Show RAM requirements for quantized model
                    ram_reqs = estimate_gguf_ram_requirements(str(output_path), verbose=False)
                    if ram_reqs and quant_type in ram_reqs:
                        ram_gb = ram_reqs[quant_type]
                        ram_requirement = get_ram_requirement_string(ram_gb, colorize=True)
                        console.print(f"[cyan]RAM requirement: {ram_requirement}[/cyan]")

                except Exception as e:
                    console.print(f"[yellow]Error analyzing output file: {str(e)}[/yellow]")

        except Exception as e:
            console.print(f"[bold red]Error downloading or quantizing model: {str(e)}[/bold red]")
            if os.environ.get("INFERNO_DEBUG"):
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
            raise typer.Exit(1)
        finally:
            # Clean up temp directory if it exists
            if temp_model_path and Path(temp_model_path).exists():
                import shutil
                shutil.rmtree(Path(temp_model_path).parent, ignore_errors=True)
    elif output is None:
        console.print("[red]Error: OUTPUT argument is required for local models[/red]")
        raise typer.Exit(1)
        

@app.command(name="compare")
def compare_command(
    models: List[str] = typer.Argument(..., help="List of model names to compare")
):
    """Compare multiple models, showing size and metrics."""
    try:
        manager = ModelManager()
        manager.compare_models(models)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

@app.command(name="estimate")
def estimate_command(
    model: str = typer.Argument(..., help="Name of the model to analyze")
):
    """Show estimated RAM usage for different quantization methods."""
    try:
        manager = ModelManager()
        manager.estimate_ram_usage(model)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

@app.command("version")
def version() -> None:
    """
    Show version information.
    """
    from ..version import __version__
    console.print(f"[bold]Inferno[/bold] version [cyan]{__version__}[/cyan]")
    console.print("🔥 Ignite Your Local AI Experience 🔥")
