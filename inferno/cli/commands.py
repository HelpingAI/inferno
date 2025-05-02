"""
Command-line interface for Inferno
"""

import os # Ensure os is imported globally
import typer
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.box import SIMPLE
from typing import Optional, Dict, Any
from pathlib import Path # Import Path for better path handling

from ..core.model_manager import ModelManager
from ..core.llm import LLMInterface
from ..api.server import start_server
from ..core.ram_estimator import (
    estimate_gguf_ram_requirements,
    get_ram_requirement_string,
    get_hardware_suggestion,
    estimate_ram_for_all_models,
    detect_gpu_vram,
    print_gpu_compatibility,
    suggest_hardware,
    get_system_ram,
    estimate_from_huggingface_repo,
    detect_quantization_from_filename
)
from ..core.gguf_reader import simple_gguf_info, debug_gguf_context_length

app: typer.Typer = typer.Typer(help="Inferno - A llama-cpp-python based LLM serving tool")
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
    model_string: str = typer.Argument(..., help="Model to run (format: 'name', 'repo_id' or 'repo_id:filename')"),
    host: Optional[str] = typer.Option(None, help="Host to bind the server to"),
    port: Optional[int] = typer.Option(None, help="Port to bind the server to"),
) -> None:
    """
    Start a model server (downloads if needed).
    """
    # First check if this is a filename that already exists
    model_path = model_manager.get_model_path(model_string)
    if model_path:
        # This is a filename that exists, find the model name
        for model_info in model_manager.list_models():
            if model_info.get("filename") == model_string or model_info.get("path") == model_path:
                model_name = model_info.get("name")
                break
        else:
            # Fallback to using the string as model name
            model_name = model_string
    else:
        # Parse the model string to see if it's a repo_id:filename format
        repo_id, _ = model_manager.parse_model_string(model_string)
        model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id

        # Check if model exists, if not try to download it
        if not model_manager.get_model_path(model_name):
            console.print(f"[yellow]Model {model_name} not found locally. Attempting to download...[/yellow]")
            try:
                # We don't need to use the parsed values directly as download_model handles this
                _ = model_manager.parse_model_string(model_string)  # Just to validate the format
                # Download the model
                model_name, _ = model_manager.download_model(model_string)
                console.print(f"[bold green]Model {model_name} downloaded successfully[/bold green]")
            except Exception as e:
                console.print(f"[bold red]Error downloading model: {str(e)}[/bold red]")
                return

    # Check RAM requirements
    model_path = model_manager.get_model_path(model_name)
    ram_requirement = "Unknown"
    ram_reqs = None
    quant_type = None

    if model_path and os.path.exists(model_path):
        try:
            # Try to detect quantization from filename
            filename = os.path.basename(model_path)
            quant_type = detect_quantization_from_filename(filename)

            # Try to estimate RAM requirements from the model file
            ram_reqs = estimate_gguf_ram_requirements(model_path, verbose=False)
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
            if size in model_name:
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
                ctx_ram = ram_reqs.get("context_overhead", {}).get("Context 4096", 0)
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
                console.print(f"[cyan]Detected maximum context length: {detected_max_context:,}[/cyan]")
            else:
                # Handle case where function returns None without raising an error
                console.print("[yellow]Could not detect maximum context length, using default (4096)[/yellow]")
                detected_max_context = 4096
        except Exception as e:
            console.print(f"[yellow]Error detecting context length: {str(e)}. Using default (4096)[/yellow]")
            detected_max_context = 4096

    # Load the model with detected context length if available
    try:
        llm = LLMInterface(model_name)
        # Ensure detected_max_context is an integer, default to 4096 if None
        n_ctx_to_load = int(detected_max_context) if detected_max_context is not None else 4096
        llm.load_model(verbose=False, n_ctx=n_ctx_to_load)
    except Exception as e:
        console.print(f"[bold red]Error loading model: {str(e)}[/bold red]")
        return

    # Start the server
    console.print(f"[bold blue]Starting Inferno server with model {model_name}...[/bold blue]")
    start_server(host=host, port=port)

@app.command("pull")
def pull_model(
    model_string: str = typer.Argument(..., help="Model to download (format: 'repo_id' or 'repo_id:filename')"),
) -> None:
    """
    Download a model from Hugging Face without running it.
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
                max_context = extract_max_context_from_gguf(model_path)
                if max_context:
                    console.print(f"[cyan]Detected maximum context length: {max_context:,}[/cyan]")
                else:
                    # Handle case where function returns None without raising an error
                    console.print("[yellow]Could not detect maximum context length, using default (4096)[/yellow]")
                    max_context = 4096
            except Exception as e:
                console.print(f"[yellow]Error detecting context length: {str(e)}. Using default (4096)[/yellow]")
                max_context = 4096

            console.print(f"[yellow]Analyzing downloaded model file: {os.path.basename(model_path)}[/yellow]")
            try:
                # Use simple_gguf_info for more comprehensive details post-download
                info = simple_gguf_info(model_path)
                metadata = info.get("metadata", {})
                filename = os.path.basename(model_path)

                # Detect quantization from filename (fallback) or metadata
                quant_type = info.get("quantization_type") or detect_quantization_from_filename(filename)
                if quant_type:
                    console.print(f"[yellow]Detected quantization: [bold]{quant_type}[/bold][/yellow]")
                else:
                    console.print("[yellow]Could not detect quantization type.[/yellow]")

                # Debug print for context length detection (already printed above)
                # console.print("[yellow]Attempting to detect maximum context length...[/yellow]")

                # Try to estimate RAM requirements using estimate_gguf_ram_requirements
                ram_reqs = estimate_gguf_ram_requirements(model_path, verbose=False)
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

@app.command("list")
def list_models() -> None:
    """
    List downloaded models.
    """
    from rich.panel import Panel
    from rich.text import Text
    import datetime

    models = model_manager.list_models()

    if not models:
        console.print("[yellow]No models found. Use 'inferno pull' to download a model.[/yellow]")
        return

    # Get system RAM for comparison
    system_ram = get_system_ram()

    # Create main table for models
    table = Table(
        title="Downloaded Models",
        box=SIMPLE,
        show_header=True,
        header_style="bold cyan",
        expand=True
    )

    table.add_column("Name", style="cyan")
    table.add_column("Repository", style="green")
    table.add_column("Filename", style="blue")
    table.add_column("Size", style="magenta", justify="right")
    table.add_column("Quantization", style="yellow")
    table.add_column("RAM Usage", style="red", justify="right")
    table.add_column("Max Context", style="blue", justify="right")
    table.add_column("Downloaded", style="dim")

    for model in models:
        # Get file path and size
        file_path = model.get("path")
        file_size = "Unknown"
        size_bytes = 0

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
        context_source = "" # Add source info

        if file_path and os.path.exists(file_path):
            try:
                # Use simple_gguf_info to get details including context length
                gguf_info = simple_gguf_info(file_path)
                ctx_len = gguf_info.get("context_length")
                if ctx_len:
                    max_context = f"{ctx_len:,}"
                    context_source = gguf_info.get("context_length_source", "") # Get source if available

                # Estimate RAM requirements (simple_gguf_info doesn't do this)
                ram_reqs = estimate_gguf_ram_requirements(file_path, verbose=False)
                if ram_reqs:
                     # Determine quant_type for RAM lookup (use detected or fallback)
                     quant_for_ram = quant_type if quant_type != "Unknown" and quant_type in ram_reqs else "Q4_K_M"
                     if quant_for_ram in ram_reqs:
                         ram_gb = ram_reqs[quant_for_ram]
                         ram_usage = get_ram_requirement_string(ram_gb, colorize=False)

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

        table.add_row(
            model["name"],
            model.get("repo_id", "Unknown"),
            filename,
            file_size,
            quant_type,
            f"[{ram_color}]{ram_usage}[/{ram_color}]",
            max_context, # Use context length from simple_gguf_info
            downloaded_at,
        )

    console.print(table)

    # Add a RAM usage comparison panel if we have models
    if models:
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

@app.command(name="remove", help="Remove a downloaded model")
def remove_model(
    model_string: str = typer.Argument(..., help="Name or filename of the model to remove"),
    force: bool = typer.Option(False, "--force", "-f", help="Force removal without confirmation"),
) -> None:
    """
    Remove a downloaded model.
    """
    # First check if this is a model name
    model_info = model_manager.get_model_info(model_string)

    # If not found by name, check if it's a filename
    if not model_info:
        for info in model_manager.list_models():
            if info.get("filename") == model_string:
                model_info = info
                model_string = info["name"]
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
    model_info = model_manager.get_model_info(model_name)

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
    model_info = model_manager.get_model_info(model_name)

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
            console.print(f"[yellow]Warning: GGUF reader encountered issues.[/yellow]")

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
            console.print(f"[yellow]No context length found. Available metadata keys:[/yellow]")
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

    # Estimate RAM requirements
    try:
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text

        ram_reqs = estimate_gguf_ram_requirements(file_path, verbose=False)
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

            # Show context overhead in a separate table with better formatting
            if "context_overhead" in ram_reqs:
                ctx_table = Table(
                    title="Context Length RAM Overhead",
                    show_header=True,
                    header_style="bold cyan",
                    box=SIMPLE,
                    expand=True,
                    padding=(0, 2)  # Add horizontal padding for better readability
                )

                ctx_table.add_column("Context Length", style="blue", no_wrap=True)
                ctx_table.add_column("Additional RAM", style="magenta", justify="right", no_wrap=True)
                ctx_table.add_column("Total RAM (with Q4_K_M)", style="green", justify="right", no_wrap=True)

                base_ram = ram_reqs.get(quant_type, ram_reqs.get("Q4_K_M", 0))

                for ctx, ram in sorted(ram_reqs["context_overhead"].items()):
                    if isinstance(ram, (int, float)):
                        ctx_len = ctx.replace("Context ", "")
                        total_ram = base_ram + ram

                        ctx_table.add_row(
                            ctx_len,
                            f"+{ram:.2f} GB",
                            get_ram_requirement_string(total_ram, colorize=True)
                        )

                console.print(ctx_table)

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

    table = Table(title="Running Models", box="SIMPLE")
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="green")

    for name in loaded_models.keys():
        table.add_row(name, "Running")

    console.print(table)
@app.command("run")
def chat(
    model_string: str = typer.Argument(..., help="Name or filename of the model to chat with"),
) -> None:
    """
    Interactive chat with a model.
    """
    # Ensure os is available in this scope if needed, though global import should work
    # import os # This line is likely redundant if the global import is correct, but added as a safeguard

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
            ram_reqs = estimate_gguf_ram_requirements(model_path, verbose=False)
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
            if size in model_name:
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
                ctx_ram = ram_reqs.get("context_overhead", {}).get("Context 4096", 0)
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
                console.print(f"[cyan]Detected maximum context length: {detected_max_context:,}[/cyan]")
            else:
                # Handle case where function returns None without raising an error
                console.print("[yellow]Could not detect maximum context length, using default (4096)[/yellow]")
                detected_max_context = 4096
        except Exception as e:
            console.print(f"[yellow]Error detecting context length: {str(e)}. Using default (4096)[/yellow]")
            detected_max_context = 4096

    # Load the model with detected context length if available
    try:
        llm = LLMInterface(model_name)
        # Ensure detected_max_context is an integer, default to 4096 if None
        n_ctx_to_load = int(detected_max_context) if detected_max_context is not None else 4096
        llm.load_model(verbose=False, n_ctx=n_ctx_to_load)
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
    """

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
                         model_info = model_manager.get_model_info(model_name)
                         saved_max_ctx = model_info.get("max_context") if model_info else None
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
                    console.print(f"[yellow]System prompt set to:[/yellow]")
                    console.print(f"[cyan]\"{system_prompt}\"[/cyan]")
                    console.print(f"[green]System prompt applied. Next responses will follow this instruction.[/green]")

                    # Force a test message to ensure the system prompt is applied
                    test_messages = messages.copy()
                    test_messages.append({"role": "user", "content": "Say 'System prompt active'."})

                    # Test if the system prompt is working
                    console.print("[dim]Testing system prompt...[/dim]")
                    response = llm.create_chat_completion(
                        messages=test_messages,
                        stream=False,
                        max_tokens=20
                    )
                    console.print("[dim]System prompt test complete.[/dim]")
                elif setting == "context":
                    try:
                        context_size = int(value)
                        # Reload the model with new context size
                        console.print(f"[yellow]Reloading model with context size: {context_size}...[/yellow]")
                        llm.load_model(n_ctx=context_size, verbose=False)
                        console.print(f"[green]Context size set to: {context_size}[/green]")
                    except ValueError:
                        console.print(f"[red]Invalid context size: {value}. Must be an integer.[/red]")
                else:
                    console.print(f"[red]Unknown setting: {setting}[/red]")
                continue
            else:
                console.print(f"[red]Unknown command: {cmd}[/red]")
                continue

        # Add user message to history
        messages.append({"role": "user", "content": user_input})

        # Generate response
        console.print("\n")  # Add extra spacing between user input and response

        # Use a buffer to collect the response
        response_buffer = ""

        def print_token(token):
            nonlocal response_buffer
            response_buffer += token
            console.print(token, end="", highlight=False)

        llm.stream_chat_completion(
            messages=messages,
            callback=print_token,
        )

        # Get the full response to add to history
        response = llm.create_chat_completion(
            messages=messages,
            stream=False,
        )

        assistant_message = response["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": assistant_message})

        # Add extra spacing after the response
        console.print("")

@app.command("version")
def version() -> None:
    """
    Show version information.
    """
    from ..version import __version__
    console.print(f"[bold]Inferno[/bold] version [cyan]{__version__}[/cyan]")
    console.print("A llama-cpp-python based LLM serving tool")
