"""
Command-line interface for Inferno
"""

import os # Ensure os is imported globally
import typer
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.box import SIMPLE
from typing import Optional
from ..core.model_manager import ModelManager
# Removed duplicate import: from ..core.quantizer import QuantizationMethod
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
    n_gpu_layers: Optional[int] = typer.Option(None, help="Number of layers to offload to GPU (-1 for all)"),
    n_ctx: Optional[int] = typer.Option(None, help="Context window size"),
    n_threads: Optional[int] = typer.Option(None, help="Number of threads to use for inference"),
    use_mlock: bool = typer.Option(False, help="Lock model in memory"),
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

                    # console.print(f"[yellow]Model quantization: [bold]{quant_type or 'Unknown'}[/bold][/yellow]")
                    # # Clarify RAM estimation context
                    # console.print(f"[yellow]Estimated RAM requirement ({quant_to_use}, base): {ram_requirement}[/yellow]")
                    # console.print(f"[yellow]Hardware suggestion: {hardware_suggestion}[/yellow]")
        except Exception as e:
            console.print(f"[dim]Error estimating RAM requirements: {str(e)}[/dim]")

    # Fall back to size-based estimation if needed
    if ram_requirement == "Unknown":
        for size, ram in FALLBACK_RAM_REQUIREMENTS.items():
            if size in model_name:
                ram_requirement = ram
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
            # if detected_max_context:
            #     console.print(f"[cyan]Detected maximum context length: {detected_max_context:,}[/cyan]")
            # else:
            #     # Handle case where function returns None without raising an error
            #     console.print("[yellow]Could not detect maximum context length, using default (4096)[/yellow]")
            #     detected_max_context = 4096
        except Exception as e:
            console.print(f"[yellow]Error detecting context length: {str(e)}. Using default (4096)[/yellow]")
            detected_max_context = 4096

    # Load the model with provided options
    try:
        llm = LLMInterface(model_name)
        # Use provided context length or fallback to detected/default
        n_ctx_to_load = n_ctx or (int(detected_max_context) if detected_max_context is not None else 4096)
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
    
    # Start server with model options
    start_server(host=host, port=port, model_options=model_options)

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
                # if max_context:
                #     console.print(f"[cyan]Detected maximum context length: {max_context:,}[/cyan]")
                # else:
                #     # Handle case where function returns None without raising an error
                #     console.print("[yellow]Could not detect maximum context length, using default (4096)[/yellow]")
                #     max_context = 4096
            except Exception as e:
                console.print(f"[yellow]Error detecting context length: {str(e)}. Using default (4096)[/yellow]")
                max_context = 4096

            # console.print(f"[yellow]Analyzing downloaded model file: {os.path.basename(model_path)}[/yellow]")
            try:
                # Use simple_gguf_info for more comprehensive details post-download
                info = simple_gguf_info(model_path)
                metadata = info.get("metadata", {})
                filename = os.path.basename(model_path)

                # Detect quantization from filename (fallback) or metadata
                quant_type = info.get("quantization_type") or detect_quantization_from_filename(filename)
                # if quant_type:
                #     console.print(f"[yellow]Detected quantization: [bold]{quant_type}[/bold][/yellow]")
                # else:
                #     console.print("[yellow]Could not detect quantization type.[/yellow]")

                # Debug print for context length detection (already printed above)
                # console.print("[yellow]Attempting to detect maximum context length...[/yellow]")

                # Try to estimate RAM requirements using estimate_gguf_ram_requirements
                ram_reqs = estimate_gguf_ram_requirements(model_path, verbose=False)
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
    Logic to list downloaded models.
    """
    import datetime # Moved import inside the function
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
