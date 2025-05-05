"""
Model management for Inferno
"""

import os
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import shutil

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from huggingface_hub import hf_hub_download, HfFileSystem

from ..utils.config import config
from .quantizer import ModelQuantizer
from ..core.ram_estimator import extract_max_context_from_gguf

console = Console()

class ModelManager:
    """
    Manager for downloading and managing models.
    Handles model download, listing, removal, and path resolution.
    """
    models_dir: Path

    def __init__(self) -> None:
        self.models_dir = config.models_dir

    def parse_model_string(self, model_string: str) -> Tuple[str, Optional[str]]:
        """
        Parse a model string in the format 'repo_id:filename' or just 'repo_id'.
        Args:
            model_string (str): The model string to parse.
        Returns:
            Tuple[str, Optional[str]]: (repo_id, filename)
        """
        if ":" in model_string:
            repo_id, filename = model_string.split(":", 1)
            return repo_id, filename
        else:
            return model_string, None

    def list_repo_gguf_files(self, repo_id: str) -> List[str]:
        """
        List all GGUF files in a repository.
        Args:
            repo_id (str): The Hugging Face repository ID.
        Returns:
            List[str]: List of filenames.
        """
        fs = HfFileSystem()
        try:
            files = fs.ls(repo_id, detail=False)
            gguf_files = [os.path.basename(f) for f in files if f.endswith(".gguf")]
            return gguf_files
        except Exception as e:
            console.print(f"[bold red]Error listing files in repository {repo_id}: {str(e)}[/bold red]")
            return []

    def select_file_interactive(self, repo_id: str) -> Optional[str]:
        """
        Interactively select a file from a repository.
        Args:
            repo_id (str): The Hugging Face repository ID.
        Returns:
            Optional[str]: Selected filename or None if cancelled.
        """
        from ..core.ram_estimator import (
            estimate_from_huggingface_repo,
            detect_quantization_from_filename,
            get_ram_requirement_string,
            get_system_ram,
            extract_max_context_from_gguf
        )
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text

        gguf_files = self.list_repo_gguf_files(repo_id)
        if not gguf_files:
            console.print(f"[bold red]No GGUF files found in repository {repo_id}[/bold red]")
            return None

        # Try to get RAM estimates for files in the repo
        ram_estimates = {}
        quant_types = {}
        file_sizes = {}

        try:
            repo_estimates = estimate_from_huggingface_repo(repo_id)
            if repo_estimates and "all_files" in repo_estimates:
                for filename, file_info in repo_estimates["all_files"].items():
                    if filename in gguf_files:
                        # Store file size
                        file_sizes[filename] = file_info.get('size_bytes', 0) / (1024**3)  # Convert to GB

                        # Detect quantization from filename
                        quant_type = detect_quantization_from_filename(filename)
                        if quant_type:
                            quant_types[filename] = quant_type
                            if quant_type in repo_estimates:
                                ram_estimates[filename] = repo_estimates[quant_type]
        except Exception as e:
            console.print(f"[dim]Error estimating RAM requirements: {str(e)}[/dim]")

        # Get system RAM for comparison
        system_ram = get_system_ram()

        # Group files by quantization type
        quant_groups = {}
        for filename in gguf_files:
            quant = quant_types.get(filename, "Unknown")
            if quant not in quant_groups:
                quant_groups[quant] = []
            quant_groups[quant].append(filename)

        # Create a table for displaying the files
        from rich.box import SIMPLE
        table = Table(title=f"[bold blue]Available GGUF Files in {repo_id}[/bold blue]",
                     show_header=True,
                     header_style="bold cyan",
                     box=SIMPLE,
                     expand=True)

        table.add_column("#", style="dim", width=4)
        table.add_column("Filename", style="green")
        table.add_column("Quantization", style="yellow")
        table.add_column("Size", style="blue", justify="right")
        table.add_column("RAM Usage", style="magenta", justify="right")
        table.add_column("Max Context", style="cyan", justify="right")

        # Add files to the table, grouped by quantization type
        file_index = 1
        for quant_type, files in sorted(quant_groups.items()):
            for filename in sorted(files):
                # Get RAM usage info
                ram_info = "Unknown"
                ram_color = "white"
                if filename in ram_estimates:
                    ram_gb = ram_estimates[filename]
                    ram_info = get_ram_requirement_string(ram_gb, colorize=False)

                    # Color code based on system RAM
                    if system_ram > 0:
                        if ram_gb > system_ram:
                            ram_color = "bold red"  # Exceeds available RAM
                        elif ram_gb > system_ram * 0.8:
                            ram_color = "bold yellow"  # Close to available RAM
                        else:
                            ram_color = "bold green"  # Well within available RAM

                # Get file size
                size_info = "Unknown"
                if filename in file_sizes:
                    size_gb = file_sizes[filename]
                    size_info = f"{size_gb:.2f} GB"

                # Try to get context length from repo metadata
                context_info = "Auto (4096)"  # Default value
                if "all_files" in repo_estimates and filename in repo_estimates["all_files"]:
                    file_info = repo_estimates["all_files"][filename]
                    if "max_context" in file_info and file_info["max_context"]:
                        context_info = f"{file_info['max_context']}"

                table.add_row(
                    f"[{file_index}]",
                    filename,
                    quant_types.get(filename, "Unknown"),
                    size_info,
                    f"[{ram_color}]{ram_info}[/{ram_color}]",
                    context_info
                )
                file_index += 1

        console.print(table)

        # Add a RAM usage comparison panel if we have estimates
        if ram_estimates:
            # Create a quantization comparison table
            quant_table = Table(title="RAM Usage by Quantization Type",
                               show_header=True,
                               header_style="bold cyan",
                               box=SIMPLE)

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

            # Only show the comparison if we have system RAM info
            if system_ram > 0:
                console.print(Panel(
                    Text(f"Your system has {system_ram:.1f} GB of RAM available", style="bold cyan"),
                    title="System RAM",
                    border_style="blue"
                ))

            console.print(quant_table)

            # Add context length information panel
            context_text = Text()
            context_text.append("Inferno automatically sets context length to 4096 tokens by default.\n", style="bold cyan")
            context_text.append("The 'Max Context' column shows the maximum supported context length for each model.\n", style="cyan")
            context_text.append("Larger context allows for longer conversations but requires more RAM.\n", style="cyan")
            context_text.append("You can manually set context length with the 'context' command in chat mode.", style="dim")

            console.print(Panel(
                context_text,
                title="Context Length Information",
                border_style="blue"
            ))

        choice = Prompt.ask(
            "Select a file to download (number or filename)",
            default="1"
        )
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(gguf_files):
                return gguf_files[idx]
        except ValueError:
            if choice in gguf_files:
                return choice
        console.print(f"[bold red]Invalid selection: {choice}[/bold red]")
        return None

    def download_model(self, model_string: str, filename: Optional[str] = None) -> Tuple[str, Path]:
        """
        Download a GGUF model from Hugging Face Hub.
        Args:
            model_string (str): The model string in format 'repo_id' or 'repo_id:filename'.
            filename (Optional[str]): Specific filename to download, overrides filename in model_string.
        Returns:
            Tuple[str, Path]: (model_name, model_path)
        """
        repo_id, file_from_string = self.parse_model_string(model_string)
        filename = filename or file_from_string
        model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
        model_dir = config.get_model_path(model_name)
        model_dir.mkdir(exist_ok=True, parents=True)
        model_info: Dict[str, Any] = {
            "repo_id": repo_id,
            "name": model_name,
            "downloaded_at": datetime.datetime.now().isoformat(),
        }
        # Save initial info in case selection/download fails
        with open(model_dir / "info.json", "w") as f:
            json.dump(model_info, f, indent=2)

        if not filename:
            console.print(f"[yellow]No filename provided, searching for GGUF files in {repo_id}...[/yellow]")
            filename = self.select_file_interactive(repo_id)
            if not filename:
                # Clean up the created directory and info file if no file is selected
                if (model_dir / "info.json").exists():
                    (model_dir / "info.json").unlink()
                if not any(model_dir.iterdir()): # Remove dir only if empty
                    model_dir.rmdir()
                raise ValueError(f"No GGUF file selected from repository {repo_id}")
            console.print(f"[green]Selected GGUF file: {filename}[/green]")

        console.print(f"[bold blue]Downloading {filename} from {repo_id}...[/bold blue]")
        try:
            # hf_hub_download uses tqdm internally, which is reasonable for a progress bar.
            # Replacing it with rich progress requires more complex integration or a different download method.
            model_path_str = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=model_dir,
                # progress=True is the default
            )
            model_path = Path(model_path_str)
        except Exception as e:
            console.print(f"[bold red]Error downloading file: {str(e)}[/bold red]")
            # Clean up potentially incomplete download and info file
            if (model_dir / filename).exists():
                (model_dir / filename).unlink()
            if (model_dir / "info.json").exists():
                (model_dir / "info.json").unlink()
            if not any(model_dir.iterdir()): # Remove dir only if empty
                 model_dir.rmdir()
            raise

        console.print(f"[bold green]Model downloaded to {model_path}[/bold green]")

        # Update info with filename and path
        model_info["filename"] = filename
        model_info["path"] = str(model_path)

        # Detect and store max context length after download
        max_context = None
        try:
            max_context = extract_max_context_from_gguf(str(model_path))
            # if max_context:
            #     console.print(f"[cyan]Detected and saved maximum context length: {max_context:,}[/cyan]")
            # else:
            #     console.print("[yellow]Could not detect maximum context length from downloaded file.[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Error detecting context length post-download: {str(e)}[/yellow]")
        model_info["max_context"] = max_context # Store detected length (or None)

        # Save final info file
        with open(model_dir / "info.json", "w") as f:
            json.dump(model_info, f, indent=2)

        return model_name, model_path

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a downloaded model.
        Args:
            model_name (str): Name of the model.
        Returns:
            Optional[Dict[str, Any]]: Model info dict or None if not found.
        """
        model_dir = config.get_model_path(model_name)
        info_file = model_dir / "info.json"
        if not info_file.exists():
            # Try finding by filename if name lookup fails
            for info in self.list_models():
                if info.get("filename") == model_name:
                    return info
            return None
        try:
            with open(info_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            console.print(f"[red]Error reading info file for model {model_name}[/red]")
            return None
        except Exception as e:
            console.print(f"[red]Unexpected error reading info file for {model_name}: {e}[/red]")
            return None

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all downloaded models with their information.
        Returns:
            List[Dict[str, Any]]: List of model info dicts.
        """
        models: List[Dict[str, Any]] = []
        seen_paths: set = set()
        if not config.models_dir.exists():
            return []
        model_dirs = [d for d in config.models_dir.iterdir() if d.is_dir()]
        for model_dir in model_dirs:
            if ":" in model_dir.name:
                continue
            info_file = model_dir / "info.json"
            if info_file.exists():
                try:
                    with open(info_file, "r") as f:
                        info = json.load(f)
                    if "path" in info and info["path"] in seen_paths:
                        continue
                    if "path" in info:
                        seen_paths.add(info["path"])
                    models.append(info)
                except Exception:
                    pass
        return models

    def remove_model(self, model_name: str) -> bool:
        """
        Remove a downloaded model.
        Args:
            model_name (str): Name of the model to remove.
        Returns:
            bool: True if removed, False if not found.
        """
        model_dir = config.get_model_path(model_name)
        if not model_dir.exists():
            return False
        shutil.rmtree(model_dir)
        return True

    def get_model_path(self, model_name: str) -> Optional[str]:
        """
        Get the path to a model file.
        Args:
            model_name (str): Name or filename of the model.
        Returns:
            Optional[str]: Path to the model file or None if not found.
        """
        info = self.get_model_info(model_name)
        if not info or "path" not in info:
            for model_info in self.list_models():
                if model_info.get("filename") == model_name:
                    return model_info.get("path")
            return None
        return info["path"]

    def quantize_model(
        self,
        model_name: str,
        output_name: str,
        method: str,
        use_imatrix: bool = False,
        train_data: Optional[str] = None,
        split_model: bool = False,
        split_size: Optional[str] = None
    ) -> List[str]:
        """
        Quantize a model to a different format.
        
        Args:
            model_name: Name of the source model
            output_name: Name for the quantized model
            method: Quantization method to use
            use_imatrix: Whether to use importance matrix quantization
            train_data: Path to training data for imatrix quantization
            split_model: Whether to split the model
            split_size: Maximum size for split parts (e.g. "2G")
            
        Returns:
            List of paths to the quantized model files
        """
        # Get source model info
        source_info = self.get_model_info(model_name)
        if not source_info or "path" not in source_info:
            raise ValueError(f"Source model {model_name} not found")

        # Create output directory
        output_dir = config.get_model_path(output_name)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Initialize quantizer
        quantizer = ModelQuantizer()

        try:
            # Quantize the model
            output_base = output_dir / f"{output_name}.gguf"
            output_files = quantizer.quantize_model(
                input_model=source_info["path"],
                output_model=str(output_base),
                method=method,
                use_imatrix=use_imatrix,
                train_data=train_data,
                split_model=split_model,
                split_size=split_size
            )

            # Create info files for each output
            for output_path in output_files:
                output_path = Path(output_path)
                model_info = source_info.copy()
                model_info.update({
                    "name": output_path.stem,
                    "path": str(output_path),
                    "quantized_from": model_name,
                    "quantization_method": method,
                    "quantization_type": "imatrix" if use_imatrix else "standard",
                    "split_model": split_model
                })

                info_path = output_path.parent / "info.json"
                with open(info_path, "w") as f:
                    json.dump(model_info, f, indent=2)

            return output_files

        except Exception as e:
            console.print(f"[bold red]Error quantizing model: {str(e)}[/bold red]")
            # Clean up on error
            if output_dir.exists():
                shutil.rmtree(output_dir)
            raise

    def compare_models(self, model_names: List[str]) -> None:
        """
        Compare multiple models, showing size, quantization, and other metrics.
        
        Args:
            model_names: List of model names to compare
        """
        model_paths = []
        for name in model_names:
            path = self.get_model_path(name)
            if path:
                model_paths.append(path)
            else:
                console.print(f"[yellow]Warning: Model {name} not found[/yellow]")

        if model_paths:
            quantizer = ModelQuantizer()
            comparison = quantizer.compare_models(model_paths)
            console.print(comparison)
        else:
            console.print("[red]No valid models to compare[/red]")

    def estimate_ram_usage(self, model_name: str) -> None:
        """
        Show estimated RAM usage for different quantization methods.
        
        Args:
            model_name: Name of the model to analyze
        """
        path = self.get_model_path(model_name)
        if not path:
            console.print(f"[red]Model {model_name} not found[/red]")
            return

        quantizer = ModelQuantizer()
        estimates = quantizer.estimate_ram_usage(path)

        table = Table(title=f"RAM Usage Estimates for {model_name}")
        table.add_column("Method", style="cyan")
        table.add_column("Estimated RAM", style="green", justify="right")
        
        for method, ram in sorted(estimates.items()):
            table.add_row(method, f"{ram:.2f} GB")
            
        console.print(table)

    def download_raw_model(self, repo_id: str) -> Tuple[str, Path]:
        """
        Download a raw model from Hugging Face Hub into a temporary directory.
        Args:
            repo_id (str): The Hugging Face repository ID.
        Returns:
            Tuple[str, Path]: (model_name, temp_path)
        """
        from huggingface_hub import snapshot_download
        import tempfile
        
        model_name = repo_id.split("/")[-1]
        temp_dir = tempfile.mkdtemp(prefix=f"inferno_{model_name}_")

        try:
            console.print(f"[yellow]Downloading model files from {repo_id}...[/yellow]")
            local_dir = snapshot_download(
                repo_id=repo_id,
                local_dir=temp_dir,
                local_dir_use_symlinks=False
            )
            return model_name, Path(local_dir)
            
        except Exception as e:
            # Clean up on error
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"Error downloading model: {str(e)}")

    def copy_model(self, source_model: str, destination_model: str) -> bool:
        """
        Copy a model to a new name.
        Args:
            source_model (str): Name of the source model.
            destination_model (str): Name for the destination model.
        Returns:
            bool: True if copied successfully, False otherwise.
        """
        # Get source model info
        source_info = self.get_model_info(source_model)
        if not source_info or "path" not in source_info:
            console.print(f"[bold red]Source model {source_model} not found[/bold red]")
            return False

        # Create destination directory
        dest_dir = config.get_model_path(destination_model)
        dest_dir.mkdir(exist_ok=True, parents=True)

        # Copy the model file
        source_path = Path(source_info["path"])
        dest_path = dest_dir / source_path.name

        try:
            console.print(f"[bold blue]Copying model from {source_path} to {dest_path}...[/bold blue]")
            shutil.copy2(source_path, dest_path)

            # Create info file for the destination model
            dest_info = source_info.copy()
            dest_info["name"] = destination_model
            dest_info["path"] = str(dest_path)
            dest_info["copied_from"] = source_model
            dest_info["copied_at"] = datetime.datetime.now().isoformat()

            with open(dest_dir / "info.json", "w") as f:
                json.dump(dest_info, f, indent=2)

            console.print(f"[bold green]Model copied successfully to {dest_path}[/bold green]")
            return True
        except Exception as e:
            console.print(f"[bold red]Error copying model: {str(e)}[/bold red]")
            # Clean up if there was an error
            if dest_path.exists():
                dest_path.unlink()
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            return False
