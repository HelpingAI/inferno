"""
Model quantization tools for Inferno.

This module provides tools for quantizing models to different formats,
with support for importance matrix quantization and model splitting.
"""

import os
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from rich.console import Console
from rich.table import Table

from ..utils.config import config
from .gguf_reader import GGUFReader

console = Console()

class QuantizationMethod:
    """Represents a GGUF quantization method with its properties."""
    
    METHODS: Dict[str, Dict[str, str]] = {
        "f16": {
            "description": "16-bit floating point - maximum accuracy, largest size",
            "bits": "16.0",
            "ram_multiplier": "2.80×",
            "file_size": "2GB per billion parameters"
        },
        "q2_k": {
            "description": "2-bit quantization (smallest size, lowest accuracy)",
            "bits": "~2.5",
            "ram_multiplier": "1.15×",
            "file_size": "0.3GB per billion parameters"
        },
        "q3_k_l": {
            "description": "3-bit quantization (large) - balanced for size/accuracy",
            "bits": "~3.5",
            "ram_multiplier": "1.28×",
            "file_size": "0.45GB per billion parameters"
        },
        "q3_k_m": {
            "description": "3-bit quantization (medium) - good balance for most use cases",
            "bits": "~3.5",
            "ram_multiplier": "1.28×",
            "file_size": "0.45GB per billion parameters"
        },
        "q3_k_s": {
            "description": "3-bit quantization (small) - optimized for speed",
            "bits": "~3.5",
            "ram_multiplier": "1.28×",
            "file_size": "0.45GB per billion parameters"
        },
        "q4_0": {
            "description": "4-bit quantization (version 0) - standard 4-bit compression",
            "bits": "~4.5",
            "ram_multiplier": "1.40×",
            "file_size": "0.6GB per billion parameters"
        },
        "q4_1": {
            "description": "4-bit quantization (version 1) - improved accuracy over q4_0",
            "bits": "~4.5",
            "ram_multiplier": "1.40×",
            "file_size": "0.6GB per billion parameters"
        },
        "q4_k_m": {
            "description": "4-bit quantization (medium) - balanced for most models",
            "bits": "~4.5",
            "ram_multiplier": "1.40×",
            "file_size": "0.6GB per billion parameters"
        },
        "q4_k_s": {
            "description": "4-bit quantization (small) - optimized for speed",
            "bits": "~4.5",
            "ram_multiplier": "1.40×",
            "file_size": "0.6GB per billion parameters"
        },
        "q5_0": {
            "description": "5-bit quantization (version 0) - high accuracy, larger size",
            "bits": "~5.5",
            "ram_multiplier": "1.65×",
            "file_size": "0.75GB per billion parameters"
        },
        "q5_1": {
            "description": "5-bit quantization (version 1) - improved accuracy over q5_0",
            "bits": "~5.5",
            "ram_multiplier": "1.65×",
            "file_size": "0.75GB per billion parameters"
        },
        "q5_k_m": {
            "description": "5-bit quantization (medium) - best balance for quality/size",
            "bits": "~5.5",
            "ram_multiplier": "1.65×",
            "file_size": "0.75GB per billion parameters"
        },
        "q5_k_s": {
            "description": "5-bit quantization (small) - optimized for speed",
            "bits": "~5.5",
            "ram_multiplier": "1.65×",
            "file_size": "0.75GB per billion parameters"
        },
        "q6_k": {
            "description": "6-bit quantization - highest accuracy, largest size",
            "bits": "~6.5",
            "ram_multiplier": "1.80×",
            "file_size": "0.9GB per billion parameters"
        },
        "q8_0": {
            "description": "8-bit quantization - maximum accuracy, largest size",
            "bits": "~8.5",
            "ram_multiplier": "2.00×",
            "file_size": "1.2GB per billion parameters"
        }
    }

    IMATRIX_METHODS: Dict[str, Dict[str, str]] = {
        "iq3_m": {
            "description": "3-bit imatrix quantization (medium) - balanced importance-based",
            "bits": "~3.5",
            "ram_multiplier": "1.28×",
            "file_size": "0.45GB per billion parameters"
        },
        "iq3_xxs": {
            "description": "3-bit imatrix quantization (extra extra small) - maximum compression",
            "bits": "~3.5",
            "ram_multiplier": "1.28×",
            "file_size": "0.45GB per billion parameters"
        },
        "iq4_nl": {
            "description": "4-bit imatrix quantization (non-linear) - best accuracy for 4-bit",
            "bits": "~4.5",
            "ram_multiplier": "1.40×",
            "file_size": "0.6GB per billion parameters"
        },
        "iq4_xs": {
            "description": "4-bit imatrix quantization (extra small) - maximum compression",
            "bits": "~4.5",
            "ram_multiplier": "1.40×",
            "file_size": "0.6GB per billion parameters"
        }
    }

    @classmethod
    def get_method_info(cls, method: str, use_imatrix: bool = False) -> Optional[Dict[str, str]]:
        """Get information about a quantization method."""
        if use_imatrix:
            return cls.IMATRIX_METHODS.get(method)
        return cls.METHODS.get(method)

    @classmethod
    def list_methods(cls, use_imatrix: bool = False) -> Table:
        """Create a table listing all available quantization methods."""
        table = Table(
            title="Available Quantization Methods",
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("Method", style="green")
        table.add_column("Bits/Param", style="blue", justify="right")
        table.add_column("File Size", style="yellow", justify="right")
        table.add_column("RAM Usage", style="magenta", justify="right")
        table.add_column("Description", style="dim")

        methods = cls.IMATRIX_METHODS if use_imatrix else cls.METHODS
        for method, info in sorted(methods.items()):
            table.add_row(
                method,
                info["bits"],
                info.get("file_size", "Unknown"),
                info["ram_multiplier"],
                info["description"]
            )

        return table

class ModelQuantizer:
    """Handles model quantization operations."""

    def __init__(self):
        """Initialize the quantizer."""
        self.models_dir = config.models_dir

    def setup_llama_cpp(self) -> Tuple[str, str]:
        """
        Set up and build llama.cpp repository in a temporary directory.
        
        Returns:
            Tuple[str, str]: Path to temp directory and path to llama.cpp build directory
        """
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="inferno_llama_cpp_")
        llama_path = Path(temp_dir) / "llama.cpp"
        
        with console.status("[bold green]Setting up llama.cpp...") as status:
            # Clone llama.cpp
            subprocess.run(
                ['git', 'clone', 'https://github.com/ggerganov/llama.cpp', str(llama_path)],
                check=True,
                cwd=temp_dir
            )
            
            # Install requirements
            subprocess.run(
                ['pip', 'install', '-r', 'requirements.txt'],
                check=True,
                cwd=str(llama_path)
            )
            
            # Configure and build
            subprocess.run(
                ['cmake', '-B', 'build'],
                check=True,
                cwd=str(llama_path)
            )
            subprocess.run(
                ['cmake', '--build', 'build', '--parallel'],
                check=True,
                cwd=str(llama_path)
            )
            
            return temp_dir, str(llama_path / "build")

    def generate_importance_matrix(
        self,
        model_path: str,
        train_data_path: str,
        output_path: str,
        build_dir: str
    ) -> None:
        """
        Generate importance matrix for quantization.
        
        Args:
            model_path: Path to input model
            train_data_path: Path to training data
            output_path: Path to output imatrix
            build_dir: Path to llama.cpp build directory
        """
        imatrix_command = [
            str(Path(build_dir) / "bin" / "llama-imatrix"),
            "-m", model_path,
            "-f", train_data_path,
            "-ngl", "99",
            "--output-frequency", "10",
            "-o", output_path
        ]

        console.print("[bold green]Generating importance matrix...")
        process = subprocess.Popen(imatrix_command)

        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            console.print("[yellow]Imatrix computation timed out. Sending SIGINT...")
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

        if process.returncode != 0:
            raise RuntimeError("Failed to generate importance matrix")

    def quantize_model(
        self,
        input_model: str,
        output_model: str,
        method: str,
        use_imatrix: bool = False,
        train_data: Optional[str] = None,
        split_model: bool = False,
        split_size: Optional[str] = None
    ) -> List[str]:
        """
        Quantize a model using the specified method.
        
        Args:
            input_model: Path to input model (can be raw HF model or GGUF)
            output_model: Path to output model
            method: Quantization method to use
            use_imatrix: Whether to use importance matrix quantization
            train_data: Path to training data for imatrix
            split_model: Whether to split the model
            split_size: Maximum size for split parts (e.g. "2G")
            
        Returns:
            List of output model paths (multiple if split)
        """
        # Check if input is a raw HF model
        if Path(input_model).is_dir():
            # Convert raw model to GGUF first
            import subprocess
            
            with console.status("[bold green]Converting raw model to GGUF..."):
                # Use convert.py from llama.cpp
                temp_dir, build_dir = self.setup_llama_cpp()
                try:
                    convert_script = Path(temp_dir) / "llama.cpp" / "convert.py"
                    subprocess.run([
                        "python", str(convert_script),
                        input_model,
                        "--outfile", output_model
                    ], check=True)
                    
                    # Now input_model is the converted GGUF
                    input_model = output_model
                    
                finally:
                    # Clean up temp dir
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
        # Validate method
        if use_imatrix:
            if method not in QuantizationMethod.IMATRIX_METHODS:
                raise ValueError(f"Invalid imatrix quantization method: {method}")
        else:
            if method not in QuantizationMethod.METHODS:
                raise ValueError(f"Invalid quantization method: {method}")

        try:
            # Set up llama.cpp in temp directory
            temp_dir, build_dir = self.setup_llama_cpp()

            # Generate importance matrix if needed
            if use_imatrix:
                if not train_data:
                    raise ValueError("Training data required for imatrix quantization")
                imatrix_path = str(Path(output_model).parent / "imatrix.dat")
                self.generate_importance_matrix(input_model, train_data, imatrix_path, build_dir)

            # Quantize the model
            quantize_cmd = [str(Path(build_dir) / "bin" / "llama-quantize")]
            if use_imatrix:
                quantize_cmd.extend(["--imatrix", imatrix_path])
            quantize_cmd.extend([input_model, output_model, method])

            result = subprocess.run(quantize_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Quantization failed: {result.stderr}")

            output_files = [output_model]

            # Split model if requested
            if split_model:
                split_cmd = [str(Path(build_dir) / "bin" / "llama-gguf-split"), "--split"]
                if split_size:
                    split_cmd.extend(["--split-max-size", split_size])
                split_cmd.extend([output_model, str(Path(output_model).parent)])

                result = subprocess.run(split_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"Model splitting failed: {result.stderr}")

                # Get list of split files
                output_dir = Path(output_model).parent
                prefix = Path(output_model).stem
                output_files = sorted(str(f) for f in output_dir.glob(f"{prefix}*.gguf"))

            # Clean up imatrix file if it was generated
            if use_imatrix and os.path.exists(imatrix_path):
                os.remove(imatrix_path)

            return output_files

        finally:
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def compare_models(self, models: List[str]) -> Table:
        """
        Create a comparison table of multiple models.
        
        Args:
            models: List of model paths to compare
            
        Returns:
            Rich table with comparison
        """
        table = Table(
            title="Model Comparison",
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("Model", style="green")
        table.add_column("Size", style="blue", justify="right")
        table.add_column("GB/Param", style="yellow", justify="right")
        table.add_column("Quantization", style="magenta")
        table.add_column("Context Length", style="cyan", justify="right")
        
        for model_path in models:
            path = Path(model_path)
            size_gb = path.stat().st_size / (1024**3)
            
            reader = GGUFReader(model_path)
            info = reader.get_model_info()
            
            # Get number of parameters in billions
            params_billions = info.get("num_params_billions", 0)
            if params_billions > 0:
                gb_per_param = size_gb / params_billions
                gb_per_param_str = f"{gb_per_param:.2f}"
            else:
                gb_per_param_str = "Unknown"
            
            table.add_row(
                path.name,
                f"{size_gb:.2f} GB",
                gb_per_param_str,
                info.get("quantization_type", "Unknown"),
                str(info.get("context_length", "Unknown"))
            )
            
        return table

    def estimate_ram_usage(self, model_path: str) -> Dict[str, Any]:
        """
        Estimate RAM usage and file size for different quantization methods.
        
        Args:
            model_path: Path to the model
            
        Returns:
            Dict containing RAM and file size estimates for each method
        """
        # Get model info to determine number of parameters
        reader = GGUFReader(model_path)
        info = reader.get_model_info()
        params_billions = info.get("num_params_billions", 0)
        
        # Get current model size
        current_size = Path(model_path).stat().st_size / (1024**3)
        
        estimates = {}
        for method, info in QuantizationMethod.METHODS.items():
            # Extract RAM multiplier from string like "1.40×"
            ram_multiplier = float(info["ram_multiplier"].rstrip("×"))
            
            # Extract file size per billion parameters from string like "0.6GB per billion parameters"
            file_size_str = info.get("file_size", "")
            if file_size_str and params_billions > 0:
                try:
                    file_size_gb = float(file_size_str.split("GB")[0].strip())
                    estimated_file_size = file_size_gb * params_billions
                except ValueError:
                    estimated_file_size = None
            else:
                estimated_file_size = None
            
            estimates[method] = {
                "ram_gb": current_size * ram_multiplier,
                "file_size_gb": estimated_file_size,
                "ram_multiplier": ram_multiplier,
                "params_billions": params_billions
            }
            
        return estimates
