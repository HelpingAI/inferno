"""
LLM interface for Inferno using llama-cpp-python
"""

from typing import Dict, Any, List, Optional, Union, Generator, Callable

from llama_cpp import Llama
from rich.console import Console

from ..utils.config import config
from .model_manager import ModelManager
from .gguf_reader import simple_gguf_info

console = Console()

class LLMInterface:
    """
    Interface for LLM models using llama-cpp-python.
    Provides methods for loading models and generating completions or chat responses.
    """
    model_name: str
    model_manager: ModelManager
    model_path: Optional[str]
    llm: Optional[Llama]

    def __init__(self, model_name: str) -> None:
        """
        Initialize the LLM interface.
        Args:
            model_name (str): Name of the model to load.
        Raises:
            ValueError: If the model is not found locally.
        """
        self.model_name = model_name
        self.model_manager = ModelManager()
        self.model_path = self.model_manager.get_model_path(model_name)
        if not self.model_path:
            raise ValueError(f"Model {model_name} not found. Please download it first.")
        self.llm = None

    def load_model(
        self,
        n_gpu_layers: Optional[int] = None,
        n_ctx: Optional[int] = None,
        verbose: bool = False,
        n_threads: Optional[int] = None,
        n_batch: Optional[int] = None,
        use_mlock: bool = False,
        use_mmap: bool = True,
        rope_freq_base: Optional[float] = None,
        rope_freq_scale: Optional[float] = None,
        low_vram: bool = False,
    ) -> None:
        """
        Load the model into memory.
        Args:
            n_gpu_layers (Optional[int]): Number of layers to offload to GPU (-1 for all).
            n_ctx (Optional[int]): Context size.
            verbose (bool): Whether to show verbose output.
            n_threads (Optional[int]): Number of threads to use.
            n_batch (Optional[int]): Batch size for prompt processing.
            use_mlock (bool): Whether to use mlock to keep model in memory.
            use_mmap (bool): Whether to use memory mapping for the model.
            rope_freq_base (Optional[float]): RoPE base frequency.
            rope_freq_scale (Optional[float]): RoPE frequency scaling factor.
            low_vram (bool): Whether to optimize for low VRAM usage.
        Raises:
            ValueError: If model loading fails.
        """
        # If model is already loaded, check if we need to reload with different parameters
        if self.llm is not None:
            if n_ctx is not None and hasattr(self.llm, 'n_ctx') and self.llm.n_ctx != n_ctx:
                # Need to reload with new context size
                self.llm = None
            else:
                # Model already loaded with compatible parameters
                return

        if n_gpu_layers is None:
            n_gpu_layers = config.get("default_gpu_layers", -1)
        if n_ctx is None:
            n_ctx = config.get("default_context_length", 4096)

        # Determine number of threads if not specified
        if n_threads is None:
            import multiprocessing
            n_threads = max(1, multiprocessing.cpu_count() // 2)

        console.print(f"[bold blue]Loading model {self.model_name}...[/bold blue]")
        try:
            # Initialize local RoPE variables to potentially override if found
            local_rope_freq_base = rope_freq_base
            local_rope_freq_scale = rope_freq_scale

            # Try to extract RoPE parameters from the model file if not provided
            if local_rope_freq_base is None or local_rope_freq_scale is None:
                try:
                    info = simple_gguf_info(self.model_path)
                    metadata = info.get("metadata", {}) # Use metadata for more reliable keys

                    # Check for rope_freq_base using common keys
                    base_val = metadata.get("llama.rope.freq_base") or metadata.get("rope_freq_base") or metadata.get("rope.freq_base")
                    if base_val is not None and local_rope_freq_base is None:
                        try:
                            local_rope_freq_base = float(base_val)
                            console.print(f"[dim]Detected RoPE frequency base: {local_rope_freq_base}[/dim]")
                        except (ValueError, TypeError):
                             console.print(f"[yellow]Warning: Could not convert detected rope_freq_base '{base_val}' to float.[/yellow]")


                    # Check for rope_freq_scale using common keys
                    scale_val = metadata.get("llama.rope.scale") or metadata.get("rope_freq_scale") or metadata.get("rope.scale") or metadata.get("rope.freq_scale")
                    if scale_val is not None and local_rope_freq_scale is None:
                         try:
                            local_rope_freq_scale = float(scale_val)
                            console.print(f"[dim]Detected RoPE frequency scale: {local_rope_freq_scale}[/dim]")
                         except (ValueError, TypeError) as e:
                             # This is the specific error we are fixing
                             console.print(f"[yellow]Warning: Could not convert detected rope_freq_scale '{scale_val}' to float: {e}. Skipping parameter.[/yellow]")
                             local_rope_freq_scale = None # Ensure it remains None if conversion fails

                except Exception as e:
                    console.print(f"[dim]Could not extract RoPE parameters: {str(e)}[/dim]")

            # Create a dictionary of parameters, only including non-None values
            params = {
                "model_path": self.model_path,
                "n_gpu_layers": n_gpu_layers,
                "n_ctx": n_ctx or 4096,  # Use default context if None
                "verbose": verbose,
                "n_threads": n_threads,
                "n_batch": n_batch or 512,
                "use_mlock": use_mlock,
                "use_mmap": use_mmap,
                "low_vram": low_vram,
            }

            # Only add rope parameters if they are valid floats
            if local_rope_freq_base is not None:
                params["rope_freq_base"] = local_rope_freq_base
            if local_rope_freq_scale is not None:
                params["rope_freq_scale"] = local_rope_freq_scale

            self.llm = Llama(**params)

            console.print(f"[bold green]Model {self.model_name} loaded successfully[/bold green]")
            if verbose:
                console.print(f"[dim]Using {n_threads} threads, context size: {n_ctx}[/dim]")
                if n_gpu_layers and n_gpu_layers > 0:
                    console.print(f"[dim]GPU acceleration: {n_gpu_layers} layers offloaded to GPU[/dim]")
        except Exception as e:
            # Add more context to the error message
            import traceback
            error_details = traceback.format_exc()
            raise ValueError(f"Failed to load model from file: {self.model_path}\nError: {str(e)}\nDetails:\n{error_details}")

    def create_completion(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        suffix: Optional[str] = None,
        images: Optional[List[str]] = None,
        system: Optional[str] = None,
        template: Optional[str] = None,
        context: Optional[List[int]] = None,
        raw: bool = False,
        format: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Create a completion for the given prompt.
        Args:
            prompt (str): The prompt to complete.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling.
            stream (bool): Whether to stream the response.
            stop (Optional[List[str]]): List of strings to stop generation when encountered.
        Returns:
            Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]: Completion result or generator for streaming.
        """
        if self.llm is None:
            self.load_model()
        if stream:
            return self.llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                stop=stop or [],
            )
        else:
            return self.llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False,
                stop=stop or [],
            )

    def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        format: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Create a chat completion for the given messages.
        Args:
            messages (List[Dict[str, str]]): List of chat messages.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling.
            stream (bool): Whether to stream the response.
            stop (Optional[List[str]]): List of strings to stop generation when encountered.
        Returns:
            Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]: Chat completion result or generator for streaming.
        """
        if self.llm is None:
            self.load_model()
        processed_messages: List[Dict[str, str]] = messages.copy()
        system_messages = [m for m in processed_messages if m.get("role") == "system"]
        non_system_messages = [m for m in processed_messages if m.get("role") != "system"]
        if system_messages:
            processed_messages = [system_messages[0]] + non_system_messages
        else:
            processed_messages = non_system_messages
        if stream:
            return self.llm.create_chat_completion(
                messages=processed_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                stop=stop or [],
            )
        else:
            return self.llm.create_chat_completion(
                messages=processed_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False,
                stop=stop or [],
            )

    def stream_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        callback: Callable[[str], None],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        format: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Stream a chat completion with a callback for each token.
        Args:
            messages (List[Dict[str, Any]]): List of chat messages.
            callback (Callable[[str], None]): Function to call with each token.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling.
            stop (Optional[List[str]]): List of strings to stop generation when encountered.
            tools (Optional[List[Dict[str, Any]]]): List of tools for function calling.
            format (Optional[Union[str, Dict[str, Any]]]): Format for structured output.
        """
        stream = self.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            stop=stop,
        )
        for chunk in stream:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                if "delta" in chunk["choices"][0] and "content" in chunk["choices"][0]["delta"]:
                    content = chunk["choices"][0]["delta"]["content"]
                    callback(content)

    def create_embeddings(
        self,
        input: Union[str, List[str]],
        truncate: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate embeddings for the given input.
        Args:
            input (Union[str, List[str]]): Text or list of texts to generate embeddings for.
            truncate (bool): Whether to truncate the input to fit within context length.
        Returns:
            Dict[str, Any]: Embeddings response.
        """
        if self.llm is None:
            self.load_model()

        # Convert input to list if it's a string
        if isinstance(input, str):
            input_texts = [input]
        else:
            input_texts = input

        # Generate embeddings for each input text
        embeddings = []
        for text in input_texts:
            # Use llama-cpp-python's embedding method
            embedding = self.llm.embed(text)
            embeddings.append(embedding)

        # Create response
        response = {
            "model": self.model_name,
            "embeddings": embeddings,
            "total_duration": 0,  # Could be improved with actual timing
            "load_duration": 0,   # Could be improved with actual timing
            "prompt_eval_count": len(input_texts)
        }

        return response
