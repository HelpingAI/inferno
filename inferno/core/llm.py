"""
LLM interface for Inferno using llama-cpp-python
"""

from typing import Dict, Any, List, Optional, Union, Generator, Callable
import time

# Only raise the import error when the module is actually used, not when imported
# This allows the package to be installed without llama-cpp-python
_llama_import_error = None
try:
    from llama_cpp import Llama
except ImportError as e:
    _llama_import_error = ImportError(
        "llama-cpp-python is not installed. "
        "Please install it with hardware acceleration support *before* installing inferno. "
        "See the 'Hardware Acceleration (llama-cpp-python)' section in README.md for instructions. "
        "Example: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
    )

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
        # New parameters from latest llama-cpp-python
        split_mode: Optional[int] = None,
        main_gpu: int = 0,
        tensor_split: Optional[List[float]] = None,
        rpc_servers: Optional[str] = None,
        vocab_only: bool = False,
        kv_overrides: Optional[Dict[str, Union[bool, int, float, str]]] = None,
        seed: Optional[int] = None,
        n_ubatch: Optional[int] = None,
        n_threads_batch: Optional[int] = None,
        rope_scaling_type: Optional[int] = None,
        pooling_type: Optional[int] = None,
        yarn_ext_factor: Optional[float] = None,
        yarn_attn_factor: Optional[float] = None,
        yarn_beta_fast: Optional[float] = None,
        yarn_beta_slow: Optional[float] = None,
        yarn_orig_ctx: Optional[int] = None,
        logits_all: bool = False,
        embedding: bool = False,
        offload_kqv: bool = True,
        flash_attn: bool = False,
        no_perf: bool = False,
        last_n_tokens_size: Optional[int] = None,
        lora_base: Optional[str] = None,
        lora_path: Optional[str] = None,
        lora_scale: Optional[float] = None,
        numa: Union[bool, int] = False,
        chat_format: Optional[str] = None,
        type_k: Optional[int] = None,
        type_v: Optional[int] = None,
        spm_infill: bool = False,
    ) -> None:
        """
        Load the model into memory with support for all llama-cpp-python parameters.
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
            split_mode (Optional[int]): How to split the model across GPUs.
            main_gpu (int): Main GPU to use based on split mode.
            tensor_split (Optional[List[float]]): How to distribute tensors across GPUs.
            rpc_servers (Optional[str]): Comma separated list of RPC servers.
            vocab_only (bool): Only load the vocabulary, not weights.
            kv_overrides (Optional[Dict]): Key-value overrides for the model.
            seed (Optional[int]): RNG seed, None for random.
            n_ubatch (Optional[int]): Physical batch size.
            n_threads_batch (Optional[int]): Threads for batch processing.
            rope_scaling_type (Optional[int]): RoPE scaling type.
            pooling_type (Optional[int]): Pooling type for embeddings.
            yarn_ext_factor (Optional[float]): YaRN extrapolation mix factor.
            yarn_attn_factor (Optional[float]): YaRN magnitude scaling factor.
            yarn_beta_fast (Optional[float]): YaRN low correction dim.
            yarn_beta_slow (Optional[float]): YaRN high correction dim.
            yarn_orig_ctx (Optional[int]): YaRN original context size.
            logits_all (bool): Return logits for all tokens.
            embedding (bool): Enable embedding mode.
            offload_kqv (bool): Offload K, Q, V to GPU.
            flash_attn (bool): Use flash attention.
            no_perf (bool): Disable performance measurements.
            last_n_tokens_size (Optional[int]): Max tokens in last_n_tokens deque.
            lora_base (Optional[str]): Path to base model for LoRA.
            lora_path (Optional[str]): Path to LoRA file.
            lora_scale (Optional[float]): Scale for LoRA adaptations.
            numa (Union[bool, int]): NUMA policy.
            chat_format (Optional[str]): Chat format for chat completions.
            type_k (Optional[int]): KV cache data type for K.
            type_v (Optional[int]): KV cache data type for V.
            spm_infill (bool): Use SPM pattern for infill.
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
                "embedding": embedding,  # Enable embeddings by default
                "offload_kqv": offload_kqv,
                "flash_attn": flash_attn,
                "main_gpu": main_gpu,
                "vocab_only": vocab_only,
                "logits_all": logits_all,
                "numa": numa,
                "spm_infill": spm_infill,
            }

            # Add optional parameters only if they're not None
            if local_rope_freq_base is not None:
                params["rope_freq_base"] = local_rope_freq_base
            if local_rope_freq_scale is not None:
                params["rope_freq_scale"] = local_rope_freq_scale
            if split_mode is not None:
                params["split_mode"] = split_mode
            if tensor_split is not None:
                params["tensor_split"] = tensor_split
            if rpc_servers is not None:
                params["rpc_servers"] = rpc_servers
            if kv_overrides is not None:
                params["kv_overrides"] = kv_overrides
            if seed is not None:
                params["seed"] = seed
            if n_ubatch is not None:
                params["n_ubatch"] = n_ubatch
            if n_threads_batch is not None:
                params["n_threads_batch"] = n_threads_batch
            if rope_scaling_type is not None:
                params["rope_scaling_type"] = rope_scaling_type
            if pooling_type is not None:
                params["pooling_type"] = pooling_type
            if yarn_ext_factor is not None:
                params["yarn_ext_factor"] = yarn_ext_factor
            if yarn_attn_factor is not None:
                params["yarn_attn_factor"] = yarn_attn_factor
            if yarn_beta_fast is not None:
                params["yarn_beta_fast"] = yarn_beta_fast
            if yarn_beta_slow is not None:
                params["yarn_beta_slow"] = yarn_beta_slow
            if yarn_orig_ctx is not None:
                params["yarn_orig_ctx"] = yarn_orig_ctx
            if last_n_tokens_size is not None:
                params["last_n_tokens_size"] = last_n_tokens_size
            if lora_base is not None:
                params["lora_base"] = lora_base
            if lora_path is not None:
                params["lora_path"] = lora_path
            if lora_scale is not None:
                params["lora_scale"] = lora_scale
            if chat_format is not None:
                params["chat_format"] = chat_format
            if type_k is not None:
                params["type_k"] = type_k
            if type_v is not None:
                params["type_v"] = type_v

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
        max_tokens: Optional[int] = 16,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        suffix: Optional[str] = None,
        echo: bool = False,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        grammar: Optional[Any] = None,
        logit_bias: Optional[Dict[int, float]] = None,
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Create a completion for the given prompt.
        Args:
            prompt (str): The prompt to complete.
            max_tokens (Optional[int]): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling.
            min_p (float): Minimum probability threshold for token selection.
            top_k (int): Top-k sampling.
            repeat_penalty (float): Penalty for repeating tokens.
            stream (bool): Whether to stream the response.
            stop (Optional[List[str]]): List of strings to stop generation when encountered.
            seed (Optional[int]): Random seed for reproducible generation.
            suffix (Optional[str]): String to append to the generated text.
            echo (bool): Whether to include the prompt in the response.
            frequency_penalty (float): Penalty for token frequency.
            presence_penalty (float): Penalty for token presence.
            grammar (Optional[Any]): Grammar for constrained generation.
            logit_bias (Optional[Dict[int, float]]): Token bias for generation.
        Returns:
            Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]: Completion result or generator for streaming.
        """
        if _llama_import_error:
            raise _llama_import_error
            
        if self.llm is None:
            self.load_model()
            
        # Prepare parameters
        params = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "min_p": min_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "stream": stream,
            "stop": stop or [],
            "echo": echo,
        }
        
        # Add optional parameters only if they're provided and not default
        if seed is not None:
            params["seed"] = seed
        if suffix is not None:
            params["suffix"] = suffix
        if frequency_penalty != 0.0:
            params["frequency_penalty"] = frequency_penalty
        if presence_penalty != 0.0:
            params["presence_penalty"] = presence_penalty
        if grammar is not None:
            params["grammar"] = grammar
        if logit_bias is not None:
            params["logit_bias"] = logit_bias
            
        return self.llm.create_completion(**params)

    def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        format: Optional[Union[str, Dict[str, Any]]] = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        logit_bias: Optional[Dict[int, float]] = None,
        grammar: Optional[Any] = None,
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Create a chat completion for the given messages.
        Args:
            messages (List[Dict[str, Any]]): List of chat messages.
            max_tokens (Optional[int]): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling.
            top_k (int): Top-k sampling.
            min_p (float): Minimum probability threshold for token selection.
            stream (bool): Whether to stream the response.
            stop (Optional[List[str]]): List of strings to stop generation when encountered.
            seed (Optional[int]): Random seed for reproducible generation.
            tools (Optional[List[Dict[str, Any]]]): List of tools the model may call.
            tool_choice (Optional[Union[str, Dict[str, Any]]]): Controls which tool is called, if any.
            format (Optional[Union[str, Dict[str, Any]]]): Format for structured output (e.g., 'json').
            frequency_penalty (float): Penalty for token frequency.
            presence_penalty (float): Penalty for token presence.
            repeat_penalty (float): Penalty for repeating tokens.
            logit_bias (Optional[Dict[int, float]]): Token bias for generation.
            grammar (Optional[Any]): Grammar for constrained generation.
        Returns:
            Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]: Chat completion result or generator for streaming.
        """
        if _llama_import_error:
            raise _llama_import_error
            
        if self.llm is None:
            self.load_model()
            
        # Use messages directly without processing
        params = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "stream": stream,
            "stop": stop or [],
        }
        
        # Add optional parameters only if they're provided and not default
        if seed is not None:
            params["seed"] = seed
        if tools:
            params["tools"] = tools
        if tool_choice:
            params["tool_choice"] = tool_choice
        if format:
            params["response_format"] = format if isinstance(format, dict) else {"type": format}
        if frequency_penalty != 0.0:
            params["frequency_penalty"] = frequency_penalty
        if presence_penalty != 0.0:
            params["presence_penalty"] = presence_penalty
        if repeat_penalty != 1.1:
            params["repeat_penalty"] = repeat_penalty
        if logit_bias:
            params["logit_bias"] = logit_bias
        if grammar:
            params["grammar"] = grammar
            
        return self.llm.create_chat_completion(**params)


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

        # Track timing
        start_time = time.time()

        # Generate embeddings for each input text
        embeddings = []
        total_tokens = 0
        
        for text in input_texts:
            # Use llama-cpp-python's embed method with the latest parameters
            # The embed method now supports normalize, truncate, and return_count
            try:
                # Try with return_count to get token count
                embedding, tokens = self.llm.embed(
                    text, 
                    normalize=True,
                    truncate=truncate,
                    return_count=True
                )
                total_tokens += tokens
            except TypeError:
                # Fallback for older versions that don't support return_count
                embedding = self.llm.embed(
                    text,
                    normalize=True,
                    truncate=truncate
                )
            
            embeddings.append(embedding)

        # Calculate durations
        end_time = time.time()
        total_duration = int((end_time - start_time) * 1e9)  # Convert to nanoseconds

        # Create response
        response = {
            "model": self.model_name,
            "embeddings": embeddings,
            "total_duration": total_duration,
            "load_duration": 0,  # We don't track load time separately
            "prompt_eval_count": total_tokens if total_tokens > 0 else len(input_texts)
        }

        return response


