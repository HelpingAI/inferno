from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import os
import torch

from inferno.utils.logger import get_logger
from inferno.utils.device import AUTO, CPU, CUDA, MPS, XLA

logger = get_logger(__name__)


@dataclass
class ServerConfig:
    """
    Configuration for the Inferno server.
    """
    # Model configuration
    model_name_or_path: str = "HelpingAI/HelpingAI-15B"
    model_revision: Optional[str] = None
    tokenizer_name_or_path: Optional[str] = None
    tokenizer_revision: Optional[str] = None

    # Hardware configuration
    device: str = AUTO
    device_map: str = "auto"
    cuda_device_idx: int = 0
    dtype: str = "float16"
    load_8bit: bool = False
    load_4bit: bool = False
    use_tpu: bool = False
    force_tpu: bool = False  # Force TPU usage even if not detected automatically
    tpu_cores: int = 8
    tpu_memory_limit: str = "90GB"

    # GGUF configuration
    enable_gguf: bool = False
    gguf_path: Optional[str] = None
    download_gguf: bool = False
    gguf_filename: Optional[str] = None
    num_gpu_layers: int = -1  # -1 means all layers
    context_size: int = 4096  # Default context size (4K tokens)
    chat_format: Optional[str] = None  # Chat format for GGUF models (llama-2, mistral, gemma, phi, etc.)

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    api_keys: List[str] = field(default_factory=list)
    max_concurrent_requests: int = 10
    max_queue_size: int = 100
    request_timeout: int = 60
    log_level: str = "info"
    log_file: Optional[str] = None

    # Additional models
    additional_models: List[str] = field(default_factory=list)

    def __post_init__(self):
        """
        Validate and normalize configuration after initialization.
        """
        # Normalize device string
        self.device = self.device.lower()

        # Set tokenizer to model path if not specified
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path

        # Validate device
        valid_devices = [AUTO, CPU, CUDA, MPS, XLA]
        if self.device not in valid_devices:
            logger.warning(f"Invalid device '{self.device}'. Using 'auto' instead.")
            self.device = AUTO

        # Validate dtype
        valid_dtypes = ["float16", "float32", "bfloat16"]
        if self.dtype not in valid_dtypes:
            logger.warning(f"Invalid dtype '{self.dtype}'. Using 'float16' instead.")
            self.dtype = "float16"

        # Normalize TPU memory limit
        if not self.tpu_memory_limit.upper().endswith("GB") and not self.tpu_memory_limit.upper().endswith("MB"):
            logger.warning(f"Invalid TPU memory limit format '{self.tpu_memory_limit}'. Using '90GB' instead.")
            self.tpu_memory_limit = "90GB"

        # Validate GGUF configuration
        if self.enable_gguf and self.gguf_path is None and not self.download_gguf:
            logger.warning("GGUF is enabled but no path is provided and download is disabled. GGUF will not be used.")
            self.enable_gguf = False

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return {
            "model": {
                "name_or_path": self.model_name_or_path,
                "revision": self.model_revision,
                "tokenizer": self.tokenizer_name_or_path,
                "tokenizer_revision": self.tokenizer_revision,
                "dtype": self.dtype,
                "quantization": {
                    "8bit": self.load_8bit,
                    "4bit": self.load_4bit
                },
                "gguf": {
                    "enabled": self.enable_gguf,
                    "path": self.gguf_path,
                    "download": self.download_gguf,
                    "filename": self.gguf_filename,
                    "num_gpu_layers": self.num_gpu_layers,
                    "context_size": self.context_size
                }
            },
            "hardware": {
                "device": self.device,
                "device_map": self.device_map,
                "cuda_device_idx": self.cuda_device_idx,
                "use_tpu": self.use_tpu,
                "tpu_cores": self.tpu_cores,
                "tpu_memory_limit": self.tpu_memory_limit
            },
            "server": {
                "host": self.host,
                "port": self.port,
                "api_keys_enabled": len(self.api_keys) > 0,
                "max_concurrent_requests": self.max_concurrent_requests,
                "max_queue_size": self.max_queue_size,
                "request_timeout": self.request_timeout,
                "log_level": self.log_level
            },
            "additional_models": self.additional_models
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ServerConfig':
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary representation of the configuration

        Returns:
            ServerConfig instance
        """
        # Extract model configuration
        model_config = config_dict.get("model", {})
        model_name_or_path = model_config.get("name_or_path", "HelpingAI/HelpingAI-15B")
        model_revision = model_config.get("revision")
        tokenizer_name_or_path = model_config.get("tokenizer")
        tokenizer_revision = model_config.get("tokenizer_revision")
        dtype = model_config.get("dtype", "float16")

        # Extract quantization configuration
        quantization = model_config.get("quantization", {})
        load_8bit = quantization.get("8bit", False)
        load_4bit = quantization.get("4bit", False)

        # Extract GGUF configuration
        gguf = config_dict.get("gguf", {})
        enable_gguf = gguf.get("enable_gguf", False)
        gguf_path = gguf.get("gguf_path")
        download_gguf = gguf.get("download_gguf", False)
        gguf_filename = gguf.get("gguf_filename")
        num_gpu_layers = gguf.get("num_gpu_layers", -1)
        context_size = gguf.get("context_size", 4096)
        chat_format = gguf.get("chat_format")

        # Extract hardware configuration
        hardware = config_dict.get("hardware", {})
        device = hardware.get("device", AUTO)
        device_map = hardware.get("device_map", "auto")
        cuda_device_idx = hardware.get("cuda_device_idx", 0)

        # Extract TPU configuration
        tpu_config = config_dict.get("tpu", {})
        use_tpu = tpu_config.get("use_tpu", False)
        force_tpu = tpu_config.get("force_tpu", False)
        tpu_cores = tpu_config.get("tpu_cores", 8)
        tpu_memory_limit = tpu_config.get("tpu_memory_limit", "90GB")

        # Extract server configuration
        server = config_dict.get("server", {})
        host = server.get("host", "0.0.0.0")
        port = server.get("port", 8000)
        api_keys = server.get("api_keys", [])
        max_concurrent_requests = server.get("max_concurrent_requests", 10)
        max_queue_size = server.get("max_queue_size", 100)
        request_timeout = server.get("request_timeout", 60)
        log_level = server.get("log_level", "info")
        log_file = server.get("log_file")

        # Extract additional models
        additional_models = config_dict.get("additional_models", [])

        return cls(
            model_name_or_path=model_name_or_path,
            model_revision=model_revision,
            tokenizer_name_or_path=tokenizer_name_or_path,
            tokenizer_revision=tokenizer_revision,
            device=device,
            device_map=device_map,
            cuda_device_idx=cuda_device_idx,
            dtype=dtype,
            load_8bit=load_8bit,
            load_4bit=load_4bit,
            use_tpu=use_tpu,
            force_tpu=force_tpu,
            tpu_cores=tpu_cores,
            tpu_memory_limit=tpu_memory_limit,
            enable_gguf=enable_gguf,
            gguf_path=gguf_path,
            download_gguf=download_gguf,
            gguf_filename=gguf_filename,
            num_gpu_layers=num_gpu_layers,
            context_size=context_size,
            chat_format=chat_format,
            host=host,
            port=port,
            api_keys=api_keys,
            max_concurrent_requests=max_concurrent_requests,
            max_queue_size=max_queue_size,
            request_timeout=request_timeout,
            log_level=log_level,
            log_file=log_file,
            additional_models=additional_models
        )

    @classmethod
    def from_args(cls, args: Any) -> 'ServerConfig':
        """
        Create configuration from command line arguments.

        Args:
            args: Command line arguments

        Returns:
            ServerConfig instance
        """
        # Process API keys (convert comma-separated string to list)
        api_keys = []
        if hasattr(args, 'api_keys') and args.api_keys:
            api_keys = [key.strip() for key in args.api_keys.split(',')]

        # Process additional models
        additional_models = []
        if hasattr(args, 'additional_models') and args.additional_models:
            additional_models = args.additional_models

        return cls(
            model_name_or_path=args.model,
            model_revision=args.model_revision if hasattr(args, 'model_revision') else None,
            tokenizer_name_or_path=args.tokenizer if hasattr(args, 'tokenizer') else None,
            tokenizer_revision=args.tokenizer_revision if hasattr(args, 'tokenizer_revision') else None,
            device=args.device if hasattr(args, 'device') else AUTO,
            device_map=args.device_map if hasattr(args, 'device_map') else "auto",
            cuda_device_idx=args.cuda_device_idx if hasattr(args, 'cuda_device_idx') else 0,
            dtype=args.dtype if hasattr(args, 'dtype') else "float16",
            load_8bit=args.load_8bit if hasattr(args, 'load_8bit') else False,
            load_4bit=args.load_4bit if hasattr(args, 'load_4bit') else False,
            use_tpu=args.use_tpu if hasattr(args, 'use_tpu') else False,
            force_tpu=args.force_tpu if hasattr(args, 'force_tpu') else False,
            tpu_cores=args.tpu_cores if hasattr(args, 'tpu_cores') else 8,
            tpu_memory_limit=args.tpu_memory_limit if hasattr(args, 'tpu_memory_limit') else "90GB",
            enable_gguf=args.enable_gguf if hasattr(args, 'enable_gguf') else False,
            gguf_path=args.gguf_path if hasattr(args, 'gguf_path') else None,
            download_gguf=args.download_gguf if hasattr(args, 'download_gguf') else False,
            gguf_filename=args.gguf_filename if hasattr(args, 'gguf_filename') else None,
            num_gpu_layers=args.num_gpu_layers if hasattr(args, 'num_gpu_layers') else -1,
            context_size=args.context_size if hasattr(args, 'context_size') else 4096,
            chat_format=args.chat_format if hasattr(args, 'chat_format') else None,
            host=args.host if hasattr(args, 'host') else "0.0.0.0",
            port=args.port if hasattr(args, 'port') else 8000,
            api_keys=api_keys,
            max_concurrent_requests=args.max_concurrent if hasattr(args, 'max_concurrent') else 10,
            max_queue_size=args.max_queue if hasattr(args, 'max_queue') else 100,
            request_timeout=args.timeout if hasattr(args, 'timeout') else 60,
            log_level=args.log_level if hasattr(args, 'log_level') else "info",
            log_file=args.log_file if hasattr(args, 'log_file') else None,
            additional_models=additional_models
        )