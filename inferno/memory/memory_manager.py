import os
import torch
import psutil
from typing import Tuple, Dict, Optional

from inferno.utils.logger import get_logger
from inferno.utils.device import CPU, CUDA, MPS, XLA, get_optimal_device

logger = get_logger(__name__)


class MemoryManager:
    """
    Memory management for Inferno server.
    Handles memory detection and allocation across different device types.
    """

    def __init__(self, device: str = "auto", cuda_device_idx: Optional[int] = None):
        """
        Initialize the memory manager.

        Args:
            device: Device type (auto, cuda, cpu, mps, xla)
            cuda_device_idx: CUDA device index to use
        """
        self.device = device
        if self.device == "auto":
            self.device = get_optimal_device()

        self.cuda_device_idx = cuda_device_idx
        if self.device == CUDA and cuda_device_idx is None:
            self.cuda_device_idx = 0

    def detect_available_memory(self) -> int:
        """
        Auto-detect the total available memory for the current device.

        Returns:
            Total available memory in bytes
        """
        device = self.device

        # For TPU devices
        if device == XLA:
            return self._detect_tpu_memory()

        # For CUDA (NVIDIA GPU) devices
        elif device == CUDA:
            return self._detect_cuda_memory()

        # For Apple Silicon (MPS) devices
        elif device == MPS:
            return self._detect_mps_memory()

        # For CPU devices
        elif device == CPU:
            return self._detect_cpu_memory()

        # Default fallback for unknown devices
        logger.warning(f"Unknown device type {device}, defaulting to 8GB memory")
        return 8 * 1000000000  # 8GB in bytes

    def _detect_tpu_memory(self) -> int:
        """
        Detect available TPU memory using multiple methods with fallbacks.

        Returns:
            Total TPU memory in bytes
        """
        try:
            # Try to get memory info from XLA
            import torch_xla.core.xla_model as xm #type: ignore[import]
            import torch_xla.utils.utils as xu #type: ignore[import]

            # Get device memory stats
            xla_device = xm.xla_device()
            memory_info = xu.get_memory_info(xla_device)

            if memory_info and 'kb_total' in memory_info:
                # Convert KB to bytes
                total_memory = memory_info['kb_total'] * 1024
                logger.info(f"Detected TPU memory from XLA: {total_memory / 1000000000:.2f}GB")
                return total_memory

            # Fallback method: try to read from TPU configuration
            try:
                import json

                # TPU configuration is often available in this file
                tpu_config_path = '/sys/class/tpu/tpu0/device/tpu'
                if os.path.exists(tpu_config_path):
                    with open(f"{tpu_config_path}/config", 'r') as f:
                        config = json.load(f)
                        if 'memory_size' in config:
                            total_memory = config['memory_size']  # Already in bytes
                            logger.info(f"Detected TPU memory from config: {total_memory / 1000000000:.2f}GB")
                            return total_memory
            except Exception as e:
                logger.warning(f"Failed to read TPU config: {e}")

            # Another fallback: try to parse from environment variables
            if 'TPU_HOST_BOUNDS' in os.environ:
                # Format is typically like "1,1,1" or "2,2,1" for different TPU sizes
                # We can estimate memory based on the TPU size
                bounds = os.environ['TPU_HOST_BOUNDS'].split(',')
                tpu_chips = 1
                for dim in bounds:
                    tpu_chips *= int(dim)

                # Rough estimate: v2 TPUs have ~8GB per chip, v3 have ~16GB, v4 have ~32GB
                # We'll assume v3 as a middle ground if we can't determine the version
                per_chip_memory = 16 * 1000000000  # 16GB per chip for v3

                # Try to determine TPU version
                if 'TPU_CHIP_VERSION' in os.environ:
                    version = os.environ['TPU_CHIP_VERSION']
                    if '2' in version:
                        per_chip_memory = 8 * 1000000000  # 8GB for v2
                    elif '4' in version:
                        per_chip_memory = 32 * 1000000000  # 32GB for v4

                total_memory = tpu_chips * per_chip_memory
                logger.info(f"Estimated TPU memory from chip count: {total_memory / 1000000000:.2f}GB")
                return total_memory

            # Default fallback for TPU: assume 105GB (common for TPU v3-8)
            logger.warning("Could not detect TPU memory, defaulting to 105GB")
            return 105 * 1000000000  # 105GB in bytes
        except Exception as e:
            logger.warning(f"Error detecting TPU memory: {e}")
            # Default fallback for TPU
            return 105 * 1000000000  # 105GB in bytes

    def _detect_cuda_memory(self) -> int:
        """
        Detect available CUDA (GPU) memory.

        Returns:
            Total CUDA memory in bytes
        """
        try:
            # Get total GPU memory
            device_idx = 0  # Default to first GPU
            if hasattr(self, 'cuda_device_idx') and self.cuda_device_idx is not None:
                device_idx = self.cuda_device_idx

            # Make sure the device index is valid
            if device_idx >= torch.cuda.device_count():
                device_idx = 0

            # Get total memory for this GPU
            total_memory = torch.cuda.get_device_properties(device_idx).total_memory
            logger.info(f"Detected CUDA memory: {total_memory / 1000000000:.2f}GB")
            return total_memory
        except Exception as e:
            logger.warning(f"Error detecting CUDA memory: {e}")
            # Fallback: assume 8GB for CUDA devices
            return 8 * 1000000000  # 8GB in bytes

    def _detect_mps_memory(self) -> int:
        """
        Detect available MPS (Apple Silicon) memory.

        Returns:
            Estimated MPS memory in bytes
        """
        try:
            # For Apple Silicon, we need to use system APIs
            # This is a rough estimate based on system memory
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
            if result.returncode == 0:
                system_memory = int(result.stdout.strip())
                # Apple Silicon typically shares memory between CPU and GPU
                # We'll assume 70% of system memory is available for ML tasks
                total_memory = int(system_memory * 0.7)
                logger.info(f"Estimated MPS memory: {total_memory / 1000000000:.2f}GB (70% of system memory)")
                return total_memory
        except Exception as e:
            logger.warning(f"Error detecting MPS memory: {e}")

        # Fallback: assume 8GB for MPS devices (common for M1/M2 Macs)
        return 8 * 1000000000  # 8GB in bytes

    def _detect_cpu_memory(self) -> int:
        """
        Detect available CPU memory.

        Returns:
            Estimated CPU memory in bytes
        """
        try:
            # Get total system memory
            system_memory = psutil.virtual_memory().total
            # For CPU, we'll use 80% of system memory as available for ML tasks
            total_memory = int(system_memory * 0.8)
            logger.info(f"Detected CPU memory: {total_memory / 1000000000:.2f}GB (80% of system memory)")
            return total_memory
        except Exception as e:
            logger.warning(f"Error detecting CPU memory: {e}")
            # Fallback: assume 8GB for CPU
            return 8 * 1000000000  # 8GB in bytes

    def get_memory_allocation(self, model_count: int = 1) -> Tuple[int, str]:
        """
        Calculate memory allocation for models based on device type and model count.

        Args:
            model_count: Number of models to allocate memory for

        Returns:
            Tuple of (per_model_memory_bytes, per_model_memory_gb_string)
        """
        # Get total available memory
        total_memory = self.detect_available_memory()

        # Calculate per-model memory
        per_model_memory = total_memory // model_count

        # Convert to GB string format for display
        per_model_memory_gb = f"{per_model_memory / 1000000000:.2f}GB"

        # Log memory allocation details
        logger.info(f"Device: {self.device}")
        logger.info(f"Total memory: {total_memory / 1000000000:.2f}GB")
        logger.info(f"Per-model memory allocation: {per_model_memory_gb} ({model_count} models)")

        return per_model_memory, per_model_memory_gb

    def set_environment_variables(self, memory_bytes: int) -> None:
        """
        Set device-specific environment variables for memory management.

        Args:
            memory_bytes: Memory allocation in bytes
        """
        if self.device == XLA:
            # Set TPU-specific environment variables
            os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = str(memory_bytes)
            logger.info(f"Set XLA_TENSOR_ALLOCATOR_MAXSIZE to {memory_bytes / 1000000000:.2f}GB")

        # Add other device-specific environment variables as needed