import os
import torch
from typing import Dict, Any, Optional, Tuple

from inferno.utils.logger import get_logger

logger = get_logger(__name__)

# Constants for device types
CPU = "cpu"
CUDA = "cuda"
MPS = "mps"  # Apple Silicon
XLA = "xla"  # TPU
AUTO = "auto"

# Feature detection flags
CPUINFO_AVAILABLE = False
BNB_AVAILABLE = False
XLA_AVAILABLE = False
MPS_AVAILABLE = False

# Try to import optional dependencies
try:
    import py_cpuinfo # type: ignore[import]
    CPUINFO_AVAILABLE = True
except ImportError:
    pass

try:
    import bitsandbytes # type: ignore[import]
    BNB_AVAILABLE = True
except ImportError:
    pass

# Check for TPU availability
try:
    # First check if libtpu.so exists, which is a more reliable indicator
    import os
    if os.path.exists('/usr/lib/libtpu.so') or os.path.exists('/lib/libtpu.so'):
        # Set TPU environment variables early
        os.environ["PJRT_DEVICE"] = "TPU"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        logger.info("TPU library detected, setting PJRT_DEVICE=TPU")

        # Now try to import torch_xla
        try:
            import torch_xla # type: ignore[import]
            import torch_xla.core.xla_model as xm # type: ignore[import]
            # Verify TPU is actually available by trying to get devices
            try:
                devices = xm.get_xla_supported_devices()
                if devices:
                    XLA_AVAILABLE = True
                    logger.info(f"TPU is available with {len(devices)} devices")
                else:
                    logger.warning("No TPU devices found despite libtpu.so being present")
            except Exception as e:
                logger.warning(f"Error initializing TPU: {e}")
        except ImportError as e:
            logger.warning(f"TPU library detected but torch_xla import failed: {e}")
            logger.warning("Install with: pip install torch_xla")
    else:
        # If no libtpu.so, still try torch_xla as a fallback
        try:
            import torch_xla # type: ignore[import]
            import torch_xla.core.xla_model as xm # type: ignore[import]
            devices = xm.get_xla_supported_devices()
            if devices:
                XLA_AVAILABLE = True
                logger.info(f"TPU is available with {len(devices)} devices")
        except (ImportError, Exception):
            pass
except Exception as e:
    logger.warning(f"Error during TPU detection: {e}")

# Check for MPS (Apple Silicon) support
if hasattr(torch.backends, "mps") and torch.backends.mps.is_built():
    try:
        # Further verify by trying to create a tensor and checking if MPS is available
        if torch.backends.mps.is_available():
            torch.zeros(1).to("mps")
            MPS_AVAILABLE = True
    except Exception:
        pass


def get_available_devices() -> Dict[str, bool]:
    """
    Get a dictionary of available devices.

    Returns:
        Dictionary with device types as keys and availability as values
    """
    return {
        CPU: True,  # CPU is always available
        CUDA: torch.cuda.is_available(),
        MPS: MPS_AVAILABLE,
        XLA: XLA_AVAILABLE
    }


def get_optimal_device() -> str:
    """
    Determine the optimal device to use based on what's available.
    Prioritizes: CUDA > XLA > MPS > CPU

    Returns:
        Device type string
    """
    devices = get_available_devices()

    if devices[CUDA]:
        return CUDA
    elif devices[XLA]:
        return XLA
    elif devices[MPS]:
        return MPS
    else:
        return CPU


def get_device_info(device_type: str = AUTO) -> Dict[str, Any]:
    """
    Get detailed information about the specified device.

    Args:
        device_type: Device type (auto, cuda, cpu, mps, xla)

    Returns:
        Dictionary with device information
    """
    if device_type == AUTO:
        device_type = get_optimal_device()

    info = {"type": device_type}

    if device_type == CUDA and torch.cuda.is_available():
        info["count"] = torch.cuda.device_count()
        info["current_device"] = torch.cuda.current_device()
        info["name"] = torch.cuda.get_device_name(info["current_device"])
        info["memory_total"] = torch.cuda.get_device_properties(info["current_device"]).total_memory
        info["memory_allocated"] = torch.cuda.memory_allocated()
        info["memory_reserved"] = torch.cuda.memory_reserved()

    elif device_type == CPU:
        import multiprocessing
        info["cores"] = multiprocessing.cpu_count()

        if CPUINFO_AVAILABLE:
            cpu_info = py_cpuinfo.get_cpu_info()
            info["name"] = cpu_info.get("brand_raw", "Unknown CPU")
            info["architecture"] = cpu_info.get("arch", "Unknown")
            info["bits"] = cpu_info.get("bits", 64)

    elif device_type == MPS and MPS_AVAILABLE:
        info["name"] = "Apple Silicon"
        try:
            # Try to get more specific information about the Apple Silicon chip
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                   capture_output=True, text=True)
            if result.returncode == 0:
                info["name"] = result.stdout.strip()
        except Exception:
            pass

    elif device_type == XLA and XLA_AVAILABLE:
        try:
            import torch_xla.core.xla_model as xm # type: ignore[import]
            devices = xm.get_xla_supported_devices()
            info["count"] = len(devices)
            info["devices"] = devices

            # Try to get TPU version from environment variables
            if "TPU_CHIP_VERSION" in os.environ:
                info["version"] = os.environ["TPU_CHIP_VERSION"]

            # Try to get TPU topology
            if "TPU_HOST_BOUNDS" in os.environ:
                bounds = os.environ["TPU_HOST_BOUNDS"].split(",")
                info["topology"] = "x".join(bounds)
        except Exception as e:
            logger.warning(f"Error getting XLA device info: {e}")

    return info


def setup_device(device_type: str = AUTO,
                cuda_device_idx: int = 0,
                use_tpu: bool = False,
                force_tpu: bool = False,
                tpu_cores: int = 8) -> Tuple[str, Optional[int]]:
    """
    Set up the specified device for use.

    Args:
        device_type: Device type (auto, cuda, cpu, mps, xla)
        cuda_device_idx: CUDA device index to use
        use_tpu: Whether to use TPU
        tpu_cores: Number of TPU cores to use

    Returns:
        Tuple of (device_type, cuda_device_idx)
    """
    # If TPU is forced, override device type
    if force_tpu:
        logger.info("TPU usage forced by configuration")
        device_type = XLA
        use_tpu = True
    # If auto, determine the best available device
    elif device_type == AUTO:
        device_type = get_optimal_device()

    # Log the selected device
    logger.info(f"Using device: {device_type}")

    # Device-specific setup
    if device_type == CUDA and torch.cuda.is_available():
        # Validate CUDA device index
        if cuda_device_idx >= torch.cuda.device_count():
            logger.warning(f"CUDA device index {cuda_device_idx} out of range. Using device 0.")
            cuda_device_idx = 0

        # Set CUDA device
        torch.cuda.set_device(cuda_device_idx)
        logger.info(f"Using CUDA device {cuda_device_idx}: {torch.cuda.get_device_name(cuda_device_idx)}")

    elif device_type == CPU:
        # Set number of threads for CPU inference
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        if hasattr(torch, 'set_num_threads'):
            torch.set_num_threads(cpu_count)
            logger.info(f"Set PyTorch to use {cpu_count} threads")

    elif device_type == MPS and MPS_AVAILABLE:
        logger.info("Using Apple Silicon (MPS) for acceleration")

    elif (device_type == XLA or use_tpu) and XLA_AVAILABLE:
        logger.info(f"Using TPU with {tpu_cores} cores")
        logger.info("TPU is available and will be used for model inference")

        # Set TPU-specific environment variables if not already set
        if "PJRT_DEVICE" not in os.environ:
            os.environ["PJRT_DEVICE"] = "TPU"
            logger.info("Set PJRT_DEVICE=TPU environment variable")
        if "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            logger.info("Set XLA_PYTHON_CLIENT_PREALLOCATE=false environment variable")

        # Enable bfloat16 for better performance
        os.environ['XLA_USE_BF16'] = '1'
        logger.info("Enabled bfloat16 precision for TPU (XLA_USE_BF16=1)")

        # Set TPU memory allocation strategy
        os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000000'  # ~100GB
        logger.info("Set TPU memory allocation strategy (XLA_TENSOR_ALLOCATOR_MAXSIZE=100GB)")

        # Try to initialize the TPU device to ensure it's working
        try:
            import torch_xla.core.xla_model as xm # type: ignore[import]
            device = xm.xla_device()
            logger.info(f"TPU device successfully initialized: {device}")

            # Get TPU device count for additional verification
            devices = xm.get_xla_supported_devices()
            logger.info(f"Found {len(devices)} TPU devices: {devices}")

            device_type = XLA
        except Exception as e:
            logger.error(f"Failed to initialize TPU device: {e}")
            logger.warning("Falling back to CPU")
            device_type = CPU
    else:
        # Fallback to CPU if the requested device is not available
        logger.warning(f"Requested device {device_type} not available. Falling back to CPU.")
        device_type = CPU

        # Set number of threads for CPU inference
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        if hasattr(torch, 'set_num_threads'):
            torch.set_num_threads(cpu_count)
            logger.info(f"Set PyTorch to use {cpu_count} threads")

    return device_type, cuda_device_idx