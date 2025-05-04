"""
GGUF file reader for Inferno

This module provides utilities for reading and extracting metadata from GGUF files.
"""

import os
import struct
import enum
import re # Import re module
from typing import Dict, Any, Optional, Union, Tuple
from io import BufferedReader


class GGUFValueType(enum.IntEnum):
    """GGUF value types (corrected to standard)"""
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6  # Corrected from 7
    BOOL = 7     # Corrected from 9
    STRING = 8   # Corrected from 10
    ARRAY = 9    # Corrected from 11
    UINT64 = 10  # Corrected from 6
    INT64 = 11   # Corrected from 8
    FLOAT64 = 12 # Correct


class GGUFReader:
    """
    A reader for GGUF files that can extract metadata and tensor information.

    This is a more comprehensive implementation than the previous extract_max_context_from_gguf
    function, allowing access to all metadata in the GGUF file.
    """

    # GGUF magic bytes for identification
    GGUF_MAGIC_BYTES = b'GGUF'

    # The GGUF format versions that this class supports
    SUPPORTED_GGUF_VERSIONS = [2, 3]

    # Arguments for struct.unpack() based on GGUF value type (corrected)
    value_packing = {
        GGUFValueType.UINT8: "<B",  # Use little-endian explicitly
        GGUFValueType.INT8: "<b",
        GGUFValueType.UINT16: "<H",
        GGUFValueType.INT16: "<h",
        GGUFValueType.UINT32: "<I",
        GGUFValueType.INT32: "<i",
        GGUFValueType.FLOAT32: "<f", # Type 6
        GGUFValueType.BOOL: "?",     # Type 7
        # STRING (Type 8) handled separately
        # ARRAY (Type 9) handled separately
        GGUFValueType.UINT64: "<Q", # Type 10
        GGUFValueType.INT64: "<q",  # Type 11
        GGUFValueType.FLOAT64: "<d", # Type 12
    }

    # Length in bytes for each GGUF value type (corrected)
    value_lengths = {
        GGUFValueType.UINT8: 1,
        GGUFValueType.INT8: 1,
        GGUFValueType.UINT16: 2,
        GGUFValueType.INT16: 2,
        GGUFValueType.UINT32: 4,
        GGUFValueType.INT32: 4,
        GGUFValueType.FLOAT32: 4, # Type 6
        GGUFValueType.BOOL: 1,    # Type 7
        # STRING (Type 8) handled separately
        # ARRAY (Type 9) handled separately
        GGUFValueType.UINT64: 8,  # Type 10
        GGUFValueType.INT64: 8,   # Type 11
        GGUFValueType.FLOAT64: 8, # Type 12
    }

    def __init__(self, model_path: str):
        """
        Initialize the GGUF reader with a model path.

        Args:
            model_path: Path to the GGUF file
        """
        self.model_path = model_path
        self.metadata = {}
        self.version = None
        self.tensor_count = None
        self.kv_count = None

    def _unpack(self, value_type: GGUFValueType, file: BufferedReader) -> Any:
        """
        Unpack a value from the file based on its type.

        Args:
            value_type: The GGUF value type
            file: The file to read from

        Returns:
            The unpacked value
        """
        return struct.unpack(
            self.value_packing.get(value_type),
            file.read(self.value_lengths.get(value_type))
        )[0]

    def _get_single_value(self, value_type: GGUFValueType, file: BufferedReader) -> Union[str, int, float, bool]:
        """
        Read a single value from an open file.

        Args:
            value_type: The GGUF value type
            file: The file to read from

        Returns:
            The read value
        """
        if value_type == GGUFValueType.STRING: # Type 8
            # Assuming V3+ format for strings within arrays if used by GGUFReader class
            string_length = struct.unpack("<Q", file.read(8))[0] # Use little-endian
            value = file.read(string_length)
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError:
                print(f"Warning: UnicodeDecodeError while reading a string from GGUF metadata")
                value = ''
        else:
            # Check if value_type is valid before unpacking
            if value_type not in self.value_packing:
                 raise ValueError(f"Unsupported GGUF value type: {value_type}")
            fmt = self.value_packing.get(value_type)
            size = self.value_lengths.get(value_type)
            value = struct.unpack(fmt, file.read(size))[0]
        return value

    def read_metadata(self) -> Dict[str, Any]:
        """
        Read metadata from the GGUF file.

        Returns:
            A dictionary of metadata
        """
        if self.metadata:
            return self.metadata

        with open(self.model_path, "rb") as file:
            # Check magic bytes
            magic = file.read(4)
            if magic != self.GGUF_MAGIC_BYTES:
                raise ValueError(
                    f"Not a valid GGUF file: magic number mismatch, got {magic}, "
                    f"expected {self.GGUF_MAGIC_BYTES}"
                )

            # Read version
            self.version = self._unpack(GGUFValueType.UINT32, file=file)
            if self.version not in self.SUPPORTED_GGUF_VERSIONS:
                raise ValueError(
                    f"Unsupported GGUF version {self.version}, only versions "
                    f"{self.SUPPORTED_GGUF_VERSIONS} are supported"
                )

            # Read tensor count
            self.tensor_count = self._unpack(GGUFValueType.UINT64, file=file)

            # Read KV count (metadata key-value pairs)
            if self.version == 3:
                self.kv_count = self._unpack(GGUFValueType.UINT64, file=file)
            elif self.version == 2:
                self.kv_count = self._unpack(GGUFValueType.UINT32, file=file)

            # Read all metadata key-value pairs
            for _ in range(self.kv_count):
                # Read key
                if self.version == 3:
                    key_length = self._unpack(GGUFValueType.UINT64, file=file)
                elif self.version == 2:
                    key_length = 0
                    while key_length == 0:
                        # Seek until next key is found
                        key_length = self._unpack(GGUFValueType.UINT32, file=file)
                    file.read(4)  # 4 byte offset for GGUFv2

                key = file.read(key_length)
                key_str = key.decode("utf-8")

                # Read value type
                value_type_int = self._unpack(GGUFValueType.UINT32, file=file) # Read raw int
                try:
                    value_type = GGUFValueType(value_type_int)
                except ValueError:
                     raise ValueError(f"Unknown GGUF value type ID: {value_type_int}")

                # Read value based on type
                if value_type == GGUFValueType.ARRAY: # Type 9
                    # Read array value type
                    array_value_type_int = self._unpack(GGUFValueType.UINT32, file=file)
                    try:
                         array_value_type = GGUFValueType(array_value_type_int)
                    except ValueError:
                         raise ValueError(f"Unknown GGUF array value type ID: {array_value_type_int}")

                    # Read array length
                    if self.version == 3:
                        array_length = self._unpack(GGUFValueType.UINT64, file=file)
                    elif self.version == 2:
                        array_length = self._unpack(GGUFValueType.UINT32, file=file)
                        file.read(4)  # 4 byte offset for GGUFv2

                    # Read array values
                    array = [
                        self._get_single_value(array_value_type, file)
                        for _ in range(array_length)
                    ]
                    self.metadata[key_str] = array
                else:
                    # Read single value
                    value = self._get_single_value(value_type, file)
                    self.metadata[key_str] = value

        return self.metadata

    def get_context_length(self) -> Optional[int]:
        """
        Get the context length from the metadata.

        Looks for any metadata key that ends with '.context_length' using regex.

        Returns:
            The context length or None if not found
        """
        metadata = self.read_metadata()
        context_length_pattern = re.compile(r'\.context_length$')

        # Look for any key ending with .context_length
        for key, value in metadata.items():
            if context_length_pattern.search(key):
                try:
                    return int(value)
                except (ValueError, TypeError):
                    # If the value can't be converted to int, try the next key
                    continue

        # If no key ending with .context_length is found or none have valid int values
        return None

    def get_model_architecture(self) -> Optional[str]:
        """
        Get the model architecture from the metadata.

        Returns:
            The model architecture or None if not found
        """
        metadata = self.read_metadata()

        # First try general.architecture
        if "general.architecture" in metadata:
            return metadata["general.architecture"]

        # Look for any key ending with .architecture
        for key in metadata:
            if key.endswith('.architecture'):
                return metadata[key]

        return None

    def get_model_name(self) -> Optional[str]:
        """
        Get the model name from the metadata.

        Returns:
            The model name or None if not found
        """
        metadata = self.read_metadata()

        # First try general.name
        if "general.name" in metadata:
            return metadata["general.name"]

        # Look for any key ending with .name
        for key in metadata:
            if key.endswith('.name'):
                return metadata[key]

        return None

    def get_rope_freq_base(self) -> Optional[float]:
        """
        Get the RoPE frequency base from the metadata.

        Returns:
            The RoPE frequency base or None if not found
        """
        metadata = self.read_metadata()

        # Look for any key ending with .rope.freq_base
        for key in metadata:
            if key.endswith('.rope.freq_base'):
                try:
                    return float(metadata[key])
                except (ValueError, TypeError):
                    # If the value can't be converted to float, try the next key
                    continue

        # Also check for keys ending with .rope_freq_base (alternative format)
        for key in metadata:
            if key.endswith('.rope_freq_base'):
                try:
                    return float(metadata[key])
                except (ValueError, TypeError):
                    continue

        return None

    def get_rope_freq_scale(self) -> Optional[float]:
        """
        Get the RoPE frequency scale from the metadata.

        Returns:
            The RoPE frequency scale or None if not found
        """
        metadata = self.read_metadata()

        # Look for any key ending with .rope.freq_scale
        for key in metadata:
            if key.endswith('.rope.freq_scale'):
                try:
                    return float(metadata[key])
                except (ValueError, TypeError):
                    # If the value can't be converted to float, try the next key
                    continue

        # Also check for keys ending with .rope_freq_scale (alternative format)
        for key in metadata:
            if key.endswith('.rope_freq_scale'):
                try:
                    return float(metadata[key])
                except (ValueError, TypeError):
                    continue

        return None

    def get_quantization_type(self) -> Optional[str]:
        """
        Get the quantization type from the metadata.

        Returns:
            The quantization type or None if not found
        """
        metadata = self.read_metadata()

        # First try general.file_type
        if "general.file_type" in metadata:
            return metadata["general.file_type"]

        # Look for any key ending with .file_type or .quantization_type
        for key in metadata:
            if key.endswith('.file_type') or key.endswith('.quantization_type'):
                return metadata[key]

        return None

    def get_embedding_length(self) -> Optional[int]:
        """
        Get the embedding length from the metadata.

        Returns:
            The embedding length or None if not found
        """
        metadata = self.read_metadata()

        # Look for any key ending with .embedding_length
        for key in metadata:
            if key.endswith('.embedding_length'):
                try:
                    return int(metadata[key])
                except (ValueError, TypeError):
                    # If the value can't be converted to int, try the next key
                    continue

        # Also check for keys ending with .hidden_size or .dim (alternative names)
        for key in metadata:
            if key.endswith('.hidden_size') or key.endswith('.dim'):
                try:
                    return int(metadata[key])
                except (ValueError, TypeError):
                    continue

        return None

    def get_block_count(self) -> Optional[int]:
        """
        Get the block count from the metadata.

        Returns:
            The block count or None if not found
        """
        metadata = self.read_metadata()

        # Look for any key ending with .block_count
        for key in metadata:
            if key.endswith('.block_count'):
                try:
                    return int(metadata[key])
                except (ValueError, TypeError):
                    # If the value can't be converted to int, try the next key
                    continue

        # Also check for keys ending with .n_layer or .num_layers (alternative names)
        for key in metadata:
            if key.endswith('.n_layer') or key.endswith('.num_layers'):
                try:
                    return int(metadata[key])
                except (ValueError, TypeError):
                    continue

        return None

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information from the metadata.

        Returns:
            A dictionary with model information
        """
        metadata = self.read_metadata()

        info = {
            "architecture": self.get_model_architecture(),
            "name": self.get_model_name(),
            "context_length": self.get_context_length(),
            "quantization_type": self.get_quantization_type(),
            "embedding_length": self.get_embedding_length(),
            "block_count": self.get_block_count(),
            "rope_freq_base": self.get_rope_freq_base(),
            "rope_freq_scale": self.get_rope_freq_scale(),
            "version": self.version,
            "tensor_count": self.tensor_count,
            "metadata_count": self.kv_count,
            "full_metadata": metadata
        }

        return info

    def print_metadata_keys(self, verbose: bool = False) -> None:
        """
        Print all metadata keys for debugging purposes.

        Args:
            verbose: Whether to print the values as well
        """
        metadata = self.read_metadata()

        print(f"GGUF file: {self.model_path}")
        print(f"GGUF version: {self.version}")
        print(f"Metadata key-value pairs: {self.kv_count}")
        print(f"Tensor count: {self.tensor_count}")
        print("\nMetadata keys:")

        # Group keys by prefix
        prefixes = {}
        for key in sorted(metadata.keys()):
            prefix = key.split('.')[0] if '.' in key else 'other'
            if prefix not in prefixes:
                prefixes[prefix] = []
            prefixes[prefix].append(key)

        # Print keys grouped by prefix
        for prefix in sorted(prefixes.keys()):
            print(f"\n[{prefix}]")
            for key in sorted(prefixes[prefix]):
                if verbose:
                    value = metadata[key]
                    # Truncate long values
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    elif isinstance(value, list) and len(value) > 10:
                        value = value[:10] + ["..."]
                    print(f"  {key} = {value}")
                else:
                    print(f"  {key}")

        # Print detected model info
        print("\nDetected model information:")
        info = self.get_model_info()
        for key, value in info.items():
            if key != "full_metadata":  # Skip the full metadata
                print(f"  {key}: {value}")


def extract_max_context_from_gguf(model_path: str, debug: bool = False) -> Optional[int]:
    """
    Extract maximum context length from a GGUF file using regex pattern matching
    on metadata keys.

    Args:
        model_path: Path to the GGUF file
        debug: Whether to print debug information during extraction

    Returns:
        Maximum context length or None if not found
    """
    import re
    try:
        # Get model filename for debugging
        filename = os.path.basename(model_path)
        if debug:
            print(f"Analyzing context length for: {filename}")

        # Get metadata from the GGUF file
        metadata = {}
        try:
            # Use simple_gguf_info first as it might be more robust
            info = simple_gguf_info(model_path)
            if "metadata" in info:
                metadata = info["metadata"]
                if debug:
                    print(f"Found {len(metadata)} metadata keys using simple_gguf_info")
            else:
                 if debug:
                     print("simple_gguf_info did not return metadata.")
                 # Fall back to GGUFReader if simple_gguf_info fails or has no metadata
                 try:
                     reader = GGUFReader(model_path)
                     metadata = reader.read_metadata()
                     if debug:
                         print(f"Found {len(metadata)} metadata keys using GGUFReader")
                 except Exception as e_reader:
                     if debug:
                         print(f"Error with GGUFReader: {e_reader}")
                     metadata = {}

        except Exception as e_simple:
            if debug:
                print(f"Error with simple_gguf_info: {e_simple}, falling back to GGUFReader")
            # Fall back to GGUFReader if simple_gguf_info fails
            try:
                reader = GGUFReader(model_path)
                metadata = reader.read_metadata()
                if debug:
                    print(f"Found {len(metadata)} metadata keys using GGUFReader")
            except Exception as e_reader_fallback:
                if debug:
                    print(f"Error with GGUFReader fallback: {e_reader_fallback}")
                metadata = {}

        if not metadata:
             if debug:
                 print("Could not retrieve metadata from GGUF file.")
             return None

        # Define regex patterns for context length keys
        context_length_pattern = re.compile(r'\.context_length$')

        # Find all keys ending with .context_length
        context_length_keys = [k for k in metadata.keys() if context_length_pattern.search(k)]
        if debug and context_length_keys:
            print(f"Found {len(context_length_keys)} keys ending with .context_length: {context_length_keys}")

        # Priority 1: Check any context_length keys in metadata
        if context_length_keys:
            # Sort keys alphabetically for deterministic behavior (optional)
            sorted_keys = sorted(context_length_keys)

            for key in sorted_keys:
                try:
                    context_length = int(metadata[key])
                    if debug:
                        print(f"Using context length from metadata key '{key}': {context_length}")
                    return context_length
                except (ValueError, TypeError):
                    if debug:
                        print(f"Failed to convert value for key '{key}' to int: {metadata[key]}")
                    continue

        # Priority 2: Try the GGUFReader method as fallback (which also uses regex now)
        try:
            reader = GGUFReader(model_path)
            context_length = reader.get_context_length()
            if context_length is not None:
                if debug:
                    print(f"Found context length using GGUFReader.get_context_length(): {context_length}")
                return context_length
        except Exception as e:
            if debug:
                print(f"Error using GGUFReader.get_context_length(): {e}")

        if debug:
            print("Could not find a valid context length in GGUF metadata")
        return None
    except Exception as e:
        if debug:
            print(f"Error extracting context length: {e}")
            import traceback
            print(traceback.format_exc())
        else:
            print(f"Error extracting context length: {e}")
        return None


def extract_rope_parameters_from_gguf(model_path: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract RoPE frequency base and scale from a GGUF file.

    Args:
        model_path: Path to the GGUF file

    Returns:
        Tuple of (rope_freq_base, rope_freq_scale) or (None, None) if not found
    """
    try:
        reader = GGUFReader(model_path)
        return reader.get_rope_freq_base(), reader.get_rope_freq_scale()
    except Exception as e:
        print(f"Error extracting RoPE parameters: {e}")
        return None, None


def get_gguf_metadata(model_path: str) -> Dict[str, Any]:
    """
    Get all metadata from a GGUF file.

    Args:
        model_path: Path to the GGUF file

    Returns:
        Dictionary with all metadata
    """
    try:
        reader = GGUFReader(model_path)
        return reader.read_metadata()
    except Exception as e:
        print(f"Error reading GGUF metadata: {e}")
        return {}


def get_model_info(model_path: str) -> Dict[str, Any]:
    """
    Get comprehensive model information from a GGUF file.

    Args:
        model_path: Path to the GGUF file

    Returns:
        Dictionary with model information
    """
    try:
        reader = GGUFReader(model_path)
        return reader.get_model_info()
    except Exception as e:
        print(f"Error getting model info: {e}")
        return {}


def print_gguf_metadata(model_path: str, verbose: bool = False) -> None:
    """
    Print metadata from a GGUF file.

    Args:
        model_path: Path to the GGUF file
        verbose: Whether to print the values as well
    """
    try:
        reader = GGUFReader(model_path)
        reader.print_metadata_keys(verbose)
    except Exception as e:
        print(f"Error reading GGUF metadata: {e}")


def simple_gguf_info(model_path: str) -> Dict[str, Any]:
    """
    A simpler function to extract basic information from a GGUF file.
    This function uses mmap directly and is more robust for problematic files.

    Args:
        model_path: Path to the GGUF file

    Returns:
        Dictionary with basic model information
    """
    import mmap
    import struct
    import os

    result = {
        "file_size_bytes": os.path.getsize(model_path),
        "file_size_gb": os.path.getsize(model_path) / (1024 * 1024 * 1024),
        "metadata": {}
    }

    try:
        with open(model_path, 'rb') as f:
            # Map the file into memory
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                # Check GGUF magic
                magic = mm.read(4)
                if magic != b'GGUF':
                    result["error"] = f"Not a GGUF file (magic: {magic})"
                    return result

                result["magic"] = "GGUF"

                # Read version
                version_bytes = mm.read(4)
                version = struct.unpack('<I', version_bytes)[0]
                result["version"] = version

                # Read tensor count
                tensor_count_bytes = mm.read(8)
                tensor_count = struct.unpack('<Q', tensor_count_bytes)[0]
                result["tensor_count"] = tensor_count

                # Read KV count
                if version == 3:
                    kv_count_bytes = mm.read(8)
                    kv_count = struct.unpack('<Q', kv_count_bytes)[0]
                elif version == 2:
                    kv_count_bytes = mm.read(4)
                    kv_count = struct.unpack('<I', kv_count_bytes)[0]
                    # GGUF V2 alignment: kv_count is uint32, followed by 4 bytes padding if tensor_count is uint64
                    # Let's assume tensor_count is always uint64 for simplicity here, as per spec
                    mm.read(4) # Skip potential padding
                else:
                    result["error"] = f"Unsupported GGUF version: {version}"
                    return result

                result["kv_count"] = kv_count


                # Read metadata keys and values
                metadata = {}
                for i in range(kv_count): # Use index i for error reporting
                    try:
                        # Read key length
                        if version == 3:
                            key_len_bytes = mm.read(8)
                            if len(key_len_bytes) < 8:
                                metadata[f"ERROR_key_len_{i}"] = f"Failed to read key length (expected 8 bytes, got {len(key_len_bytes)}) at index {i}"
                                break # Stop reading metadata if structure is broken
                            key_len = struct.unpack('<Q', key_len_bytes)[0]
                        elif version == 2:
                            key_len_bytes = mm.read(4)
                            if len(key_len_bytes) < 4:
                                metadata[f"ERROR_key_len_{i}"] = f"Failed to read key length (expected 4 bytes, got {len(key_len_bytes)}) at index {i}"
                                break
                            key_len = struct.unpack('<I', key_len_bytes)[0]
                            mm.read(4)  # Skip 4 bytes alignment padding

                        # Read key
                        key_bytes = mm.read(key_len)
                        if len(key_bytes) < key_len:
                            metadata[f"ERROR_key_read_{i}"] = f"Failed to read key (expected {key_len} bytes, got {len(key_bytes)}) at index {i}"
                            break

                        try:
                            key = key_bytes.decode('utf-8', errors='replace')
                        except Exception:
                            key = f"binary_key_{i}_{key_bytes.hex()[:16]}" # Make error key unique

                        # Read value type
                        val_type_bytes = mm.read(4)
                        if len(val_type_bytes) < 4:
                            metadata[f"ERROR_val_type_{i}"] = f"Failed to read value type for key '{key}' (expected 4 bytes, got {len(val_type_bytes)}) at index {i}"
                            break
                        val_type = struct.unpack('<I', val_type_bytes)[0]

                        # Read value based on type (simplified error handling for brevity)
                        value = None
                        try:
                            if val_type == GGUFValueType.UINT8: value = struct.unpack('<B', mm.read(1))[0]
                            elif val_type == GGUFValueType.INT8: value = struct.unpack('<b', mm.read(1))[0]
                            elif val_type == GGUFValueType.UINT16: value = struct.unpack('<H', mm.read(2))[0]
                            elif val_type == GGUFValueType.INT16: value = struct.unpack('<h', mm.read(2))[0]
                            elif val_type == GGUFValueType.UINT32: value = struct.unpack('<I', mm.read(4))[0]
                            elif val_type == GGUFValueType.INT32: value = struct.unpack('<i', mm.read(4))[0]
                            elif val_type == GGUFValueType.FLOAT32: value = struct.unpack('<f', mm.read(4))[0]
                            elif val_type == GGUFValueType.BOOL: value = struct.unpack('<?', mm.read(1))[0]
                            elif val_type == GGUFValueType.STRING:
                                str_len = struct.unpack('<Q', mm.read(8))[0]
                                str_bytes = mm.read(str_len)
                                value = str_bytes.decode('utf-8', errors='replace')
                            elif val_type == GGUFValueType.UINT64: value = struct.unpack('<Q', mm.read(8))[0]
                            elif val_type == GGUFValueType.INT64: value = struct.unpack('<q', mm.read(8))[0]
                            elif val_type == GGUFValueType.FLOAT64: value = struct.unpack('<d', mm.read(8))[0]
                            elif val_type == GGUFValueType.ARRAY:
                                arr_type = struct.unpack('<I', mm.read(4))[0]
                                if version == 3:
                                    arr_len = struct.unpack('<Q', mm.read(8))[0]
                                elif version == 2:
                                    arr_len = struct.unpack('<I', mm.read(4))[0]
                                    mm.read(4) # Skip alignment padding

                                # Simplified array handling: read if small, otherwise skip
                                if arr_len < 100: # Arbitrary limit
                                    arr_values = []
                                    elem_size = GGUFReader.value_lengths.get(GGUFValueType(arr_type))
                                    if elem_size:
                                        fmt = GGUFReader.value_packing.get(GGUFValueType(arr_type))
                                        for _ in range(arr_len):
                                            arr_values.append(struct.unpack(fmt, mm.read(elem_size))[0])
                                        value = arr_values
                                    elif arr_type == GGUFValueType.STRING: # Handle string array
                                        for _ in range(arr_len):
                                            str_len = struct.unpack('<Q', mm.read(8))[0]
                                            str_bytes = mm.read(str_len)
                                            arr_values.append(str_bytes.decode('utf-8', errors='replace'))
                                        value = arr_values
                                    else: # Unknown array element type
                                        value = f"<ARRAY type {arr_type} len {arr_len} - Cannot read elements>"
                                        # Attempt to skip based on known sizes (best effort)
                                        # This part is complex and prone to errors if types are unknown.
                                        # For simplicity, we might just break metadata reading here.
                                        metadata[f"ERROR_array_skip_{i}"] = f"Cannot reliably skip array elements of unknown type {arr_type} for key '{key}'"
                                        break # Stop reading metadata
                                else: # Array too large or cannot read elements
                                    value = f"<ARRAY type {arr_type} len {arr_len} - Too large or unreadable>"
                                    # Attempt to skip based on calculated size
                                    bytes_to_skip = 0
                                    elem_size = GGUFReader.value_lengths.get(GGUFValueType(arr_type))
                                    if elem_size:
                                        bytes_to_skip = arr_len * elem_size
                                    elif arr_type == GGUFValueType.STRING:
                                        # Need to read each length to skip strings - complex and slow
                                        # For simplicity, break reading here if array is large
                                        metadata[f"ERROR_array_skip_{i}"] = f"Cannot reliably skip large string array for key '{key}'"
                                        break
                                    else: # Unknown type
                                        metadata[f"ERROR_array_skip_{i}"] = f"Cannot skip large array of unknown type {arr_type} for key '{key}'"
                                        break

                                    if bytes_to_skip > 0:
                                        try:
                                            mm.seek(bytes_to_skip, 1)
                                        except ValueError:
                                            metadata[f"ERROR_array_seek_{i}"] = f"Failed seeking past large array for key '{key}'"
                                            break
                            else:
                                metadata[f"ERROR_unknown_type_{i}"] = f"Unknown metadata value type: {val_type} for key '{key}'"
                                # Stop processing further metadata as alignment is likely lost
                                break

                            metadata[key] = value

                        except struct.error as se:
                            metadata[f"ERROR_unpack_{i}"] = f"Struct error reading value for key '{key}' (type {val_type}): {se}"
                            break # Stop reading on unpack error
                        except MemoryError as me:
                             metadata[f"ERROR_memory_{i}"] = f"Memory error reading value for key '{key}' (type {val_type}): {me}"
                             break # Stop reading on memory error
                        except Exception as e_val:
                            metadata[f"ERROR_value_{i}"] = f"General error reading value for key '{key}' (type {val_type}): {e_val}"
                            # Depending on the error, we might want to break or continue
                            break # Safer to break

                    except Exception as e_outer:
                        metadata[f"ERROR_outer_{i}"] = f"Outer loop error at index {i}: {e_outer}"
                        # Break the loop if something went wrong reading key/type structure
                        break

                result["metadata"] = metadata

                # Extract specific information using helper functions

                # Helper function to find keys by suffix (case-insensitive)
                def find_by_suffix(suffix, default=None):
                    pattern = re.compile(re.escape(suffix) + '$', re.IGNORECASE)
                    # Prioritize keys with fewer parts (e.g., 'llama.context_length' over 'a.b.llama.context_length')
                    candidates = []
                    for key, value in metadata.items():
                        if isinstance(key, str) and pattern.search(key): # Check if key is string
                             candidates.append((key.count('.'), key, value))
                    if candidates:
                        candidates.sort() # Sort by number of dots, then alphabetically
                        return candidates[0][2] # Return value of the best match
                    return default

                # Helper function to find keys by contains (case-insensitive)
                def find_by_contains(substring, default=None):
                    pattern = re.compile(re.escape(substring), re.IGNORECASE)
                    for key, value in metadata.items():
                         if isinstance(key, str) and pattern.search(key): # Check if key is string
                            return value
                    return default

                # Get architecture
                result["architecture"] = metadata.get("general.architecture",
                                        find_by_suffix(".architecture"))

                # Get model name
                result["name"] = metadata.get("general.name",
                               find_by_suffix(".name"))

                # Get context length - look for any key ending with .context_length
                context_length_val = find_by_suffix(".context_length")
                if context_length_val is not None:
                    try:
                        result["context_length"] = int(context_length_val)
                        # Find the key that provided the value for source info
                        context_length_pattern = re.compile(r'\.context_length$', re.IGNORECASE)
                        source_key = "unknown"
                        candidates = []
                        for k, v in metadata.items():
                             if isinstance(k, str) and context_length_pattern.search(k):
                                try:
                                    if int(v) == result["context_length"]:
                                        candidates.append((k.count('.'), k))
                                except (ValueError, TypeError):
                                    continue
                        if candidates:
                            candidates.sort()
                            source_key = candidates[0][1]
                        result["context_length_source"] = source_key
                    except (ValueError, TypeError):
                        result["context_length"] = None
                        result["context_length_error"] = f"Invalid value found for context length: {context_length_val}"
                else:
                    result["context_length"] = None

                # Get embedding length
                result["embedding_length"] = find_by_suffix(".embedding_length",
                                           find_by_suffix(".hidden_size",
                                           find_by_suffix(".dim")))
                if result["embedding_length"] is not None:
                    try: result["embedding_length"] = int(result["embedding_length"])
                    except (ValueError, TypeError): result["embedding_length"] = None


                # Get block count
                result["block_count"] = find_by_suffix(".block_count",
                                      find_by_suffix(".n_layer",
                                      find_by_suffix(".num_layers")))
                if result["block_count"] is not None:
                    try: result["block_count"] = int(result["block_count"])
                    except (ValueError, TypeError): result["block_count"] = None

                # Get quantization type (prefer general.quantization_version if available)
                quant_version = metadata.get("general.quantization_version")
                file_type = metadata.get("general.file_type") # Often contains enum value
                if quant_version:
                     result["quantization_type"] = f"v{quant_version}" # More informative
                elif file_type is not None:
                     # Try to map enum to string if possible (requires gguf library or manual mapping)
                     result["quantization_type"] = str(file_type) # Fallback to raw value
                else:
                     result["quantization_type"] = find_by_suffix(".file_type",
                                                 find_by_suffix(".quantization_type"))


                # Get RoPE parameters
                result["rope_freq_base"] = find_by_suffix(".rope.freq_base",
                                         find_by_suffix(".rope_freq_base"))
                if result["rope_freq_base"] is not None:
                     try: result["rope_freq_base"] = float(result["rope_freq_base"])
                     except (ValueError, TypeError): result["rope_freq_base"] = None

                result["rope_freq_scale"] = find_by_suffix(".rope.freq_scale",
                                          find_by_suffix(".rope_freq_scale"))
                if result["rope_freq_scale"] is not None:
                     try: result["rope_freq_scale"] = float(result["rope_freq_scale"])
                     except (ValueError, TypeError): result["rope_freq_scale"] = None


                # Get attention parameters
                result["head_count"] = find_by_suffix(".attention.head_count",
                                     find_by_suffix(".head_count",
                                     find_by_suffix(".num_heads",
                                     find_by_suffix(".n_head")))) # Add n_head
                if result["head_count"] is not None:
                     try: result["head_count"] = int(result["head_count"])
                     except (ValueError, TypeError): result["head_count"] = None


                result["head_count_kv"] = find_by_suffix(".attention.head_count_kv",
                                        find_by_suffix(".head_count_kv",
                                        find_by_suffix(".num_kv_heads",
                                        find_by_suffix(".n_head_kv")))) # Add n_head_kv
                if result["head_count_kv"] is not None:
                     try: result["head_count_kv"] = int(result["head_count_kv"])
                     except (ValueError, TypeError): result["head_count_kv"] = None


                # Get feed forward length
                result["feed_forward_length"] = find_by_suffix(".feed_forward_length",
                                              find_by_suffix(".intermediate_size",
                                              find_by_suffix(".ffn_hidden_size"))) # Add ffn_hidden_size
                if result["feed_forward_length"] is not None:
                     try: result["feed_forward_length"] = int(result["feed_forward_length"])
                     except (ValueError, TypeError): result["feed_forward_length"] = None


    except mmap.error as me:
        result["error"] = f"Mmap error: {me}"
    except FileNotFoundError:
        result["error"] = f"File not found: {model_path}"
    except Exception as e:
        result["error"] = f"An unexpected error occurred: {e}"
        import traceback
        result["traceback"] = traceback.format_exc()

    return result


def debug_gguf_context_length(model_path: str) -> None:
    """
    Debug function to print context length information from a GGUF file.
    Uses regex pattern matching to find context-related metadata keys.

    Args:
        model_path: Path to the GGUF file
    """
    import re
    from rich import print
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table

    filename = os.path.basename(model_path)
    print(f"[bold]Debugging context length for {filename}[/bold]")

    try:
        # Use the robust simple_gguf_info first
        info = simple_gguf_info(model_path)

        # Print context length from top-level info if available
        if "context_length" in info and info["context_length"] is not None:
            print(f"[green]Context length from simple_gguf_info: {info['context_length']}[/green]")
            if "context_length_source" in info:
                print(f"[green]Source key (from simple_gguf_info): {info['context_length_source']}[/green]")
            else:
                 print("[yellow]Source key not determined by simple_gguf_info[/yellow]")
        elif "context_length_error" in info:
             print(f"[yellow]Could not determine context length from simple_gguf_info: {info['context_length_error']}[/yellow]")
        else:
            print("[yellow]No context length found by simple_gguf_info[/yellow]")

        # Print all metadata keys if available
        if "metadata" in info and info["metadata"]:
            metadata = info["metadata"]
            print(f"\n[bold]Found {len(metadata)} metadata entries[/bold]")

            # Use regex to find keys ending with .context_length
            context_length_pattern = re.compile(r'\.context_length$', re.IGNORECASE)
            context_length_keys = []
            invalid_context_keys = []

            for k, v in metadata.items():
                 # Ensure key is a string before matching
                 if isinstance(k, str) and context_length_pattern.search(k):
                    context_length_keys.append((k, v))
                    try:
                        int(v) # Check if value is a valid integer
                    except (ValueError, TypeError, OverflowError):
                         invalid_context_keys.append((k,v))


            if context_length_keys:
                print(f"\n[bold]Found {len(context_length_keys)} keys potentially related to context length (ending with .context_length):[/bold]")

                # Create a table for better visualization
                table = Table(title="Potential Context Length Keys")
                table.add_column("Key", style="cyan", overflow="fold")
                table.add_column("Value", style="green", overflow="fold")
                table.add_column("Is Integer?", style="yellow")

                # Sort keys for consistent output (e.g., alphabetically)
                sorted_keys = sorted(context_length_keys, key=lambda item: item[0])

                for key, value in sorted_keys:
                    try:
                        int_value = int(value)
                        valid = "[green]✓ Yes[/green]"
                    except (ValueError, TypeError, OverflowError):
                        valid = "[red]✗ No[/red]"
                    # Limit value display length
                    display_value = str(value)
                    if len(display_value) > 60:
                        display_value = display_value[:57] + "..."
                    table.add_row(key, display_value, valid)

                print(table)
            else:
                print("\n[yellow]No keys ending with '.context_length' found in metadata.[/yellow]")

            # Create a summary panel
            context_text = Text()
            final_context = None
            source = None

            # Determine the final context length using priority:
            # 1. Value from simple_gguf_info if valid integer
            # 2. Iterate through found .context_length keys and take the first valid integer

            if "context_length" in info and isinstance(info["context_length"], int):
                final_context = info["context_length"]
                source = info.get("context_length_source", "simple_gguf_info (key unspecified)")
            elif context_length_keys:
                 # Iterate through sorted keys to find the first valid one
                 sorted_valid_keys = sorted([k for k,v in context_length_keys if (k,v) not in invalid_context_keys], key=lambda item: item[0])
                 for key, value in sorted_valid_keys:
                     try:
                         final_context = int(value)
                         source = f"'{key}' from metadata"
                         break # Use the first valid one found
                     except (ValueError, TypeError, OverflowError):
                         continue # Should not happen due to pre-filtering, but check anyway

            if final_context is not None:
                context_text.append(f"Final detected context length: {final_context}\n", style="bold green")
                context_text.append(f"Source: {source}\n", style="cyan")
                context_text.append("\nThis is the context length that will likely be used by default.", style="yellow")
            else:
                context_text.append("Could not determine a valid context length from metadata.\n", style="bold red")
                if invalid_context_keys:
                     context_text.append("Found keys ending in '.context_length' but values were not valid integers:\n", style="yellow")
                     for k,v in invalid_context_keys:
                         context_text.append(f"  - {k}: {str(v)[:50]}...\n", style="yellow")
                context_text.append("Consider checking the model card or specifying context manually.", style="yellow")


            print("\n", Panel(
                context_text,
                title="Context Length Summary",
                border_style="blue" if final_context is not None else "red"
            ))

        elif "error" in info:
             print(f"[red]Error reading GGUF info: {info['error']}[/red]")
             if "traceback" in info:
                 print(f"[red]{info['traceback']}[/red]")
        else:
            print("[yellow]No metadata found in the GGUF file.[/yellow]")


    except Exception as e:
        print(f"[red]An unexpected error occurred during debugging: {str(e)}[/red]")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    from rich import print
    from rich.console import Console
    from rich.panel import Panel
    import sys
    import os
    import argparse
    import re

    # Create a command-line interface
    parser = argparse.ArgumentParser(description="GGUF Context Length Analyzer")
    parser.add_argument("path", nargs="?", help="Path to the GGUF model file")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug output for extract function")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose metadata output using GGUFReader")
    parser.add_argument("--extract", "-e", action="store_true", help="Use extract_max_context_from_gguf function instead of debug function")
    parser.add_argument("--regex", "-r", help="Search metadata keys using a regex pattern (case-insensitive)")

    args = parser.parse_args()
    console = Console()

    # Use command line argument if provided, otherwise try to find one or use default
    if args.path:
        path = args.path
    else:
        # Try to find a GGUF file in the current directory
        gguf_files = [f for f in os.listdir('.') if f.lower().endswith('.gguf')]
        if gguf_files:
            # Sort files, maybe alphabetically or by size/date? Alphabetical is simple.
            gguf_files.sort()
            path = gguf_files[0]
            print(f"[yellow]No path provided, using first GGUF file found: {path}[/yellow]")
        else:
            # Use default path as fallback if no GGUF file found
            default_path = r"""C:\Users\koula\.inferno\models\Qwen2-0.5B-Instruct-GGUF\qwen2-0_5b-instruct-q2_k.gguf"""
            if os.path.exists(default_path):
                 path = default_path
                 print(f"[yellow]No GGUF file found in current directory. Using default path: {path}[/yellow]")
            else:
                 print("[red]Error: No GGUF file path provided, none found in current directory, and default path does not exist.[/red]")
                 sys.exit(1)


    if not os.path.exists(path):
        print(f"[red]File not found: {path}[/red]")
        sys.exit(1)

    # Print file information
    try:
        file_size_bytes = os.path.getsize(path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        file_size_gb = file_size_mb / 1024
        print(f"[bold]Analyzing GGUF file:[/bold] {os.path.basename(path)}")
        print(f"[bold]File size:[/bold] {file_size_mb:.2f} MB ({file_size_gb:.2f} GB)")
    except OSError as e:
        print(f"[red]Error accessing file properties: {e}[/red]")
        sys.exit(1)

    # Run the appropriate analysis based on arguments
    if args.extract:
        # Use the extract_max_context_from_gguf function with debug flag
        print("\n[bold cyan]Running extract_max_context_from_gguf...[/bold cyan]")
        context_length = extract_max_context_from_gguf(path, debug=args.debug)
        if context_length is not None:
            console.print(Panel(f"Detected context length: {context_length}",
                               title="Context Length (from extract function)", border_style="green"))
        else:
            console.print(Panel("Could not detect context length using extract function",
                               title="Context Length (from extract function)", border_style="red"))
    else:
        # Use the debug_gguf_context_length function by default
        print("\n[bold cyan]Running debug_gguf_context_length...[/bold cyan]")
        debug_gguf_context_length(path)

    # If regex pattern is provided, search metadata keys using simple_gguf_info
    if args.regex:
        print(f"\n[bold cyan]Searching metadata for regex pattern: '{args.regex}'[/bold cyan]")
        try:
            pattern = re.compile(args.regex, re.IGNORECASE)
            # Use simple_gguf_info as it's generally more robust
            info = simple_gguf_info(path)
            metadata = info.get('metadata', {})

            if "error" in info and not metadata:
                 print(f"[red]Could not read metadata to search: {info['error']}[/red]")
            elif not metadata:
                 print("[yellow]No metadata found to search.[/yellow]")
            else:
                matching_keys = {}
                for key, value in metadata.items():
                     # Ensure key is a string before matching
                     if isinstance(key, str) and pattern.search(key):
                         matching_keys[key] = value

                if matching_keys:
                    print(f"\n[bold green]Found {len(matching_keys)} metadata keys matching pattern:[/bold green]")
                    # Sort keys for display
                    for key in sorted(matching_keys.keys()):
                        value = matching_keys[key]
                        # Truncate long values for display
                        display_value = str(value)
                        if isinstance(value, list) and len(value) > 10:
                            display_value = f"{str(value[:10])[:-1]} ...]" # Show first 10 elements
                        elif isinstance(value, str) and len(value) > 100:
                            display_value = value[:97] + "..."
                        print(f"  [cyan]{key}[/cyan]: {display_value}")
                else:
                    print(f"\n[yellow]No metadata keys match the pattern '{args.regex}'[/yellow]")

        except re.error as e:
            print(f"[red]Invalid regex pattern: {e}[/red]")
        except Exception as e:
             print(f"[red]Error during regex search: {e}[/red]")


    # Show all metadata if verbose flag is set, using GGUFReader
    if args.verbose:
        print("\n[bold cyan]Running GGUFReader to print verbose metadata...[/bold cyan]")
        try:
            reader = GGUFReader(path)
            reader.print_metadata_keys(verbose=True)
        except Exception as e:
            print(f"[red]Error printing verbose metadata using GGUFReader: {e}[/red]")
            # Optionally, try printing metadata from simple_gguf_info if GGUFReader failed
            if 'info' in locals() and 'metadata' in info and info['metadata']:
                 print("\n[yellow]Attempting to print metadata from simple_gguf_info instead:[/yellow]")
                 from rich.pretty import pprint
                 pprint(info['metadata'])


    print("\n[bold green]Analysis complete![/bold green]")