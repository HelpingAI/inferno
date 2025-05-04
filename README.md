# 🔥 Inferno: Ignite Your Local AI Experience 🔥

<div align="center">
  <img src="https://img.shields.io/badge/Inferno-Local%20LLM%20Server-orange?style=for-the-badge&logo=python&logoColor=white" alt="Inferno Logo">

  <p><strong>Unleash the Blazing Power of Cutting-Edge LLMs on Your Own Hardware</strong></p>

  <p>
    Run Llama 3.3, DeepSeek-R1, Phi-4, Gemma 3, Mistral Small 3.1, and other state-of-the-art language models locally with scorching-fast performance. Inferno provides an intuitive CLI and an OpenAI/Ollama-compatible API, putting the inferno of AI innovation directly in your hands.
  </p>

  <!-- Badges -->
  <p>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-HelpingAI%20Open%20Source-blue?style=flat-square" alt="License"></a>
    <a href="#requirements"><img src="https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python&logoColor=white" alt="Python Version"></a>
    <a href="#installation"><img src="https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey?style=flat-square" alt="Platform"></a>
  </p>

  <div>
    <img src="https://img.shields.io/badge/GPU-Accelerated-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="GPU Accelerated">
    <img src="https://img.shields.io/badge/API-OpenAI%20Compatible-000000?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI Compatible">
    <img src="https://img.shields.io/badge/Models-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white" alt="Hugging Face">
  </div>
</div>

---

**Navigation**

*   [✨ Overview](https://github.com/HelpingAI/inferno#-overview)
*   [🚀 Key Features](https://github.com/HelpingAI/inferno#-key-features)
*   [⚙️ Installation](#️-installation)
    *   [Hardware Acceleration (Critical Prerequisite)](#hardware-acceleration-llama-cpp-python-critical-prerequisite)
*   [🖥️ Command Line Interface (CLI)](#️-command-line-interface-cli)
*   [🔥 Getting Started](https://github.com/HelpingAI/inferno#-getting-started)
*   [📋 Usage Guide](https://github.com/HelpingAI/inferno#-usage-guide)
    *   [Download Models](#download-a-model)
    *   [Quantization](#model-quantization)
    *   [List Models](#list-downloaded-models)
    *   [Start the Server](#start-the-server)
    *   [Chat (CLI)](#chat-with-a-model)
*   [🔌 API Usage](https://github.com/HelpingAI/inferno#-api-usage)
    *   [OpenAI Compatible](#openai-api-endpoints)
    *   [Ollama Compatible](#ollama-api-endpoints)
    *   [Python Examples](#python-examples)
*   [🐍 Native Python Client](https://github.com/HelpingAI/inferno#-native-python-client)
*   [🧩 Integrations](https://github.com/HelpingAI/inferno#-integration-with-applications)
*   [📦 Requirements](https://github.com/HelpingAI/inferno#-requirements)
*   [🔧 Advanced Configuration](https://github.com/HelpingAI/inferno#-advanced-configuration)
*   [🤝 Contributing](https://github.com/HelpingAI/inferno#-contributing)
*   [📄 License](https://github.com/HelpingAI/inferno#-license)
*   [📚 Full Documentation](https://github.com/HelpingAI/inferno#-full-documentation)

---

## ✨ Overview

Inferno is your personal gateway to the blazing frontier of Artificial Intelligence. Designed for both newcomers and seasoned developers, it provides a powerful yet user-friendly platform to run the latest Large Language Models (LLMs) directly on your local machine. Experience the raw power of models like Llama 3.3 and Phi-4 without relying on cloud services, ensuring full control over your data and costs.

Inferno offers an experience similar to Ollama but turbo-charged with enhanced features, including seamless Hugging Face integration, advanced quantization tools, and flexible model management. Its OpenAI & Ollama-compatible APIs ensure drop-in compatibility with your favorite AI frameworks and tools.

> [!TIP]
> New to local LLMs? Inferno makes it incredibly easy to get started. Pull a model and ignite your first conversation within minutes!

## 🚀 Key Features

- **Bleeding-Edge Model Support:** Run the latest models such as Llama 3.3, DeepSeek-R1, Phi-4, Gemma 3, Mistral Small 3.1, and more as soon as GGUF versions are available.

- **Hugging Face Integration:** Download models with interactive file selection, repository browsing, and direct `repo_id:filename` targeting.

- **Dual API Compatibility:** Serve models through both OpenAI and Ollama compatible API endpoints. Use Inferno with almost any AI client or framework.

- **Native Python Client:** Includes a built-in, OpenAI-compatible Python client for seamless integration into your Python projects. Supports streaming, embeddings, multimodal inputs, and tool calling.

- **Interactive CLI:** Command-line interface for downloading, managing, quantizing, and chatting with models.

- **Blazing-Fast Inference:** GPU acceleration (CUDA, Metal, ROCm, Vulkan, SYCL) for faster response times. CPU acceleration via OpenBLAS is also supported.

- **Real-time Streaming:** Get instant feedback with streaming support for both chat and completions APIs.

- **Flexible Context Control:** Adjust the context window size (`n_ctx`) per model or session. Max context length is automatically detected from GGUF metadata.

- **Smart Model Management:** List, show details, copy, remove, and see running models (`ps`). Includes RAM requirement estimates.

- **Embeddings Generation:** Create embeddings using your local models via the API.

- **Advanced Quantization:** Convert models between various GGUF quantization levels (including importance matrix methods like `iq4_nl`) with interactive comparison and RAM estimates.

- **Keep-Alive Management:** Control how long models stay loaded in memory when idle.

- **Fine-Grained Configuration:** Customize inference parameters such as GPU layers, threads, batch size, RoPE settings, and mlock.

## ⚙️ Installation

> [!IMPORTANT]
> **Critical Prerequisite: Install `llama-cpp-python` First!**
> Inferno relies heavily on `llama-cpp-python`. For optimal performance, especially GPU acceleration, you **MUST** install `llama-cpp-python` with the correct hardware backend flags *before* installing Inferno. Failure to do this may result in suboptimal performance or CPU-only operation.

### 1. Install `llama-cpp-python` with Hardware Acceleration

Choose **one** of the following commands based on your hardware. See the detailed [Hardware Acceleration](#hardware-acceleration-llama-cpp-python-critical-prerequisite) section below for more options and explanations.

*   **NVIDIA GPU (CUDA):**
    ```bash
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
    # Or use pre-built wheels if available (see details below)
    ```
*   **Apple Silicon GPU (Metal):**
    ```bash
    CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
    # Or use pre-built wheels if available (see details below)
    ```
*   **AMD GPU (ROCm):**
    ```bash
    CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
    ```
*   **CPU Only (OpenBLAS):**
    ```bash
    CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
    ```
*   **Other Backends (Vulkan, SYCL, etc.):** See the detailed section below.

> [!TIP]
> Using a virtual environment (like `venv` or `conda`) is highly recommended. Ensure you have Python 3.9+ and the necessary build tools (CMake, C++ compiler) installed. Adding `--force-reinstall --upgrade --no-cache-dir` helps ensure a clean build against your system's libraries.

### 2. Install Inferno

Once `llama-cpp-python` is installed with your desired backend, you can install Inferno directly from PyPI:

```bash
# Install the latest stable release from PyPI
pip install inferno-llm
```

Or, for development or the latest features, install from source:

```bash
# Clone the Inferno repository
git clone https://github.com/HelpingAI/inferno.git
cd inferno

# Install Inferno in editable mode (recommended for development)
pip install -e .

# Or install with all optional dependencies (like quantization tools)
# pip install -e ".[dev]"
```

### Hardware Acceleration (`llama-cpp-python` Critical Prerequisite)

`llama.cpp` (the engine behind `llama-cpp-python`) supports multiple hardware acceleration backends. You need to tell `pip` how to build `llama-cpp-python` using `CMAKE_ARGS`.

<details>
<summary>How to Set Build Options (Environment Variables vs. CLI)</summary>

You can set `CMAKE_ARGS` either as an environment variable before running `pip install` or directly via the `-C / --config-settings` flag.

**Environment Variable Method (Linux/macOS):**
```bash
CMAKE_ARGS="-DOPTION=on" pip install llama-cpp-python ...
```

**Environment Variable Method (Windows PowerShell):**
```powershell
$env:CMAKE_ARGS = "-DOPTION=on"
pip install llama-cpp-python ...
```

**CLI Method (Works Everywhere, Good for requirements.txt):**
```bash
# Use semicolons to separate multiple CMake args with -C
pip install llama-cpp-python -C cmake.args="-DOPTION1=on;-DOPTION2=off" ...
```
</details>

<details open>
<summary>Supported Backends (Install ONE)</summary>

*   **CUDA (NVIDIA):** Requires NVIDIA drivers & CUDA Toolkit.
    ```bash
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
    ```
    *   **Pre-built Wheels (Alternative):** If you have CUDA 12.1-12.5 and Python 3.10-3.12, try:
        ```bash
        # Replace <cuda-version> with cu121, cu122, cu123, cu124, or cu125
        pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir \
          --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/<cuda-version>
        # Example: pip install ... --extra-index-url .../whl/cu121
        ```

*   **Metal (Apple Silicon):** Requires macOS 11.0+ & Xcode Command Line Tools.
    ```bash
    CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
    ```
    *   **Pre-built Wheels (Alternative):** If you have macOS 11.0+ and Python 3.10-3.12, try:
        ```bash
        pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir \
          --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
        ```

*   **hipBLAS / ROCm (AMD):** Requires ROCm toolkit.
    ```bash
    CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
    ```

*   **OpenBLAS (CPU Acceleration):** Recommended for CPU-only systems. Requires OpenBLAS library installed.
    ```bash
    CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
    ```

*   **Vulkan:** Requires Vulkan SDK. May accelerate various GPUs (Intel, AMD).
    ```bash
    CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
    ```

*   **SYCL (Intel GPU):** Requires Intel oneAPI Base Toolkit.
    ```bash
    # Set up oneAPI environment first (adjust path as needed)
    source /opt/intel/oneapi/setvars.sh
    CMAKE_ARGS="-DGGML_SYCL=on -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
    ```

*   **RPC (Distributed):** For multi-machine inference setups.
    ```bash
    CMAKE_ARGS="-DGGML_RPC=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
    ```

</details>

## 🖥️ Command Line Interface (CLI)

Access Inferno's features through its intuitive CLI:

```bash
# Show available commands and options
inferno --help

# Alternatively, run as a Python module
python -m inferno --help
```

**Core Commands:**

| Command                    | Description                                          | Example                                                   |
| :------------------------- | :--------------------------------------------------- | :-------------------------------------------------------- |
| `pull <model_id_or_path>`  | Download models (GGUF) from Hugging Face             | `inferno pull meta-llama/Llama-3.3-8B-Instruct-GGUF`        |
| `list` or `ls`             | List locally downloaded models & RAM estimates       | `inferno list`                                            |
| `serve <model_name_or_id>` | Start API server (OpenAI & Ollama compatible)        | `inferno serve MyLlama3 --port 8080`                      |
| `run <model_name_or_id>`   | Start interactive chat session in the terminal     | `inferno run MyLlama3`                                    |
| `remove <model_name>`      | Delete a downloaded model                            | `inferno remove MyLlama3`                                 |
| `copy <source> <dest>`     | Duplicate a model locally                            | `inferno copy MyLlama3 MyLlama3-Experiment`               |
| `show <model_name>`        | Display detailed model info (metadata, path, etc.)   | `inferno show MyLlama3`                                   |
| `ps`                       | Show running Inferno server processes/models         | `inferno ps`                                              |
| `quantize <input> [out]`   | Convert models (HF or GGUF) to different quant levels | `inferno quantize hf:Qwen/Qwen3-0.6B Qwen3-0.6B-Q4_K_M` |
| `compare <models...>`      | Compare specs of multiple local models               | `inferno compare ModelA ModelB`                           |
| `estimate <model_name>`    | Show RAM usage estimates for different quants        | `inferno estimate MyLlama3-f16`                           |
| `version`                  | Display Inferno version information                  | `inferno version`                                         |

## 🔥 Getting Started

Let's ignite your first model!

1.  **Download a Model:** Choose a model from Hugging Face (GGUF format). Inferno helps you select the specific file.
    ```bash
    # Example: Download Llama 3.3 8B Instruct (will prompt for file selection)
    inferno pull meta-llama/Llama-3.3-8B-Instruct-GGUF

    # Example: Download Mistral Small 3.1
    inferno pull mistralai/Mistral-Small-3.1-GGUF

    # Example: Download Phi-4 Mini
    inferno pull microsoft/Phi-4-mini-GGUF

    # Example: Specify a direct file if you know the exact name
    # inferno pull user/repo-GGUF:model-q4_k_m.gguf
    ```
    > [!WARNING]
    > Some models require Hugging Face authentication. Run `huggingface-cli login` in your terminal beforehand if needed. Inferno will warn you about estimated RAM requirements.

2.  **List Your Models:** Verify the download.
    ```bash
    inferno list
    ```
    *(You'll see your downloaded model listed, e.g., `Llama-3.3-8B-Instruct-GGUF`)*

3.  **Chat with the Model:** Start an interactive session.
    ```bash
    inferno run Llama-3.3-8B-Instruct-GGUF
    ```
    Type your questions and press Enter. Use `/help` inside the chat for commands like changing the system prompt (`/set system ...`) or context size (`/set context ...`). Use `/bye` to exit.

4.  **(Alternative) Start the API Server:** Serve the model for use with other applications.
    ```bash
    inferno serve Llama-3.3-8B-Instruct-GGUF --port 8000
    ```
    The server will be available at `http://localhost:8000`. You can now use clients pointing to `http://localhost:8000/v1` (OpenAI API) or `http://localhost:8000/api` (Ollama API).

## 📋 Usage Guide

### Download a Model

```bash
# Interactive download (recommended)
inferno pull <repo_id>
# Example: inferno pull google/gemma-1.1-7b-it-gguf

# Direct file download
inferno pull <repo_id>:<filename.gguf>
# Example: inferno pull google/gemma-1.1-7b-it-gguf:gemma-1.1-7b-it-Q4_K_M.gguf
```
Inferno shows available GGUF files, sizes, and estimated RAM needed, warning if it exceeds your system's available RAM.

### Model Quantization

Convert models to smaller, faster GGUF formats. This is useful if you download a large `F16` model or want to experiment with different precision levels.

```bash
# Quantize a downloaded F16 GGUF model (interactive method selection)
inferno quantize MyModel-f16 MyModel-Q4_K_M

# Quantize directly from a Hugging Face repo (interactive)
# This downloads the original (often PyTorch/Safetensors) model and converts it.
inferno quantize hf:NousResearch/Hermes-2-Pro-Llama-3-8B Hermes-2-Pro-Llama-3-8B-Q5_K_M

# Specify quantization method directly (e.g., q4_k_m)
inferno quantize MyModel-f16 MyModel-Q4_K_M --method q4_k_m
```

**Common Quantization Methods:**

| Method | Approx Bits | Size Multiplier (vs F16) | Use Case                                    |
| :----- | :---------- | :----------------------- | :------------------------------------------ |
| q2_k   | ~2.5 bits   | ~0.16x                   | Minimum RAM, experimental quality           |
| iq3_m  | ~3.0 bits   | ~0.21x                   | Good quality for 3-bit (Importance Matrix)  |
| q3_k_m | ~3.5 bits   | ~0.24x                   | Balanced low RAM / decent quality           |
| iq4_nl | ~4.0 bits   | ~0.29x                   | Best 4-bit quality (Non-Linear Importance)  |
| iq4_xs | ~4.0 bits   | ~0.29x                   | Extra small 4-bit (Importance Matrix)       |
| q4_k_m | ~4.5 bits   | ~0.31x                   | **Excellent general-purpose balance**       |
| q5_k_m | ~5.5 bits   | ~0.38x                   | Higher quality, moderate RAM increase       |
| q6_k   | ~6.5 bits   | ~0.44x                   | Very high quality, significant RAM          |
| q8_0   | ~8.5 bits   | ~0.53x                   | Near-lossless quality, highest non-F16 RAM |
| f16    | 16.0 bits   | 1.00x                    | Full precision, highest RAM, source quality |

> [!NOTE]
> RAM estimates provided during quantization are approximate. Actual usage depends on context size and backend. Use `inferno estimate <model_name>` for more detailed projections.

### List Downloaded Models

```bash
inferno list # or inferno ls
```
Shows local model names, original repo, file size, quantization type, estimated base RAM, and download date.

### Start the Server

```bash
# Serve a locally downloaded model
inferno serve MyModel-Q4_K_M

# Serve directly from Hugging Face (downloads if not present)
inferno serve teknium/OpenHermes-2.5-Mistral-7B-GGUF:openhermes-2.5-mistral-7b.Q4_K_M.gguf

# Specify host and port (0.0.0.0 makes it accessible on your network)
inferno serve MyModel-Q4_K_M --host 0.0.0.0 --port 8080

# Advanced: Offload layers to GPU, set context size
inferno serve MyModel-Q4_K_M --n_gpu_layers 35 --n_ctx 8192
```
> [!WARNING]
> Using `--host 0.0.0.0` exposes the server to your local network. Ensure your firewall settings are appropriate.

### Chat with a Model

```bash
inferno run MyModel-Q4_K_M

# Set context size on launch
inferno run MyModel-Q4_K_M --n_ctx 4096
```
**In-Chat Commands:**

| Command                 | Description                             |
| :---------------------- | :---------------------------------------------------- |
| `/help` or `/?`         | Show this help message                                |
| `/bye`                  | Exit the chat                                         |
| `/set system <prompt>`  | Set the system prompt                                 |
| `/set context <size>`   | Set context window size (reloads model)               |
| `/show context`         | Show the current and maximum context window size      |
| `/clear` or `/cls`      | Clear the terminal screen                             |
| `/reset`                | Reset chat history and system prompt                  |

## 🔌 API Usage

Inferno exposes OpenAI and Ollama compatible API endpoints when you run `inferno serve`.

*   **OpenAI Base URL:** `http://localhost:8000/v1` (Default port)
*   **Ollama Base URL:** `http://localhost:8000/api` (Default port)

### OpenAI API Endpoints

*   `/v1/models` (GET): List available models (returns the currently served model).
*   `/v1/chat/completions` (POST): Generate chat responses (supports streaming).
*   `/v1/completions` (POST): Generate text completions (legacy, use chat).
*   `/v1/embeddings` (POST): Generate text embeddings.

### Ollama API Endpoints

*   `/api/tags` (GET): List available models (returns the currently served model).
*   `/api/chat` (POST): Generate chat responses (supports streaming).
*   `/api/generate` (POST): Generate text completions.
*   `/api/embed` (POST): Generate text embeddings.
*   `/api/show` (POST): Show details for a loaded model.
*   *(Model management endpoints like `/api/pull`, `/api/copy`, `/api/delete` are generally handled by the CLI)*

### Python Examples

#### Using `openai` Package

```python
import openai

# Point the official OpenAI client to your Inferno server
client = openai.OpenAI(
    api_key="dummy-key", # Required by the library, but not used by Inferno
    base_url="http://localhost:8000/v1" # Your Inferno server URL
)

# --- Chat Completion ---
try:
    response = client.chat.completions.create(
        model="MyModel-Q4_K_M", # Must match the model name used in `inferno serve`
        messages=[
            {"role": "system", "content": "You are Inferno, a helpful AI assistant."},
            {"role": "user", "content": "Explain the concept of quantization."}
        ],
        max_tokens=150,
        temperature=0.7,
        stream=False # Set to True for streaming
    )
    print("Full Response:")
    print(response.choices[0].message.content)

except openai.APIConnectionError as e:
    print(f"Connection Error: Is the Inferno server running? {e}")
except Exception as e:
    print(f"An error occurred: {e}")


# --- Streaming Chat Completion ---
try:
    print("\nStreaming Response:")
    stream = client.chat.completions.create(
        model="MyModel-Q4_K_M",
        messages=[{"role": "user", "content": "Write a short poem about fire."}],
        stream=True
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print() # Newline after stream

except openai.APIConnectionError as e:
    print(f"Connection Error: Is the Inferno server running? {e}")
except Exception as e:
    print(f"An error occurred: {e}")

# --- Embeddings ---
try:
    response = client.embeddings.create(
        model="MyModel-Q4_K_M", # Ensure the model supports embeddings
        input="Inferno is heating up the local AI scene!"
    )
    print(f"\nEmbedding Vector (first 5 dims): {response.data[0].embedding[:5]}...")
    print(f"Total dimensions: {len(response.data[0].embedding)}")

except openai.APIConnectionError as e:
    print(f"Connection Error: Is the Inferno server running? {e}")
except Exception as e:
    print(f"An error occurred: {e}")

```

#### Using `requests` (Ollama API)

```python
import requests
import json

base_url = "http://localhost:8000/api" # Ollama API base
model_name = "MyModel-Q4_K_M" # Replace with your served model

# --- Chat Completion ---
try:
    response = requests.post(
        f"{base_url}/chat",
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": "Hello there!"}],
            "stream": False
        }
    )
    response.raise_for_status() # Raise an exception for bad status codes
    print("Ollama API Chat Response:")
    print(response.json()["message"]["content"])

except requests.exceptions.RequestException as e:
    print(f"Ollama API Error: {e}")


# --- Embeddings ---
try:
    response = requests.post(
        f"{base_url}/embed",
        json={
            "model": model_name,
            "input": "This is text to embed."
        }
    )
    response.raise_for_status()
    print("\nOllama API Embedding (first 5 dims):")
    print(response.json()["embeddings"][:5],"...")
    print(f"Total dimensions: {len(response.json()['embeddings'])}")

except requests.exceptions.RequestException as e:
    print(f"Ollama API Error: {e}")

```

## 🐍 Native Python Client

Inferno includes its own `InfernoClient`, a drop-in replacement for the official `openai` client, offering the same interface.

```python
# Ensure inferno is installed: pip install -e .
from inferno.client import InfernoClient
import json # For parsing tool arguments if needed

# Initialize pointing to your server
client = InfernoClient(
    api_key="dummy",
    base_url="http://localhost:8000/v1", # Use the OpenAI-compatible endpoint
)

model_name = "MyModel-Q4_K_M" # Replace with your served model

# --- Basic Chat ---
print("--- Native Client Chat ---")
response = client.chat.create(
    model=model_name,
    messages=[{"role": "user", "content": "What is Inferno?"}],
)
print(response["choices"][0]["message"]["content"])


# --- Multimodal Chat (Example - Requires a multimodal model like LLaVA) ---
# Make sure you are serving a model that supports image input (e.g., a LLaVA GGUF)
# model_name_multimodal = "llava-v1.6-mistral-7b-GGUF" # Example name
# print("\n--- Native Client Multimodal Chat (Requires Multimodal Model) ---")
# try:
#     response = client.chat.create(
#         model=model_name_multimodal, # Use the multimodal model name
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": "Describe this image."},
#                     {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}}
#                 ]
#             }
#         ]
#     )
#     print(response.get("choices", [{}])[0].get("message", {}).get("content"))
# except Exception as e:
#     print(f"Could not run multimodal example: {e}")


# --- Tool Calling (Example - Requires a model supporting tools/functions) ---
# Make sure you are serving a model fine-tuned for tool/function calling
# model_name_tools = "hermes-2-pro-llama-3-8b-GGUF" # Example name
# print("\n--- Native Client Tool Calling (Requires Tool-Supporting Model) ---")
# tools = [{
#     "type": "function",
#     "function": {
#         "name": "get_current_weather",
#         "description": "Get the current weather in a given location",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
#                 "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
#             },
#             "required": ["location"]
#         }
#     }
# }]
# try:
#     response = client.chat.create(
#         model=model_name_tools, # Use the tool-supporting model name
#         messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
#         tools=tools,
#         tool_choice="auto" # or force: {"type": "function", "function": {"name": "get_current_weather"}}
#     )
#     message = response.get("choices", [{}])[0].get("message", {})
#     if message.get("tool_calls"):
#         tool_call = message["tool_calls"][0]
#         function_name = tool_call["function"]["name"]
#         function_args = json.loads(tool_call["function"]["arguments"])
#         print(f"Function Call Requested: {function_name}")
#         print(f"Arguments: {function_args}")
#         # --- Here you would execute the function and send back the result ---
#     else:
#         print(f"Response Content (No Tool Call): {message.get('content')}")
# except Exception as e:
#     print(f"Could not run tool calling example: {e}")

```
> [!TIP]
> The `InfernoClient` provides a familiar interface for developers already using the `openai` package, simplifying integration.

## 🧩 Integration with Applications

Inferno's OpenAI API compatibility makes it easy to integrate with popular AI frameworks.

```python
# Example using LangChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Configure LangChain to use your local Inferno server
llm = ChatOpenAI(
    model="MyModel-Q4_K_M", # Your served model name
    openai_api_key="dummy",
    openai_api_base="http://localhost:8000/v1",
    temperature=0.7,
    streaming=True # Enable streaming if desired
)

print("--- LangChain Integration Example ---")
# Simple invocation
# response = llm.invoke([HumanMessage(content="Explain the difference between heat and temperature.")])
# print(response.content)

# Streaming invocation
print("Streaming with LangChain:")
for chunk in llm.stream("Write a haiku about a campfire."):
    print(chunk.content, end="", flush=True)
print()
```
Works similarly with LlamaIndex, Semantic Kernel, Haystack, and any tool supporting the OpenAI API standard. Just point them to your Inferno server's URL (`http://localhost:8000/v1`).

## 📦 Requirements

### Hardware

*   **RAM:** This is the most critical factor.
    *   ~2-4 GB RAM for 1-3B parameter models (e.g., Phi-3 Mini, Gemma 2B)
    *   **8 GB+ RAM** recommended for 7-8B models (e.g., Llama 3.3 8B, Mistral 7B)
    *   **16 GB+ RAM** recommended for 13B models
    *   **32 GB+ RAM** needed for ~30B models
    *   **64 GB+ RAM** needed for ~70B models
*   **CPU:** A modern multi-core CPU. Performance scales with core count and speed.
*   **GPU (Highly Recommended):** An NVIDIA, AMD, or Apple Silicon GPU significantly accelerates inference. VRAM requirements depend on the model size and number of layers offloaded (`--n_gpu_layers`). Even partial offloading helps.
*   **Disk Space:** Enough space for downloaded models (GGUF files can range from ~1GB to 100GB+).

> [!WARNING]
> Running models requiring more RAM than physically available will lead to *extreme* slowdowns due to disk swapping. Check model RAM estimates (`inferno list`, `inferno estimate`) before running.

### Software

*   **Python:** 3.9 or newer.
*   **Build Tools:** A C++ compiler (like GCC, Clang, or MSVC) and CMake are required for building `llama-cpp-python`.
*   **Core Dependencies:** `llama-cpp-python`, `fastapi`, `uvicorn`, `rich`, `typer`, `huggingface-hub`, `pydantic`, `requests`. (Installed automatically with `pip install -e .`)
*   **(Optional) Git:** For cloning the repository.

## 🔧 Advanced Configuration

Pass these options to `inferno serve` and/or `inferno run` as indicated:

| Option              | Description                                                      | Example                               | Default            |
| :------------------ | :--------------------------------------------------------------- | :------------------------------------ | :----------------- |
| `--host <ip>`       | IP address to bind the server to                                 | `--host 0.0.0.0`                      | `127.0.0.1`        | `serve` only   |
| `--port <num>`      | Port for the API server                                          | `--port 8080`                         | `8000`             | `serve` only   |
| `--n_gpu_layers <n>`| Number of model layers to offload to GPU (-1 for max)            | `--n_gpu_layers 35`                   | `0` (CPU only)     | `serve`, `run` |
| `--n_ctx <n>`       | Context window size (tokens), overrides auto-detection           | `--n_ctx 8192`                        | Auto/4096          | `serve`, `run` |
| `--n_threads <n>`   | Number of CPU threads for computation                            | `--n_threads 4`                       | (Auto-detected)    | `serve`, `run` |
| `--use_mlock`       | Force model to stay in RAM (prevents swapping if possible)       | `--use_mlock`                         | (Disabled)         | `serve`, `run` |

> [!TIP]
> For optimal CPU performance, set `--n_threads` to the number of *physical* cores on your CPU. Check your CPU specs (e.g., via Task Manager on Windows or `lscpu` on Linux). Start with `--n_gpu_layers -1` to offload as much as possible to VRAM, then reduce if you encounter memory errors.

## 🤝 Contributing

Help fuel the fire! Contributions are highly welcome.

1.  **Fork** the repository on GitHub.
2.  **Clone** your fork locally: `git clone https://github.com/HelpingAI/inferno.git`
3.  Create a **new branch** for your changes: `git checkout -b feature/my-cool-feature` or `bugfix/fix-that-issue`.
4.  Make your changes, **commit** them with clear messages: `git commit -m "Add feature X"`
5.  **Push** your branch to your fork: `git push origin feature/my-cool-feature`
6.  Open a **Pull Request** (PR) from your branch to the `main` branch of the `HelpingAI/inferno` repository.

Please ensure your code follows basic Python best practices and includes relevant tests or documentation updates if applicable.

## 📄 License

Inferno is licensed under the [HelpingAI Open Source License](LICENSE). This license promotes open innovation and collaboration while ensuring responsible and ethical use of AI technology.

## 📚 Full Documentation

<div align="center">
  <h3><a href="https://deepwiki.com/HelpingAI/inferno">📖 Dive Deeper: Read the Full Documentation</a></h3>
  <p>Find comprehensive guides, API references, advanced configuration details, and tutorials at <code>deepwiki.com/HelpingAI/inferno</code></p>
</div>

---

<div align="center">
  <p>Made with ❤️ by <a href="https://helpingai.co">HelpingAI</a></p>
</div>