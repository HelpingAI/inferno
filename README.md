<div align="center">
  <img src="https://img.shields.io/badge/Inferno-Local%20LLM%20Server-orange?style=for-the-badge&logo=python&logoColor=white" alt="Inferno Logo">

  <h1>Inferno</h1>

  <p><strong>A powerful llama-cpp-python based LLM serving tool</strong></p>

  <p>
    Run local LLMs with an OpenAI-compatible API, interactive CLI, and seamless Hugging Face integration.
  </p>

  <!-- Badges -->
  <p>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-HelpingAI%20Open%20Source-blue?style=flat-square" alt="License"></a>
    <a href="#"><img src="https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python&logoColor=white" alt="Python Version"></a>
    <a href="#"><img src="https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey?style=flat-square" alt="Platform"></a>
  </p>
</div>

> [!NOTE]
> Inferno automatically sets context length to 4096 tokens by default. You can adjust this with the `/set context` command in chat mode.

<div align="center">
  <img src="https://img.shields.io/badge/GPU-Accelerated-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="GPU Accelerated">
  <img src="https://img.shields.io/badge/API-OpenAI%20Compatible-000000?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI Compatible">
  <img src="https://img.shields.io/badge/Models-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white" alt="Hugging Face">
</div>

## 📚 Documentation

<div align="center">
  <h3><a href="https://deepwiki.com/HelpingAI/inferno">📖 Read the Full Documentation</a></h3>
  <p>Comprehensive guides, API references, and examples available at <code>deepwiki.com/HelpingAI/inferno</code></p>
</div>

## ✨ Overview

Inferno is a powerful tool for running Large Language Models (LLMs) locally on your machine. It provides an experience similar to Ollama but with enhanced features and flexibility. Inferno makes it easy to download, manage, and use GGUF models from Hugging Face with an intuitive command-line interface and API compatibility with popular tools.

## 🚀 Key Features

- **🤗 Hugging Face Integration:** Download models directly with interactive file selection and repository browsing
- **🔄 Flexible Model Specification:** Support for `repo_id:filename` format for direct file targeting
- **🔌 OpenAI & Ollama Compatible APIs:** Use with any client that supports these APIs
- **🐍 Native Python Client:** Built-in OpenAI-compatible Python client for seamless integration
- **💬 Interactive CLI:** Powerful command-line interface for model management and chat
- **⚡ Streaming Support:** Real-time streaming responses for chat and completions
- **🖥️ GPU Acceleration:** Utilize GPU for faster inference when available
- **📏 Context Window Control:** Adjust context size for different models and use cases
- **🧠 Model Management:** Copy, show details, and list running models
- **📊 Embeddings Support:** Generate embeddings from models
- **⚠️ RAM Requirement Warnings:** Automatic warnings about RAM requirements for different model sizes
- **🔍 Max Context Detection:** Automatically detects and displays maximum context length from GGUF files
- **📈 Quantization Tools:** Convert models between different quantization levels with visual comparison and importance matrix support
- **🔄 Keep-Alive Control:** Configure model unloading behavior with keep-alive settings
- **🛠️ Advanced Configuration:** Set custom parameters like threads, batch size, and RoPE settings

## ⚙️ Installation

Install Inferno directly from source:

```bash
# Clone the repository
git clone https://github.com/HelpingAI/inferno.git
cd inferno

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[dev]"
```

## 🖥️ Command Line Interface

Inferno provides a powerful command-line interface for managing and using LLMs:

```bash
# Show available commands
inferno --help

# Using as a Python module
python -m inferno --help
```

| Command | Description |
|---------|-------------|
| `inferno pull <model>` | Download a model from Hugging Face |
| `inferno list` | List downloaded models with RAM requirements |
| `inferno serve <model>` | Start a model server with OpenAI & Ollama compatible APIs |
| `inferno run <model>` | Chat with a model interactively |
| `inferno remove <model>` | Remove a downloaded model |
| `inferno copy <source> <dest>` | Copy a model to a new name |
| `inferno show <model>` | Show detailed model information |
| `inferno ps` | List running models |
| `inferno quantize <model> <output>` | Quantize a model to a different format |
| `inferno compare <models...>` | Compare multiple models (size, metrics) |
| `inferno estimate <model>` | Show RAM usage estimates for quantization |
| `inferno version` | Show version information |

## 📋 Usage Guide

### Download a Model

```bash
# Download a model from Hugging Face (interactive file selection)
inferno pull Abhaykoul/HAI3-raw-Q4_K_M-GGUF

# Download a specific file using repo_id:filename format
inferno pull Abhaykoul/HAI3-raw-Q4_K_M-GGUF:hai3-raw-q4_k_m.gguf
```

When downloading models, Inferno will:
- Show available GGUF files in the repository
- Display file sizes and RAM requirements
- Show maximum context length for each model
- Provide a comparison of RAM usage by quantization type
- Warn if your system has insufficient RAM

### Model Quantization

Inferno provides an interactive quantization interface:

```bash
# Quantize a HuggingFace model (interactive)
inferno quantize hf:Qwen/Qwen3-0.6B

# The command will:
# 1. Show available methods with RAM estimates
# 2. Let you select the preferred method
# 3. Download and convert the model
# 4. Save in the inferno models directory
```

#### Interactive Quantization UI

When you run the quantize command, you'll see:

1. A table of available methods showing:
   - Method name (e.g., q4_k_m)
   - Bits per parameter
   - RAM multiplier (e.g., 1.40× model size)
   - Description and use case

2. RAM Usage Examples:
   For a 3GB gguf model file:
   - q2_k: ~3.45GB RAM (3GB × 1.15)
   - q4_k_m: ~4.20GB RAM (3GB × 1.40)
   - q8_0: ~6.00GB RAM (3GB × 2.00)
   - f16: ~8.40GB RAM (3GB × 2.80)

#### Available Methods

| Method | Bits/Param | RAM Usage | Best For |
|--------|------------|-----------|----------|
| q2_k   | ~2.5 bits | 1.15× size| Minimum RAM usage, lower quality |
| q3_k_m | ~3.5 bits | 1.28× size| Good balance of RAM/quality |
| q4_k_m | ~4.5 bits | 1.40× size| Best general-purpose choice |
| q5_k_m | ~5.5 bits | 1.65× size| Better quality, more RAM |
| q6_k   | ~6.5 bits | 1.80× size| High quality, high RAM |
| q8_0   | ~8.5 bits | 2.00× size| Very high quality |
| f16    | 16.0 bits | 2.80× size| Maximum quality, highest RAM |

RAM usage consists of two parts:

#### RAM Usage Calculation (Recommended)

> [!NOTE]
> The following RAM usage estimates are based on how the Hugging Face `transformers` library loads models in FP16 (float16) precision. We use the FP16 model file size as the baseline for these calculations in this README. Actual RAM usage may vary depending on backend and quantization.

1. **Base Model RAM:**  
    The FP16 model file size is roughly 2× the number of parameters in billions:
    - 1B parameters ≈ 2GB (FP16) file size
    - 3B parameters ≈ 6GB (FP16) file size
    - 7B parameters ≈ 14GB (FP16) file size

    Multiply by quantization factor for estimated RAM:
    - 2GB (gguf model) × 1.40 (q4_k_m) ≈ **2.8GB estimated RAM**
    - 6GB (gguf model) × 1.40 (q4_k_m) ≈ **8.4GB estimated RAM**

2. **Context RAM:**  
    Additional RAM is needed for the context window (per billion parameters):
    - 4K context ≈ **+0.2GB RAM**
    - 8K context ≈ **+0.4GB RAM**
    - 16K context ≈ **+0.8GB RAM**
    - 32K context ≈ **+1.6GB RAM**

> [!NOTE]
> You can run models on systems with less RAM than recommended, but expect slower performance and possible swapping to disk.


#### Importance Matrix Quantization

For better quality at the same size, use importance matrix quantization:

| Method | Description |
|--------|-------------|
| iq3_m  | 3-bit importance-weighted |
| iq4_nl | 4-bit non-linear (best accuracy) |
| iq4_xs | 4-bit extra small size |

### List Downloaded Models

```bash
inferno list
```

The list command shows:
- Model names and repositories
- File sizes and quantization types
- RAM requirements (color-coded based on your system's RAM)
- Download dates
- Quantization comparison table

### Start the Server

```bash
# Start the server with a downloaded model
inferno serve HAI3-raw-Q4_K_M-GGUF

# Start the server with a model from Hugging Face (downloads if needed)
inferno serve Abhaykoul/HAI3-raw-Q4_K_M-GGUF

# Specify host and port
inferno serve HAI3-raw-Q4_K_M-GGUF --host 0.0.0.0 --port 8080
```

The server provides:
- OpenAI-compatible API endpoints (/v1/...)
- Ollama-compatible API endpoints (/api/...)
- Support for chat completions, text completions, and embeddings
- Streaming responses
- Automatic model loading and unloading

### Chat with a Model

```bash
inferno run HAI3-raw-Q4_K_M-GGUF
```

#### Available Chat Commands

| Command | Description |
|---------|-------------|
| `/help` or `/?` | Show available commands |
| `/bye` | Exit the chat |
| `/set system <prompt>` | Set the system prompt (use quotes for multi-word prompts) |
| `/set context <size>` | Set context window size (default: 4096) |
| `/clear` or `/cls` | Clear the terminal screen |
| `/reset` | Reset all settings |

## 🔌 API Usage

Inferno provides both OpenAI-compatible and Ollama-compatible APIs. You can use it with any client that supports either API.

### OpenAI API Endpoints

- `/v1/models` - List available models
- `/v1/chat/completions` - Create chat completions
- `/v1/completions` - Create text completions
- `/v1/embeddings` - Generate embeddings

### Ollama API Endpoints

- `/api/chat` - Create chat completions
- `/api/generate` - Create text completions
- `/api/embed` - Generate embeddings
- `/api/tags` - List available models
- `/api/show` - Show model details
- `/api/copy` - Copy a model
- `/api/delete` - Delete a model
- `/api/pull` - Pull a model

### Python Example (OpenAI API)

```python
import openai

# Configure the client
openai.api_key = "dummy"  # Not used but required
openai.api_base = "http://localhost:8000/v1"  # Default Inferno API URL

# Chat completion
response = openai.ChatCompletion.create(
    model="HAI3-raw-Q4_K_M-GGUF",  # Use the model name
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message.content)

# Streaming chat completion
for chunk in openai.ChatCompletion.create(
    model="HAI3-raw-Q4_K_M-GGUF",
    messages=[
        {"role": "user", "content": "Tell me a joke"}
    ],
    stream=True
):
    if hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content"):
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## 🧩 Integration with Applications

Inferno can be easily integrated with various applications that support the OpenAI API format:

```python
# Example with LangChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Configure to use local Inferno server with OpenAI API
chat = ChatOpenAI(
    model_name="HAI3-raw-Q4_K_M-GGUF",
    openai_api_key="dummy",
    openai_api_base="http://localhost:8000/v1",
    streaming=True
)

# Use the model
response = chat([HumanMessage(content="Explain quantum computing in simple terms")])
print(response.content)
```

### Ollama API Example

```python
import requests
import json

# Chat completion with Ollama API
response = requests.post(
    "http://localhost:8000/api/chat",
    json={
        "model": "HAI3-raw-Q4_K_M-GGUF",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ]
    }
)

print(response.json()["message"]["content"])

# Generate embeddings
response = requests.post(
    "http://localhost:8000/api/embed",
    json={
        "model": "HAI3-raw-Q4_K_M-GGUF",
        "input": "Hello, world!"
    }
)

print(response.json()["embeddings"])
```

## 🐍 Native Python Client

Inferno includes a built-in Python client that provides a drop-in replacement for the OpenAI Python client. This allows you to use Inferno with existing code that uses the OpenAI client without any modifications.

### Using the Native Client

```python
from inferno.client import InfernoClient

# Initialize the client
client = InfernoClient(
    api_key="dummy",  # Not used by Inferno but kept for OpenAI compatibility
    api_base="http://localhost:8000/v1",  # Default Inferno API URL
)

# Chat completions
response = client.chat.create(
    model="HAI3-raw-Q4_K_M-GGUF",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
    max_tokens=100,
    temperature=0.7,
)

print(response["choices"][0]["message"]["content"])

# Streaming chat completions
stream = client.chat.create(
    model="HAI3-raw-Q4_K_M-GGUF",
    messages=[
        {"role": "user", "content": "Tell me a joke"}
    ],
    max_tokens=100,
    temperature=0.7,
    stream=True,
)

for chunk in stream:
    if "choices" in chunk and len(chunk["choices"]) > 0:
        if "delta" in chunk["choices"][0] and "content" in chunk["choices"][0]["delta"]:
            content = chunk["choices"][0]["delta"]["content"]
            print(content, end="", flush=True)

# Embeddings
response = client.embeddings.create(
    model="HAI3-raw-Q4_K_M-GGUF",
    input="Hello, world!",
)

print(response["data"][0]["embedding"])

# List models
models = client.models.list()
for model in models["data"]:
    print(model["id"])
```

### Client Features

- **OpenAI Compatibility**: Drop-in replacement for the OpenAI Python client
- **Streaming Support**: Stream responses for chat completions and text completions
- **Embeddings**: Generate embeddings from text
- **Model Management**: List and retrieve available models
- **Error Handling**: Comprehensive error handling with retries
- **Configuration Options**: Customize timeout, retries, and headers

For more details, see the [Python Client README](inferno/client/README.md).

## 📦 Requirements

### Software Requirements
- Python 3.9+
- llama-cpp-python
- FastAPI
- Uvicorn
- Rich (for terminal UI)
- Typer (for CLI)
- Hugging Face Hub
- Pydantic
- Requests

### Hardware Requirements
- Around 2 GB of RAM is needed for 1B models
- Around 4 GB of RAM is needed for 3B models
- You should have at least 8 GB of RAM available to run 7B models
- 16 GB of RAM is recommended for 13B models
- 32 GB of RAM is required for 33B models
- GPU acceleration is recommended for better performance

### Quantization Types and RAM Usage
| Quantization | Bits/Param | RAM Multiplier | Description |
|--------------|------------|----------------|-------------|
| Q2_K         | ~2.5       | 1.15×          | 2-bit quantization (lowest quality, smallest size) |
| Q3_K_M       | ~3.5       | 1.28×          | 3-bit quantization (medium) |
| Q4_K_M       | ~4.5       | 1.40×          | 4-bit quantization (balanced quality/size) |
| Q5_K_M       | ~5.5       | 1.65×          | 5-bit quantization (better quality) |
| Q6_K         | ~6.5       | 1.80×          | 6-bit quantization (high quality) |
| Q8_0         | ~8.5       | 2.00×          | 8-bit quantization (very high quality) |
| F16          | 16.0       | 2.80×          | 16-bit float (highest quality, largest size) |

## 🔧 Advanced Configuration

Inferno allows you to configure various aspects of model loading and inference:

### GPU Acceleration
```bash
# Set number of layers to offload to GPU
inferno serve HAI3-raw-Q4_K_M-GGUF --n_gpu_layers 32
```

### Context Length
```bash
# Set custom context length
inferno serve HAI3-raw-Q4_K_M-GGUF --n_ctx 8192
```

### Threading
```bash
# Set number of threads for inference
inferno serve HAI3-raw-Q4_K_M-GGUF --n_threads 8
```

### Memory Options
```bash
# Use mlock to keep model in memory
inferno serve HAI3-raw-Q4_K_M-GGUF --use_mlock
```

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute to Inferno, please follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them with descriptive messages
4. Push your branch to your forked repository
5. Submit a pull request to the main repository

## 📄 License

This project is licensed under the [HelpingAI Open Source License](LICENSE) - a custom license that promotes open innovation and collaboration while ensuring responsible and ethical use of AI technology.

---

<div align="center">
  <p>Made with ❤️ by <a href="https://helpingai.co">HelpingAI</a></p>
</div>
