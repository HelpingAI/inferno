# Inferno Python Client

An OpenAI-compatible Python client for Inferno, allowing you to easily interact with Inferno's API using a familiar interface.

## Features

- **OpenAI Compatibility**: Drop-in replacement for the OpenAI Python client
- **Streaming Support**: Stream responses for chat completions and text completions
- **Embeddings**: Generate embeddings from text
- **Model Management**: List and retrieve available models
- **Error Handling**: Comprehensive error handling with retries

## Installation

The Inferno client is included with the Inferno package. If you have Inferno installed, you already have access to the client.

## Usage

### Initializing the Client

```python
from inferno.client import InfernoClient

# Initialize the client
client = InfernoClient(
    api_key="dummy",  # Not used by Inferno but kept for OpenAI compatibility
    api_base="http://localhost:8000/v1",  # Default Inferno API URL
)
```

### Chat Completions

```python
# Create a chat completion
response = client.chat.create(
    model="HAI3-raw-Q4_K_M-GGUF",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
    max_tokens=100,
    temperature=0.7,
)

# Print the response
print(response["choices"][0]["message"]["content"])
```

### Streaming Chat Completions

```python
# Create a streaming chat completion
stream = client.chat.create(
    model="HAI3-raw-Q4_K_M-GGUF",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a short joke."}
    ],
    max_tokens=100,
    temperature=0.7,
    stream=True,
)

# Process the streaming response
for chunk in stream:
    if "choices" in chunk and len(chunk["choices"]) > 0:
        if "delta" in chunk["choices"][0] and "content" in chunk["choices"][0]["delta"]:
            content = chunk["choices"][0]["delta"]["content"]
            print(content, end="", flush=True)
```

### Text Completions

```python
# Create a text completion
response = client.completions.create(
    model="HAI3-raw-Q4_K_M-GGUF",
    prompt="Once upon a time",
    max_tokens=50,
    temperature=0.7,
)

# Print the response
print(response["choices"][0]["text"])
```

### Embeddings

```python
# Create embeddings
response = client.embeddings.create(
    model="HAI3-raw-Q4_K_M-GGUF",
    input="Hello, world!",
)

# Print the embedding
print(response["data"][0]["embedding"])
```

### Listing Models

```python
# List available models
models = client.models.list()

# Print the models
for model in models["data"]:
    print(model["id"])
```

## Configuration

The client can be configured with the following parameters:

- `api_key`: API key for authentication (not used by Inferno but kept for OpenAI compatibility)
- `api_base`: Base URL for the Inferno API (default: `http://localhost:8000/v1`)
- `api_version`: API version (not used by Inferno but kept for OpenAI compatibility)
- `organization`: Organization ID (not used by Inferno but kept for OpenAI compatibility)
- `timeout`: Request timeout in seconds (default: 60.0)
- `max_retries`: Maximum number of retries for failed requests (default: 3)
- `default_headers`: Default headers to include in all requests

## Environment Variables

The client can also be configured using environment variables:

- `INFERNO_API_KEY`: API key for authentication
- `INFERNO_API_BASE`: Base URL for the Inferno API
- `INFERNO_API_VERSION`: API version
- `INFERNO_ORGANIZATION`: Organization ID
- `INFERNO_TIMEOUT`: Request timeout in seconds
- `INFERNO_MAX_RETRIES`: Maximum number of retries for failed requests

## Error Handling

The client provides comprehensive error handling with the following exception types:

- `InfernoError`: Base exception for all Inferno client errors
- `InfernoAPIError`: Exception raised when the Inferno API returns an error
- `InfernoConnectionError`: Exception raised when there's a connection error to the Inferno API
- `InfernoTimeoutError`: Exception raised when a request to the Inferno API times out

## Example

See the `example.py` file for a complete example of using the client.

## License

MIT

---

Made with ❤️ by [HelpingAI](https://helpingai.co)
