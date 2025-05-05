"""
API server with OpenAI compatibility
"""

import json
import time
import os
import datetime
from typing import Dict, Any, List, Optional, AsyncGenerator, Union

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..utils.config import config
from ..core.model_manager import ModelManager
from ..core.llm import LLMInterface

app = FastAPI(title="inferno API", description="OpenAI-compatible API for Inferno")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Models
model_manager: ModelManager = ModelManager()
loaded_models: Dict[str, LLMInterface] = {}  # Cache for loaded models

class Image(BaseModel):
    """
    Represents an image for multimodal models.
    """
    url: Optional[str] = None
    data: Optional[str] = None

class ToolFunction(BaseModel):
    """
    Represents a function for function calling.
    """
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)

class Tool(BaseModel):
    """
    Represents a tool for function calling.
    """
    type: str = "function"
    function: ToolFunction

class ToolCall(BaseModel):
    """
    Represents a tool call from the model.
    """
    function: Dict[str, Any]

class ChatMessage(BaseModel):
    """
    Represents a single chat message for the chat completion endpoint.
    """
    role: str
    content: Union[str, List[Dict[str, Any]], None] = ""
    images: Optional[List[str]] = None
    tool_calls: Optional[List[ToolCall]] = None

class ChatCompletionRequest(BaseModel):
    """
    Request model for chat completions.
    """
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: Optional[int] = None # Changed default from 256 to None
    stream: bool = False
    stop: Optional[List[str]] = None
    tools: Optional[List[Tool]] = None
    format: Optional[Union[str, Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[str] = "5m"

class CompletionRequest(BaseModel):
    """
    Request model for text completions.
    """
    model: str
    prompt: str
    suffix: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 256
    stream: bool = False
    stop: Optional[List[str]] = None
    images: Optional[List[str]] = None
    format: Optional[Union[str, Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    raw: bool = False
    keep_alive: Optional[str] = "5m"

class ModelInfo(BaseModel):
    """
    Model information for listing available models.
    """
    id: str
    object: str = "model"
    created: int
    owned_by: str = "inferno"

class ModelList(BaseModel):
    """
    List of available models.
    """
    object: str = "list"
    data: List[ModelInfo]

class EmbeddingRequest(BaseModel):
    """
    Request model for embeddings.
    """
    model: str
    input: Union[str, List[str]]
    truncate: bool = True
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[str] = "5m"

class CopyRequest(BaseModel):
    """
    Request model for copying a model.
    """
    source: str
    destination: str

class DeleteRequest(BaseModel):
    """
    Request model for deleting a model.
    """
    model: str

class ShowRequest(BaseModel):
    """
    Request model for showing model information.
    """
    model: str
    verbose: bool = False

class PullRequest(BaseModel):
    """
    Request model for pulling a model.
    """
    model: str
    insecure: bool = False
    stream: bool = True

class ModelDetails(BaseModel):
    """
    Detailed model information.
    """
    format: str = "gguf"
    family: str = "llama"
    families: Optional[List[str]] = None
    parameter_size: Optional[str] = None
    quantization_level: Optional[str] = None

class ModelResponse(BaseModel):
    """
    Response model for model information.
    """
    name: str
    modified_at: str
    size: int
    digest: Optional[str] = None
    details: Optional[ModelDetails] = None

class ModelsResponse(BaseModel):
    """
    Response model for listing models.
    """
    models: List[ModelResponse]

class RunningModel(BaseModel):
    """
    Information about a running model.
    """
    name: str
    model: str
    size: int
    digest: Optional[str] = None
    details: Optional[ModelDetails] = None
    expires_at: Optional[str] = None
    size_vram: Optional[int] = None

class RunningModelsResponse(BaseModel):
    """
    Response model for listing running models.
    """
    models: List[RunningModel]

class VersionResponse(BaseModel):
    """
    Response model for version information.
    """
    version: str

def parse_keep_alive(keep_alive: Optional[str]) -> int:
    """
    Parse the keep_alive parameter to seconds.
    Args:
        keep_alive (Optional[str]): Keep alive duration string.
    Returns:
        int: Keep alive duration in seconds.
    """
    if not keep_alive:
        return 300  # Default 5 minutes

    if keep_alive.endswith("ms"):
        return int(keep_alive[:-2]) // 1000
    elif keep_alive.endswith("s"):
        return int(keep_alive[:-1])
    elif keep_alive.endswith("m"):
        return int(keep_alive[:-1]) * 60
    elif keep_alive.endswith("h"):
        return int(keep_alive[:-1]) * 3600
    elif keep_alive == "0":
        return 0
    else:
        try:
            return int(keep_alive)
        except ValueError:
            return 300  # Default 5 minutes

def get_model(model_name: str, options: Optional[Dict[str, Any]] = None) -> LLMInterface:
    """
    Get or load a model by name, using a cache for efficiency.
    Args:
        model_name (str): Name of the model to load.
        options (Optional[Dict[str, Any]]): Additional options for loading the model.
    Returns:
        LLMInterface: Loaded model interface.
    Raises:
        HTTPException: If the model cannot be loaded.
    """
    if model_name not in loaded_models:
        try:
            loaded_models[model_name] = LLMInterface(model_name)

            # Merge options from config and request
            model_options = config.get("model_options", {}).copy()
            if options:
                model_options.update(options)

            # Extract all supported options
            n_gpu_layers = model_options.get("n_gpu_layers", None)
            n_ctx = model_options.get("n_ctx", None) or model_options.get("num_ctx", None)
            n_threads = model_options.get("n_threads", None)
            use_mlock = model_options.get("use_mlock", False)
            verbose = model_options.get("verbose", False)

            loaded_models[model_name].load_model(
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                n_threads=n_threads,
                use_mlock=use_mlock,
                verbose=verbose
            )
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found: {str(e)}")
    return loaded_models[model_name]

def unload_model(model_name: str) -> None:
    """
    Unload a model from memory.
    Args:
        model_name (str): Name of the model to unload.
    """
    if model_name in loaded_models:
        # Free the model resources
        loaded_models[model_name].llm = None
        # Remove from loaded models cache
        del loaded_models[model_name]

@app.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    """
    List available models.
    Returns:
        ModelList: List of available models.
    """
    models = model_manager.list_models()
    model_list: List[ModelInfo] = []
    for model in models:
        model_list.append(
            ModelInfo(
                id=model["name"],
                created=int(time.time()),
            )
        )
    return ModelList(object="list", data=model_list)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, background_tasks: BackgroundTasks) -> Any:
    """
    Create a chat completion.
    Args:
        request (ChatCompletionRequest): Chat completion request.
        background_tasks (BackgroundTasks): Background tasks for model unloading.
    Returns:
        StreamingResponse or dict: Streaming or regular response.
    """
    # Parse keep_alive parameter
    keep_alive_seconds = parse_keep_alive(request.keep_alive)

    # If messages is empty, just load the model and return
    if not request.messages:
        model = get_model(request.model, request.options)

        # Schedule unloading if keep_alive is 0
        if keep_alive_seconds == 0:
            background_tasks.add_task(unload_model, request.model)
            done_reason = "unload"
        else:
            done_reason = "load"

        return {
            "model": request.model,
            "created_at": datetime.datetime.now().isoformat(),
            "message": {
                "role": "assistant",
                "content": ""
            },
            "done_reason": done_reason,
            "done": True
        }

    # Process messages with images if present
    processed_messages = []
    for msg in request.messages:
        message_dict = {"role": msg.role, "content": msg.content}

        # Handle images for multimodal models
        if msg.images:
            message_dict["images"] = msg.images

        # Handle tool calls
        if msg.tool_calls:
            message_dict["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]

        processed_messages.append(message_dict)

    # Get or load the model
    model = get_model(request.model, request.options)

    # Handle streaming response
    if request.stream:
        async def generate() -> AsyncGenerator[str, None]:
            stream = model.create_chat_completion(
                messages=processed_messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=True,
                stop=request.stop,
                tools=request.tools,
                format=request.format,
            )
            for chunk in stream:
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

            # Schedule unloading if keep_alive is 0
            if keep_alive_seconds == 0:
                background_tasks.add_task(unload_model, request.model)

        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        # Non-streaming response
        response = model.create_chat_completion(
            messages=processed_messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=False,
            stop=request.stop,
            tools=request.tools,
            format=request.format,
        )

        # Schedule unloading if keep_alive is 0
        if keep_alive_seconds == 0:
            background_tasks.add_task(unload_model, request.model)

        return response

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, background_tasks: BackgroundTasks) -> Any:
    """
    Create a text completion.
    Args:
        request (CompletionRequest): Completion request.
        background_tasks (BackgroundTasks): Background tasks for model unloading.
    Returns:
        StreamingResponse or dict: Streaming or regular response.
    """
    # Parse keep_alive parameter
    keep_alive_seconds = parse_keep_alive(request.keep_alive)

    # If prompt is empty, just load the model and return
    if not request.prompt:
        model = get_model(request.model, request.options)

        # Schedule unloading if keep_alive is 0
        if keep_alive_seconds == 0:
            background_tasks.add_task(unload_model, request.model)
            done_reason = "unload"
        else:
            done_reason = "load"

        return {
            "model": request.model,
            "created_at": datetime.datetime.now().isoformat(),
            "response": "",
            "done": True,
            "done_reason": done_reason
        }

    # Get or load the model
    model = get_model(request.model, request.options)

    # Handle streaming response
    if request.stream:
        async def generate() -> AsyncGenerator[str, None]:
            stream = model.create_completion(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=True,
                stop=request.stop,
                suffix=request.suffix,
                images=request.images,
                system=request.system,
                template=request.template,
                context=request.context,
                raw=request.raw,
                format=request.format,
            )
            for chunk in stream:
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

            # Schedule unloading if keep_alive is 0
            if keep_alive_seconds == 0:
                background_tasks.add_task(unload_model, request.model)

        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        # Non-streaming response
        response = model.create_completion(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=False,
            stop=request.stop,
            suffix=request.suffix,
            images=request.images,
            system=request.system,
            template=request.template,
            context=request.context,
            raw=request.raw,
            format=request.format,
        )

        # Schedule unloading if keep_alive is 0
        if keep_alive_seconds == 0:
            background_tasks.add_task(unload_model, request.model)

        return response

@app.post("/api/embed")
async def create_embeddings(request: EmbeddingRequest, background_tasks: BackgroundTasks) -> Any:
    """
    Generate embeddings from a model.
    Args:
        request (EmbeddingRequest): Embedding request.
        background_tasks (BackgroundTasks): Background tasks for model unloading.
    Returns:
        dict: Embedding response.
    """
    # Parse keep_alive parameter
    keep_alive_seconds = parse_keep_alive(request.keep_alive)

    # Get or load the model
    model = get_model(request.model, request.options)

    try:
        # Generate embeddings
        embeddings = model.create_embeddings(
            input=request.input,
            truncate=request.truncate
        )

        # Schedule unloading if keep_alive is 0
        if keep_alive_seconds == 0:
            background_tasks.add_task(unload_model, request.model)

        return embeddings
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

def start_server(host: str = None, port: int = None, model_options: Dict[str, Any] = None) -> None:
    """
    Start the API server.
    Args:
        host (str, optional): Host to bind to. Defaults to config value.
        port (int, optional): Port to bind to. Defaults to config value.
        model_options (Dict[str, Any], optional): Options for model loading.
    """
    host = host or config.get("api_host", "127.0.0.1")
    port = port or config.get("api_port", 8000)
    
    # Store model options in config for use by get_model
    if model_options:
        config["model_options"] = model_options
    
    uvicorn.run(app, host=host, port=port)
