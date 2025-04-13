from fastapi import APIRouter, FastAPI, HTTPException, Request, BackgroundTasks, Query
from fastapi.responses import JSONResponse, StreamingResponse
import json
import asyncio
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union, Literal
import time
import sys

from inferno.utils.logger import get_logger
from inferno.config.server_config import ServerConfig
from inferno.models.registry import MODEL_REGISTRY
from inferno.models.loader import load_and_register_model, unload_and_unregister_model
from inferno.server.task_queue import TaskQueue
from inferno.server.generation import generate_completion, generate_chat_completion

logger = get_logger(__name__)

# Create a task queue for handling requests
task_queue = TaskQueue()


# Define API models
class ModelInfo(BaseModel):
    id: str
    path: str
    is_default: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelsResponse(BaseModel):
    models: List[ModelInfo]


class Message(BaseModel):
    role: str
    content: Optional[str] = ""


class CompletionRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = None
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    message: Message
    index: int
    finish_reason: Optional[str] = None

    class Config:
        # Allow extra fields to be ignored
        extra = "ignore"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Optional[UsageInfo] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[UsageInfo] = None

    class Config:
        # Allow extra fields to be ignored
        extra = "ignore"


class ErrorResponse(BaseModel):
    error: str


class HealthResponse(BaseModel):
    status: str
    version: str
    models: int


class LoadModelRequest(BaseModel):
    model_path: str
    set_default: bool = False
    enable_gguf: bool = False
    download_gguf: bool = False
    load_8bit: bool = False
    load_4bit: bool = False


class LoadModelResponse(BaseModel):
    model_id: str
    status: str


class UnloadModelResponse(BaseModel):
    status: str


class ShutdownResponse(BaseModel):
    status: str


def register_openai_routes(app: FastAPI, config: ServerConfig):
    """
    Register OpenAI-compatible API routes.

    Args:
        app: FastAPI application
        config: Server configuration
    """
    router = APIRouter(prefix="/v1", tags=["OpenAI API"])

    @router.get("/models", response_model=ModelsResponse)
    async def list_models():
        """
        List all available models.
        """
        models = MODEL_REGISTRY.list_models()
        return {"models": models}

    @router.post("/completions", response_model=CompletionResponse)
    async def create_completion(request: CompletionRequest, background_tasks: BackgroundTasks):
        """
        Create a completion.
        """
        # Get the model
        model_info = MODEL_REGISTRY.get_model(request.model)
        if model_info is None:
            raise HTTPException(status_code=404, detail="Model not found")

        # Generate the completion
        task_id = f"completion-{int(time.time() * 1000)}"

        try:
            # Check if streaming is requested
            if request.stream:
                # Call the generation function with streaming
                streamer = generate_completion(
                    model_info=model_info,
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    frequency_penalty=request.frequency_penalty,
                    presence_penalty=request.presence_penalty,
                    stop=request.stop,
                    stream=True
                )

                # Create a streaming response
                async def stream_response():
                    # Send the initial response
                    chunk_id = f"cmpl-{int(time.time() * 1000)}"
                    created = int(time.time())

                    # Stream the content
                    for text_chunk in streamer:
                        if text_chunk:
                            chunk = {
                                "id": chunk_id,
                                "object": "text_completion",
                                "created": created,
                                "model": model_info.model_id,
                                "choices": [
                                    {
                                        "text": text_chunk,
                                        "index": 0,
                                        "logprobs": None,
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"

                    # Send the final chunk
                    final_chunk = {
                        "id": chunk_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_info.model_id,
                        "choices": [
                            {
                                "text": "",
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": "length"
                            }
                        ]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"

                    # Send the [DONE] message
                    yield "data: [DONE]\n\n"

                return StreamingResponse(stream_response(), media_type="text/event-stream")
            else:
                # Call the generation function without streaming
                completion_text, finish_reason = generate_completion(
                    model_info=model_info,
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    frequency_penalty=request.frequency_penalty,
                    presence_penalty=request.presence_penalty,
                    stop=request.stop,
                    stream=False
                )

                # Estimate token counts (this is a rough estimate)
                prompt_tokens = 0
                completion_tokens = 0

                # If we have a tokenizer, try to get actual token counts
                if model_info.tokenizer:
                    try:
                        # Count prompt tokens
                        if isinstance(request.prompt, str):
                            prompt_tokens = len(model_info.tokenizer.encode(request.prompt))

                        # Count completion tokens
                        if completion_text:
                            completion_tokens = len(model_info.tokenizer.encode(completion_text))
                    except Exception as e:
                        logger.warning(f"Error estimating token counts: {e}")
                        # Fallback to rough estimates
                        prompt_tokens = len(request.prompt) // 4 if isinstance(request.prompt, str) else 0
                        completion_tokens = len(completion_text) // 4
                else:
                    # Rough estimate based on characters (assuming ~4 chars per token on average)
                    prompt_tokens = len(request.prompt) // 4 if isinstance(request.prompt, str) else 0
                    completion_tokens = len(completion_text) // 4

                total_tokens = prompt_tokens + completion_tokens

                # Create a proper CompletionChoice object
                choice = CompletionChoice(
                    text=completion_text if completion_text is not None else "",
                    index=0,
                    finish_reason=finish_reason
                )

                # Create a proper UsageInfo object
                usage = UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens
                )

                # Create a proper CompletionResponse object
                response = CompletionResponse(
                    id=task_id,
                    created=int(time.time()),
                    model=model_info.model_id,
                    choices=[choice],
                    usage=usage
                )

                # Log the response for debugging
                logger.debug(f"Completion response: {response}")

                # Return the response as a dictionary with explicit serialization
                response_dict = {
                    "id": response.id,
                    "object": response.object,
                    "created": response.created,
                    "model": response.model,
                    "choices": [
                        {
                            "text": choice.text,
                            "index": choice.index,
                            "finish_reason": choice.finish_reason
                        } for choice in response.choices
                    ],
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0
                    }
                }

                # Log the final response dictionary for debugging
                logger.debug(f"Final completion response dictionary: {response_dict}")

                return response_dict
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/chat/completions", response_model=ChatCompletionResponse)
    async def create_chat_completion(request: ChatCompletionRequest, background_tasks: BackgroundTasks):
        """
        Create a chat completion.
        """
        # Get the model
        model_info = MODEL_REGISTRY.get_model(request.model)
        if model_info is None:
            raise HTTPException(status_code=404, detail="Model not found")

        # Generate the chat completion
        task_id = f"chat-completion-{int(time.time() * 1000)}"

        try:
            # Convert Pydantic models to dictionaries
            try:
                # Use model_dump for newer versions of Pydantic
                messages = [msg.model_dump() for msg in request.messages]
            except AttributeError:
                # Fallback to dict for older versions
                messages = [msg.dict() for msg in request.messages]

            # Log the messages for debugging
            logger.debug(f"Messages for chat completion: {messages}")

            # Check if streaming is requested
            if request.stream:
                # Call the generation function with streaming
                streamer = generate_chat_completion(
                    model_info=model_info,
                    messages=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    frequency_penalty=request.frequency_penalty,
                    presence_penalty=request.presence_penalty,
                    stop=request.stop,
                    stream=True
                )

                # Create a streaming response
                async def stream_response():
                    # Send the initial response
                    chunk_id = f"chatcmpl-{int(time.time() * 1000)}"
                    created = int(time.time())

                    # Send the first chunk with role
                    first_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_info.model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(first_chunk)}\n\n"

                    # Stream the content
                    for text_chunk in streamer:
                        if text_chunk:
                            chunk = {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_info.model_id,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": text_chunk},
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"

                    # Send the final chunk
                    final_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_info.model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "length"
                            }
                        ]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"

                    # Send the [DONE] message
                    yield "data: [DONE]\n\n"

                return StreamingResponse(stream_response(), media_type="text/event-stream")
            else:
                # Call the generation function without streaming
                completion_text, finish_reason = generate_chat_completion(
                    model_info=model_info,
                    messages=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    frequency_penalty=request.frequency_penalty,
                    presence_penalty=request.presence_penalty,
                    stop=request.stop,
                    stream=False
                )

                # Log the completion text for debugging
                logger.info(f"Generated completion text: {completion_text}")

                # Ensure we have a valid string for the content
                if completion_text is None:
                    completion_text = ""
                elif not isinstance(completion_text, str):
                    logger.warning(f"Completion text is not a string: {type(completion_text)}")
                    completion_text = str(completion_text)

                # Estimate token counts (this is a rough estimate)
                prompt_tokens = 0
                completion_tokens = 0

                # If we have a tokenizer, try to get actual token counts
                if model_info.tokenizer:
                    try:
                        # Count prompt tokens
                        for msg in messages:
                            if isinstance(msg.get('content', ''), str):
                                prompt_tokens += len(model_info.tokenizer.encode(msg.get('content', '')))

                        # Count completion tokens
                        if completion_text:
                            completion_tokens = len(model_info.tokenizer.encode(completion_text))
                    except Exception as e:
                        logger.warning(f"Error estimating token counts: {e}")
                        # Fallback to rough estimates
                        prompt_tokens = sum(len(msg.get('content', '')) // 4 for msg in messages if isinstance(msg.get('content', ''), str))
                        completion_tokens = len(completion_text) // 4
                else:
                    # Rough estimate based on characters (assuming ~4 chars per token on average)
                    prompt_tokens = sum(len(msg.get('content', '')) // 4 for msg in messages if isinstance(msg.get('content', ''), str))
                    completion_tokens = len(completion_text) // 4

                total_tokens = prompt_tokens + completion_tokens

                # Create a proper Message object
                message = Message(role="assistant", content=completion_text if completion_text is not None else "")

                # Create a proper ChatCompletionChoice object
                choice = ChatCompletionChoice(
                    message=message,
                    index=0,
                    finish_reason=finish_reason
                )

                # Create a proper UsageInfo object
                usage = UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens
                )

                # Create a proper ChatCompletionResponse object
                response = ChatCompletionResponse(
                    id=task_id,
                    created=int(time.time()),
                    model=model_info.model_id,
                    choices=[choice],
                    usage=usage
                )

                # Log the response for debugging
                logger.debug(f"Chat completion response: {response}")

                # Return the response as a dictionary with explicit serialization
                response_dict = {
                    "id": response.id,
                    "object": response.object,
                    "created": response.created,
                    "model": response.model,
                    "choices": [
                        {
                            "message": {
                                "role": choice.message.role,
                                "content": choice.message.content
                            },
                            "index": choice.index,
                            "finish_reason": choice.finish_reason
                        } for choice in response.choices
                    ],
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0
                    }
                }

                # Log the final response dictionary for debugging
                logger.debug(f"Final response dictionary: {response_dict}")

                return response_dict
        except Exception as e:
            logger.error(f"Error generating chat completion: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    app.include_router(router)


def register_admin_routes(app: FastAPI, config: ServerConfig):
    """
    Register admin API routes.

    Args:
        app: FastAPI application
        config: Server configuration
    """
    router = APIRouter(prefix="/admin", tags=["Admin API"])

    @router.post("/models/load", response_model=LoadModelResponse)
    async def load_model(
        model_path: str = Query(..., description="Path to the model to load"),
        set_default: bool = Query(False, description="Set as default model"),
        enable_gguf: bool = Query(False, description="Enable GGUF model support"),
        download_gguf: bool = Query(False, description="Download GGUF model"),
        load_8bit: bool = Query(False, description="Load in 8-bit precision"),
        load_4bit: bool = Query(False, description="Load in 4-bit precision")
    ):
        """
        Load a new model.
        """
        try:
            # Create a new config for this model
            model_config = ServerConfig(
                model_name_or_path=model_path,
                enable_gguf=enable_gguf,
                download_gguf=download_gguf,
                device=config.device,
                load_8bit=load_8bit,
                load_4bit=load_4bit,
                use_tpu=config.use_tpu,
                tpu_memory_limit=config.tpu_memory_limit
            )

            # Load and register the model
            model_id = load_and_register_model(model_config, set_default=set_default)

            return {
                "model_id": model_id,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/models/unload/{model_id}", response_model=UnloadModelResponse)
    async def unload_model(model_id: str):
        """
        Unload a model.
        """
        try:
            # Unload and unregister the model
            success = unload_and_unregister_model(model_id)

            if not success:
                raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

            return {"status": "success"}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/shutdown", response_model=ShutdownResponse)
    async def shutdown():
        """
        Gracefully shut down the server.
        """
        # Schedule server shutdown
        def shutdown_server():
            # Wait a moment to allow the response to be sent
            time.sleep(1)
            # Exit the process
            sys.exit(0)

        # Run shutdown in background
        import threading
        threading.Thread(target=shutdown_server).start()

        return {"status": "shutting down"}

    app.include_router(router)


def register_health_routes(app: FastAPI, config: ServerConfig):
    """
    Register health check routes.

    Args:
        app: FastAPI application
        config: Server configuration
    """
    router = APIRouter(prefix="/health", tags=["Health"])

    @router.get("", response_model=HealthResponse)
    async def health_check():
        """
        Check the health of the server.
        """
        return {
            "status": "ok",
            "version": "0.1.0",
            "models": MODEL_REGISTRY.get_model_count()
        }

    app.include_router(router)