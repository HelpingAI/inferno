"""
API endpoints for Inferno
"""

import time
import json
import logging
import os
from typing import Dict, List, Optional, Union, Any, cast, AsyncGenerator
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..core.model_manager import ModelManager

logger = logging.getLogger(__name__)


# API Models
class GenerateRequest(BaseModel):
    model: str
    prompt: str = ""
    suffix: Optional[str] = None
    images: Optional[List[str]] = None
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: bool = True
    raw: bool = False
    format: Optional[Union[str, Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[str] = "5m"


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]
    images: Optional[List[str]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = True
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None  # Added tool_choice
    format: Optional[Union[str, Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[str] = "5m"


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    truncate: bool = True
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[str] = "5m"


class PullModelRequest(BaseModel):
    model: str
    insecure: bool = False
    stream: bool = True


class DeleteModelRequest(BaseModel):
    model: str


class ModelResponse(BaseModel):
    name: str
    modified_at: str
    size: int
    details: Optional[Dict[str, Any]] = None


class ModelsResponse(BaseModel):
    models: List[ModelResponse]


# API Router
router = APIRouter()


# Dependency to get model manager
def get_model_manager() -> ModelManager:
    return ModelManager()


@router.post("/api/generate", response_model=None)
async def generate(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager),
):
    """Generate a completion for a given prompt"""
    try:
        # Parse keep_alive
        keep_alive_seconds = 300  # Default 5 minutes
        if request.keep_alive:
            if request.keep_alive.endswith("ms"):
                keep_alive_seconds = int(request.keep_alive[:-2]) / 1000
            elif request.keep_alive.endswith("s"):
                keep_alive_seconds = int(request.keep_alive[:-1])
            elif request.keep_alive.endswith("m"):
                keep_alive_seconds = int(request.keep_alive[:-1]) * 60
            elif request.keep_alive.endswith("h"):
                keep_alive_seconds = int(request.keep_alive[:-1]) * 3600
            elif request.keep_alive == "0":
                # Special case: unload immediately after completion
                keep_alive_seconds = 0
            else:
                try:
                    keep_alive_seconds = int(request.keep_alive)
                except ValueError:
                    pass

        # If prompt is empty, just load the model and return
        if not request.prompt:
            model = model_manager.get_model_path(request.model)
            if not model:
                raise HTTPException(
                    status_code=404, detail=f"Model {request.model} not found"
                )

            # Schedule unloading if keep_alive is 0
            if keep_alive_seconds == 0:
                background_tasks.add_task(model_manager.remove_model, request.model)

            return {
                "model": request.model,
                "created_at": datetime.now().isoformat(),
                "response": "",
                "done": True,
                "done_reason": "load" if keep_alive_seconds > 0 else "unload",
            }

        # Load the model
        model_path = model_manager.get_model_path(request.model)
        if not model_path:
            raise HTTPException(
                status_code=404, detail=f"Model {request.model} not found"
            )

        # Create LLM interface
        from ..core.llm import LLMInterface

        model = LLMInterface(request.model)
        model.load_model(**(request.options or {}))

        # Process images if provided
        image_data = None
        if request.images and len(request.images) > 0:
            # For now, we only support the first image
            image_base64 = request.images[0]
            if image_base64.startswith("data:"):
                # Handle data URI
                image_base64 = image_base64.split(",", 1)[1]
            # Keep base64-encoded string to pass along to the model (expected type is list[str])
            image_data = image_base64

        # Prepare generation parameters
        generation_params = {
            "prompt": request.prompt,
            "suffix": request.suffix,
            "max_tokens": cast(int, request.options.get("num_predict", 128))
            if request.options
            else 128,
            "temperature": cast(float, request.options.get("temperature", 0.8))
            if request.options
            else 0.8,
            "top_p": cast(float, request.options.get("top_p", 0.95))
            if request.options
            else 0.95,
            "echo": False,
        }

        # Add system prompt if provided
        if request.system:
            generation_params["system_prompt"] = request.system

        # Add format if provided
        if request.format:
            generation_params["format"] = request.format

        # Add images if provided
        if image_data:
            generation_params["images"] = [image_data]

        # Add context if provided
        if request.context:
            generation_params["context"] = request.context

        # Stream the response
        if request.stream:

            async def generate_stream() -> AsyncGenerator[str, None]:
                start_time = time.time()
                load_time = 0  # We don't track this separately

                # Initial response
                yield (
                    json.dumps(
                        {
                            "model": request.model,
                            "created_at": datetime.now().isoformat(),
                            "response": "",
                            "done": False,
                        }
                    )
                    + "\n"
                )

                # Generate completion - make sure we're not using streaming mode here
                generation_params["stream"] = (
                    False  # Ensure we get a dictionary, not a generator
                )
                completion = cast(
                    Dict[str, Any],
                    model.create_completion(
                        prompt=request.prompt,
                        max_tokens=cast(int, request.options.get("num_predict", 128))
                        if request.options
                        else 128,
                        temperature=cast(float, request.options.get("temperature", 0.8))
                        if request.options
                        else 0.8,
                        top_p=cast(float, request.options.get("top_p", 0.95))
                        if request.options
                        else 0.95,
                        stream=False,
                        suffix=request.suffix,
                        images=[image_data] if image_data else None,
                        context=request.context,
                        system=request.system,
                        template=request.template,
                        format=request.format,
                        echo=False,
                    ),
                )

                # Final response with stats
                end_time = time.time()
                total_duration = int(
                    (end_time - start_time) * 1e9
                )  # Convert to nanoseconds

                yield (
                    json.dumps(
                        {
                            "model": request.model,
                            "created_at": datetime.now().isoformat(),
                            "response": completion["choices"][0]["text"],
                            "done": True,
                            "context": completion.get("context", []),
                            "total_duration": total_duration,
                            "load_duration": load_time,
                            "prompt_eval_count": completion.get("usage", {}).get(
                                "prompt_tokens", 0
                            ),
                            "prompt_eval_duration": 0,  # Not tracked
                            "eval_count": completion.get("usage", {}).get(
                                "completion_tokens", 0
                            ),
                            "eval_duration": 0,  # Not tracked
                        }
                    )
                    + "\n"
                )

                # Schedule unloading if keep_alive is 0
                if keep_alive_seconds == 0:
                    model.llm = None  # Free resources

            return StreamingResponse(generate_stream(), media_type="application/json")
        else:
            # Non-streaming response
            start_time = time.time()

            # Generate completion
            completion = cast(
                Dict[str, Any],
                model.create_completion(
                    prompt=request.prompt,
                    max_tokens=cast(int, request.options.get("num_predict", 128))
                    if request.options
                    else 128,
                    temperature=cast(float, request.options.get("temperature", 0.8))
                    if request.options
                    else 0.8,
                    top_p=cast(float, request.options.get("top_p", 0.95))
                    if request.options
                    else 0.95,
                    stream=False,
                    suffix=request.suffix,
                    images=[image_data] if image_data else None,
                    context=request.context,
                    system=request.system,
                    template=request.template,
                    format=request.format,
                    echo=False,
                ),
            )

            # Calculate durations
            end_time = time.time()
            total_duration = int(
                (end_time - start_time) * 1e9
            )  # Convert to nanoseconds

            # Schedule unloading if keep_alive is 0
            if keep_alive_seconds == 0:
                model.llm = None  # Free resources

            return {
                "model": request.model,
                "created_at": datetime.now().isoformat(),
                "response": completion["choices"][0]["text"],
                "done": True,
                "context": completion.get("context", []),
                "total_duration": total_duration,
                "load_duration": 0,  # Not tracked separately
                "prompt_eval_count": completion.get("usage", {}).get(
                    "prompt_tokens", 0
                ),
                "prompt_eval_duration": 0,  # Not tracked
                "eval_count": completion.get("usage", {}).get("completion_tokens", 0),
                "eval_duration": 0,  # Not tracked
            }

    except Exception as e:
        logger.error(f"Error in generate: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/chat", response_model=None)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager),
):
    """Generate a chat completion"""
    try:
        # Parse keep_alive
        keep_alive_seconds = 300  # Default 5 minutes
        if request.keep_alive:
            if request.keep_alive.endswith("ms"):
                keep_alive_seconds = int(request.keep_alive[:-2]) / 1000
            elif request.keep_alive.endswith("s"):
                keep_alive_seconds = int(request.keep_alive[:-1])
            elif request.keep_alive.endswith("m"):
                keep_alive_seconds = int(request.keep_alive[:-1]) * 60
            elif request.keep_alive.endswith("h"):
                keep_alive_seconds = int(request.keep_alive[:-1]) * 3600
            elif request.keep_alive == "0":
                # Special case: unload immediately after completion
                keep_alive_seconds = 0
            else:
                try:
                    keep_alive_seconds = int(request.keep_alive)
                except ValueError:
                    pass

        # If messages is empty, just load the model and return
        if not request.messages:
            model_path = model_manager.get_model_path(request.model)
            if not model_path:
                raise HTTPException(
                    status_code=404, detail=f"Model {request.model} not found"
                )

            return {
                "model": request.model,
                "created_at": datetime.now().isoformat(),
                "message": {"role": "assistant", "content": ""},
                "done_reason": "load" if keep_alive_seconds > 0 else "unload",
                "done": True,
            }

        # Load the model
        model_path = model_manager.get_model_path(request.model)
        if not model_path:
            raise HTTPException(
                status_code=404, detail=f"Model {request.model} not found"
            )

        # Create LLM interface
        from ..core.llm import LLMInterface

        model = LLMInterface(request.model)
        model.load_model(**(request.options or {}))

        # Convert messages to the format expected by llama-cpp-python
        messages = []
        for msg in request.messages:
            if isinstance(msg.content, str):
                messages.append({"role": msg.role, "content": msg.content})
            else:
                # Handle multimodal content
                messages.append({"role": msg.role, "content": msg.content})

        # Prepare chat parameters
        chat_params = {
            "messages": messages,
            "temperature": cast(float, request.options.get("temperature", 0.8))
            if request.options
            else 0.8,
            "top_p": cast(float, request.options.get("top_p", 0.95))
            if request.options
            else 0.95,
            "max_tokens": cast(int, request.options.get("num_predict", 128))
            if request.options
            else 128,
        }

        # Add tools if provided
        if request.tools:
            chat_params["tools"] = request.tools

        # Add format if provided
        if request.format:
            chat_params["format"] = request.format

        # Add tool_choice if provided
        if request.tool_choice:
            chat_params["tool_choice"] = request.tool_choice

        # Stream the response
        if request.stream:

            async def generate_stream() -> AsyncGenerator[str, None]:
                start_time = time.time()

                # Initial response
                yield (
                    json.dumps(
                        {
                            "model": request.model,
                            "created_at": datetime.now().isoformat(),
                            "message": {"role": "assistant", "content": ""},
                            "done": False,
                        }
                    )
                    + "\n"
                )

                # Generate chat completion - make sure we're not using streaming mode here
                chat_params["stream"] = (
                    False  # Ensure we get a dictionary, not a generator
                )
                completion = cast(
                    Dict[str, Any],
                    model.create_chat_completion(
                        messages=messages,
                        max_tokens=cast(int, request.options.get("num_predict", 128))
                        if request.options
                        else 128,
                        temperature=cast(float, request.options.get("temperature", 0.8))
                        if request.options
                        else 0.8,
                        top_p=cast(float, request.options.get("top_p", 0.95))
                        if request.options
                        else 0.95,
                        stream=False,
                        tools=request.tools,
                        tool_choice=request.tool_choice,
                        format=request.format,
                    ),
                )

                # Final response with stats
                end_time = time.time()
                total_duration = int(
                    (end_time - start_time) * 1e9
                )  # Convert to nanoseconds

                response_message = completion["choices"][0]["message"]

                yield (
                    json.dumps(
                        {
                            "model": request.model,
                            "created_at": datetime.now().isoformat(),
                            "message": response_message,
                            "done": True,
                            "done_reason": "stop",
                            "total_duration": total_duration,
                            "load_duration": 0,  # Not tracked separately
                            "prompt_eval_count": completion.get("usage", {}).get(
                                "prompt_tokens", 0
                            ),
                            "prompt_eval_duration": 0,  # Not tracked
                            "eval_count": completion.get("usage", {}).get(
                                "completion_tokens", 0
                            ),
                            "eval_duration": 0,  # Not tracked
                        }
                    )
                    + "\n"
                )

                # Schedule unloading if keep_alive is 0
                if keep_alive_seconds == 0:
                    model.llm = None  # Free resources

            return StreamingResponse(generate_stream(), media_type="application/json")
        else:
            # Non-streaming response
            start_time = time.time()

            # Generate chat completion
            completion = cast(
                Dict[str, Any],
                model.create_chat_completion(
                    messages=messages,
                    max_tokens=cast(int, request.options.get("num_predict", 128))
                    if request.options
                    else 128,
                    temperature=cast(float, request.options.get("temperature", 0.8))
                    if request.options
                    else 0.8,
                    top_p=cast(float, request.options.get("top_p", 0.95))
                    if request.options
                    else 0.95,
                    stream=False,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    format=request.format,
                ),
            )

            # Calculate durations
            end_time = time.time()
            total_duration = int(
                (end_time - start_time) * 1e9
            )  # Convert to nanoseconds

            response_message = completion["choices"][0]["message"]

            # Schedule unloading if keep_alive is 0
            if keep_alive_seconds == 0:
                model.llm = None  # Free resources

            return {
                "model": request.model,
                "created_at": datetime.now().isoformat(),
                "message": response_message,
                "done": True,
                "done_reason": "stop",
                "total_duration": total_duration,
                "load_duration": 0,  # Not tracked separately
                "prompt_eval_count": completion.get("usage", {}).get(
                    "prompt_tokens", 0
                ),
                "prompt_eval_duration": 0,  # Not tracked
                "eval_count": completion.get("usage", {}).get("completion_tokens", 0),
                "eval_duration": 0,  # Not tracked
            }

    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/embed", response_model=None)
async def embed(
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager),
):
    """Generate embeddings from a model"""
    try:
        # Parse keep_alive
        keep_alive_seconds = 300  # Default 5 minutes
        if request.keep_alive:
            if request.keep_alive.endswith("ms"):
                keep_alive_seconds = int(request.keep_alive[:-2]) / 1000
            elif request.keep_alive.endswith("s"):
                keep_alive_seconds = int(request.keep_alive[:-1])
            elif request.keep_alive.endswith("m"):
                keep_alive_seconds = int(request.keep_alive[:-1]) * 60
            elif request.keep_alive.endswith("h"):
                keep_alive_seconds = int(request.keep_alive[:-1]) * 3600
            elif request.keep_alive == "0":
                # Special case: unload immediately after completion
                keep_alive_seconds = 0
            else:
                try:
                    keep_alive_seconds = int(request.keep_alive)
                except ValueError:
                    pass

        # Load the model
        model_path = model_manager.get_model_path(request.model)
        if not model_path:
            raise HTTPException(
                status_code=404, detail=f"Model {request.model} not found"
            )

        # Create LLM interface
        from ..core.llm import LLMInterface

        model = LLMInterface(request.model)
        model.load_model(**(request.options or {}))

        # Generate embeddings
        start_time = time.time()

        embeddings = model.create_embeddings(
            input=request.input, truncate=request.truncate
        )

        # Calculate durations
        end_time = time.time()
        _total_duration = int((end_time - start_time) * 1e9)  # Convert to nanoseconds

        # Schedule unloading if keep_alive is 0
        if keep_alive_seconds == 0:
            model.llm = None  # Free resources

        return embeddings

    except Exception as e:
        logger.error(f"Error in embed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/pull", response_model=None)
async def pull_model(
    request: PullModelRequest,
    model_manager: ModelManager = Depends(get_model_manager),
) -> Union[Dict[str, str], StreamingResponse]:
    """Pull a model from Hugging Face Hub"""
    try:
        if request.stream:

            async def generate_stream() -> AsyncGenerator[str, None]:
                # Initial response
                yield json.dumps({"status": "pulling manifest"}) + "\n"

                # Pull the model
                try:
                    model_name, model_path = model_manager.download_model(request.model)

                    # Success response
                    yield (
                        json.dumps(
                            {
                                "status": "success",
                                "model": model_name,
                                "path": str(model_path),
                            }
                        )
                        + "\n"
                    )
                except Exception as e:
                    # Error response
                    yield json.dumps({"status": "error", "error": str(e)}) + "\n"

            return StreamingResponse(generate_stream(), media_type="application/json")
        else:
            # Non-streaming response
            try:
                model_name, model_path = model_manager.download_model(request.model)

                return {
                    "status": "success",
                    "model": model_name,
                    "path": str(model_path),
                }
            except Exception as e:
                return {"status": "error", "error": str(e)}

    except Exception as e:
        logger.error(f"Error in pull_model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/delete", response_model=None)
async def delete_model(
    request: DeleteModelRequest,
    model_manager: ModelManager = Depends(get_model_manager),
):
    """Delete a model"""
    try:
        success = model_manager.remove_model(request.model)

        if success:
            return {"status": "success", "model": request.model}
        else:
            return {"status": "error", "error": f"Model {request.model} not found"}

    except Exception as e:
        logger.error(f"Error in delete_model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/models", response_model=ModelsResponse)
async def list_models(
    model_manager: ModelManager = Depends(get_model_manager),
):
    """List available models"""
    try:
        models = model_manager.list_models()

        model_responses = []
        for model in models:
            model_responses.append(
                ModelResponse(
                    name=model["name"],
                    modified_at=model.get("downloaded_at", datetime.now().isoformat()),
                    size=os.path.getsize(model["path"])
                    if "path" in model and os.path.exists(model["path"])
                    else 0,
                    details={
                        "format": "gguf",
                        "family": "llama",
                        "parameter_size": model.get("parameter_size", None),
                        "quantization_level": model.get("quantization_level", None),
                    },
                )
            )

        return ModelsResponse(models=model_responses)

    except Exception as e:
        logger.error(f"Error in list_models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
