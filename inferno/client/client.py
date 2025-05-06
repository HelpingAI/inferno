"""
Main client implementation for the Inferno API.
"""

import json
import requests
from typing import Dict, Any, List, Optional, Union, Generator, Iterator, Callable
import time
from dataclasses import asdict, is_dataclass

from .config import InfernoConfig
from .exceptions import InfernoAPIError, InfernoConnectionError, InfernoTimeoutError
from .models import Message, Tool, ToolChoice # Import necessary models
from .utils import (
    MessagesInput, # Import the type alias
    generate_request_id,
    current_timestamp,
    parse_stream_response,
    handle_error_response,
    prepare_stop_sequences,
    calculate_usage
)


class InfernoClient:
    """
    Main client for interacting with the Inferno API.
    
    This client is designed to be compatible with the OpenAI Python client,
    allowing for easy migration from OpenAI to Inferno.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the Inferno client.
        
        Args:
            api_key: API key for authentication (not used by Inferno but kept for OpenAI compatibility)
            api_base: Base URL for the Inferno API
            api_version: API version (not used by Inferno but kept for OpenAI compatibility)
            organization: Organization ID (not used by Inferno but kept for OpenAI compatibility)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            default_headers: Default headers to include in all requests
        """
        self.config = InfernoConfig(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            organization=organization,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
        )
        
        # Initialize API resources
        self.completions = Completion(self)
        self.chat = ChatCompletion(self)
        self.embeddings = Embedding(self)
        self.models = Model(self)
    
    def request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        stream: bool = False,
        timeout: Optional[float] = None,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Make a request to the Inferno API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path
            params: Query parameters
            json_data: JSON data for the request body
            headers: Additional headers
            stream: Whether to stream the response
            timeout: Request timeout in seconds
            
        Returns:
            Union[Dict[str, Any], Iterator[Dict[str, Any]]]: API response
            
        Raises:
            InfernoAPIError: If the API returns an error
            InfernoConnectionError: If there's a connection error
            InfernoTimeoutError: If the request times out
        """
        # Build the request URL
        url = f"{self.config.api_base.rstrip('/')}/{path.lstrip('/')}"
        
        # Prepare headers
        request_headers = self.config.get_headers(headers)
        
        # Set timeout
        request_timeout = timeout if timeout is not None else self.config.timeout
        
        # Make the request with retries
        retries = 0
        while True:
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data,
                    headers=request_headers,
                    timeout=request_timeout,
                    stream=stream,
                )
                
                # Handle streaming response
                if stream and response.status_code == 200:
                    return self._handle_streaming_response(response)
                
                # Handle non-streaming response
                if response.status_code != 200:
                    error_message = f"API request failed with status code: {response.status_code}"
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_message = error_data["error"].get("message", error_message)
                    except Exception:
                        pass
                    
                    raise InfernoAPIError(response.status_code, error_message, response)
                
                return response.json()
                
            except requests.exceptions.Timeout:
                if retries >= self.config.max_retries:
                    raise InfernoTimeoutError(request_timeout)
                
                retries += 1
                time.sleep(min(2 ** retries, 30))  # Exponential backoff
                
            except requests.exceptions.RequestException as e:
                if retries >= self.config.max_retries:
                    raise InfernoConnectionError(str(e), e)
                
                retries += 1
                time.sleep(min(2 ** retries, 30))  # Exponential backoff
    
    def _handle_streaming_response(self, response: requests.Response) -> Iterator[Dict[str, Any]]:
        """
        Handle a streaming response from the API.
        
        Args:
            response: Streaming response
            
        Returns:
            Iterator[Dict[str, Any]]: Stream of parsed responses
            
        Raises:
            InfernoAPIError: If the API returns an error
        """
        for line in response.iter_lines():
            if not line:
                continue
            
            line_text = line.decode("utf-8")
            
            # Check for data prefix and parse JSON
            if line_text.startswith("data: "):
                line_text = line_text[6:]
                
                # Check for end of stream
                if line_text.strip() == "[DONE]":
                    break
                
                try:
                    data = json.loads(line_text)
                    yield data
                except json.JSONDecodeError:
                    continue


class Completion:
    """
    Completions API resource.
    """
    
    def __init__(self, client: InfernoClient):
        """
        Initialize the Completions API resource.
        
        Args:
            client: Inferno client
        """
        self.client = client
    
    def create(
        self,
        model: str,
        prompt: Union[str, List[str]],
        suffix: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        n: int = 1,
        stream: bool = False,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        best_of: int = 1,
        logit_bias: Optional[Dict[str, float]] = None,
        user: Optional[str] = None,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Create a completion.
        
        Args:
            model: ID of the model to use
            prompt: Prompt to generate completions for
            suffix: Suffix that comes after a completion of inserted text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            n: Number of completions to generate
            stream: Whether to stream the response
            logprobs: Include log probabilities of the top n tokens
            echo: Echo back the prompt in addition to the completion
            stop: Up to 4 sequences where the API will stop generating further tokens
            presence_penalty: Penalty for token presence
            frequency_penalty: Penalty for token frequency
            best_of: Generate best_of completions server-side and return the "best"
            logit_bias: Modify the likelihood of specified tokens appearing in the completion
            user: A unique identifier representing your end-user
            
        Returns:
            Union[Dict[str, Any], Iterator[Dict[str, Any]]]: Completion response
        """
        # Prepare request data
        json_data = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        
        # Add optional parameters
        if suffix is not None:
            json_data["suffix"] = suffix
        
        if n != 1:
            json_data["n"] = n
            
        if logprobs is not None:
            json_data["logprobs"] = logprobs
            
        if echo:
            json_data["echo"] = echo
            
        if stop is not None:
            json_data["stop"] = prepare_stop_sequences(stop)
            
        if presence_penalty != 0.0:
            json_data["presence_penalty"] = presence_penalty
            
        if frequency_penalty != 0.0:
            json_data["frequency_penalty"] = frequency_penalty
            
        if best_of != 1:
            json_data["best_of"] = best_of
            
        if logit_bias is not None:
            json_data["logit_bias"] = logit_bias
            
        if user is not None:
            json_data["user"] = user
        
        # Make the request
        return self.client.request(
            method="POST",
            path="/completions",
            json_data=json_data,
            stream=stream,
        )


class ChatCompletion:
    """
    Chat completions API resource.
    """
    
    def __init__(self, client: InfernoClient):
        """
        Initialize the Chat Completions API resource.
        
        Args:
            client: Inferno client
        """
        self.client = client
    
    def create(
        self,
        model: str,
        messages: MessagesInput, # Use the MessagesInput type alias
        temperature: float = 0.7,
        top_p: float = 0.95,
        n: int = 1,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        max_tokens: int = 256,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[Dict[str, float]] = None,
        user: Optional[str] = None,
        tools: Optional[List[Tool]] = None, # Added tools
        tool_choice: Optional[ToolChoice] = None, # Added tool_choice
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Create a chat completion.

        Args:
            model: ID of the model to use.
            messages: A list of messages comprising the conversation so far.
                Can be a list of `Message` dataclasses or a list of dictionaries
                following the OpenAI format.
                The `content` field can be a string or a list of content parts
                (text or image_url) for multimodal input. Image URLs can be
                data URIs (e.g., "data:image/jpeg;base64,...") or standard
                HTTPS URLs.
                Example (using dicts):
                ```python
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": {"url": "https://..."}}
                    ]}
                ]
                ```
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            n: Number of chat completion choices to generate.
            stream: Whether to stream the response
            stop: Up to 4 sequences where the API will stop generating further tokens
            max_tokens: Maximum number of tokens to generate
            presence_penalty: Penalty for token presence.
            frequency_penalty: Penalty for token frequency.
            logit_bias: Modify the likelihood of specified tokens appearing in the completion.
            user: A unique identifier representing your end-user.
            tools: A list of tools the model may call. Currently only functions are supported.
            tool_choice: Controls which tool is called by the model. "none" means no tool call,
                         "auto" lets the model decide, and a specific tool name forces that tool.

        Returns:
            Union[Dict[str, Any], Iterator[Dict[str, Any]]]: Chat completion response
        """
        # Prepare request data
        json_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "max_tokens": max_tokens,
        }
        
        # Add optional parameters
        if n != 1:
            json_data["n"] = n
            
        if stop is not None:
            json_data["stop"] = prepare_stop_sequences(stop)
            
        if presence_penalty != 0.0:
            json_data["presence_penalty"] = presence_penalty
            
        if frequency_penalty != 0.0:
            json_data["frequency_penalty"] = frequency_penalty
            
        if logit_bias is not None:
            json_data["logit_bias"] = logit_bias

        if user is not None:
            json_data["user"] = user

        # Add tools and tool_choice if provided
        if tools is not None:
            # Convert Tool dataclasses to dicts if necessary
            json_data["tools"] = [asdict(tool) if is_dataclass(tool) else tool for tool in tools]

        if tool_choice is not None:
             # Convert ToolChoice dataclass to dict if necessary
            json_data["tool_choice"] = asdict(tool_choice) if is_dataclass(tool_choice) else tool_choice

        # Make the request
        return self.client.request(
            method="POST",
            path="/chat/completions",
            json_data=json_data,
            stream=stream,
        )


class Embedding:
    """
    Embeddings API resource.
    """
    
    def __init__(self, client: InfernoClient):
        """
        Initialize the Embeddings API resource.
        
        Args:
            client: Inferno client
        """
        self.client = client
    
    def create(
        self,
        model: str,
        input: Union[str, List[str]],
        user: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create embeddings.
        
        Args:
            model: ID of the model to use
            input: Input text to embed
            user: A unique identifier representing your end-user
            
        Returns:
            Dict[str, Any]: Embeddings response
        """
        # Prepare request data
        json_data = {
            "model": model,
            "input": input,
        }
        
        # Add optional parameters
        if user is not None:
            json_data["user"] = user
        
        # Make the request
        return self.client.request(
            method="POST",
            path="/embeddings",
            json_data=json_data,
        )


class Model:
    """
    Models API resource.
    """
    
    def __init__(self, client: InfernoClient):
        """
        Initialize the Models API resource.
        
        Args:
            client: Inferno client
        """
        self.client = client
    
    def list(self) -> Dict[str, Any]:
        """
        List available models.
        
        Returns:
            Dict[str, Any]: List of models
        """
        return self.client.request(
            method="GET",
            path="/models",
        )
    
    def retrieve(self, model: str) -> Dict[str, Any]:
        """
        Retrieve a model.
        
        Args:
            model: ID of the model to retrieve
            
        Returns:
            Dict[str, Any]: Model details
        """
        return self.client.request(
            method="GET",
            path=f"/models/{model}",
        )


