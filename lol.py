#!/usr/bin/env python
"""
Test script for Inferno API
This script tests both regular and streaming requests to the Inferno API.
"""

import argparse
import json
import requests
import sys
import time
from typing import Dict, List, Optional, Union, Any

def make_regular_completion_request(
    prompt: str,
    api_url: str = "http://localhost:8000/v1/completions",
    api_key: str = "your_api_key",
    max_tokens: int = 150,
    temperature: float = 0.7,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """Make a regular completion request to the API."""
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key
    }
    
    data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    if model:
        data["model"] = model
    
    print(f"\n\033[1m[Regular Completion Request]\033[0m")
    print(f"Prompt: {prompt}")
    
    start_time = time.time()
    response = requests.post(api_url, headers=headers, json=data)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n\033[32mResponse (in {end_time - start_time:.2f}s):\033[0m")
        if "choices" in result and len(result["choices"]) > 0:
            print(f"\033[32m{result['choices'][0].get('text', '')}\033[0m")
        print("\nFull response:")
        print(json.dumps(result, indent=2))
        return result
    else:
        print(f"\n\033[31mError: {response.status_code}\033[0m")
        print(response.text)
        return {"error": response.text}

def make_streaming_completion_request(
    prompt: str,
    api_url: str = "http://localhost:8000/v1/completions",
    api_key: str = "your_api_key",
    max_tokens: int = 150,
    temperature: float = 0.7,
    model: Optional[str] = None
) -> None:
    """Make a streaming completion request to the API."""
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key
    }
    
    data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True
    }
    
    if model:
        data["model"] = model
    
    print(f"\n\033[1m[Streaming Completion Request]\033[0m")
    print(f"Prompt: {prompt}")
    print("\n\033[32mResponse:\033[0m")
    
    start_time = time.time()
    with requests.post(api_url, headers=headers, json=data, stream=True) as response:
        if response.status_code != 200:
            print(f"\n\033[31mError: {response.status_code}\033[0m")
            print(response.text)
            return
        
        # Process the streaming response
        full_text = ""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: ') and not line.startswith('data: [DONE]'):
                    json_str = line[6:]  # Remove 'data: ' prefix
                    try:
                        chunk = json.loads(json_str)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            text_chunk = chunk["choices"][0].get("text", "")
                            if text_chunk:
                                print(text_chunk, end="", flush=True)
                                full_text += text_chunk
                    except json.JSONDecodeError:
                        print(f"\n\033[31mError decoding JSON: {json_str}\033[0m")
    
    end_time = time.time()
    print(f"\n\n\033[32mCompleted in {end_time - start_time:.2f}s\033[0m")
    print(f"\033[32mFull text: {full_text}\033[0m")

def make_regular_chat_completion_request(
    messages: List[Dict[str, str]],
    api_url: str = "http://localhost:8000/v1/chat/completions",
    api_key: str = "your_api_key",
    max_tokens: int = 150,
    temperature: float = 0.7,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """Make a regular chat completion request to the API."""
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key
    }
    
    data = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    if model:
        data["model"] = model
    
    print(f"\n\033[1m[Regular Chat Completion Request]\033[0m")
    print(f"Messages: {json.dumps(messages, indent=2)}")
    
    start_time = time.time()
    response = requests.post(api_url, headers=headers, json=data)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n\033[32mResponse (in {end_time - start_time:.2f}s):\033[0m")
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})
            content = message.get("content", "")
            print(f"\033[32m{content}\033[0m")
        print("\nFull response:")
        print(json.dumps(result, indent=2))
        return result
    else:
        print(f"\n\033[31mError: {response.status_code}\033[0m")
        print(response.text)
        return {"error": response.text}

def make_streaming_chat_completion_request(
    messages: List[Dict[str, str]],
    api_url: str = "http://localhost:8000/v1/chat/completions",
    api_key: str = "your_api_key",
    max_tokens: int = 150,
    temperature: float = 0.7,
    model: Optional[str] = None
) -> None:
    """Make a streaming chat completion request to the API."""
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key
    }
    
    data = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True
    }
    
    if model:
        data["model"] = model
    
    print(f"\n\033[1m[Streaming Chat Completion Request]\033[0m")
    print(f"Messages: {json.dumps(messages, indent=2)}")
    print("\n\033[32mResponse:\033[0m")
    
    start_time = time.time()
    with requests.post(api_url, headers=headers, json=data, stream=True) as response:
        if response.status_code != 200:
            print(f"\n\033[31mError: {response.status_code}\033[0m")
            print(response.text)
            return
        
        # Process the streaming response
        full_text = ""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: ') and not line.startswith('data: [DONE]'):
                    json_str = line[6:]  # Remove 'data: ' prefix
                    try:
                        chunk = json.loads(json_str)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                print(content, end="", flush=True)
                                full_text += content
                    except json.JSONDecodeError:
                        print(f"\n\033[31mError decoding JSON: {json_str}\033[0m")
    
    end_time = time.time()
    print(f"\n\n\033[32mCompleted in {end_time - start_time:.2f}s\033[0m")
    print(f"\033[32mFull text: {full_text}\033[0m")

def main():
    parser = argparse.ArgumentParser(description="Test the Inferno API")
    parser.add_argument("--host", default="localhost", help="API host")
    parser.add_argument("--port", default=8000, type=int, help="API port")
    parser.add_argument("--key", default="your_api_key", help="API key")
    parser.add_argument("--model", help="Model to use")
    parser.add_argument("--max-tokens", default=150, type=int, help="Maximum tokens to generate")
    parser.add_argument("--temperature", default=0.7, type=float, help="Temperature for generation")
    parser.add_argument("--mode", choices=["all", "completion", "chat", "stream-completion", "stream-chat"], 
                        default="all", help="Test mode")
    
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}/v1"
    completion_url = f"{base_url}/completions"
    chat_completion_url = f"{base_url}/chat/completions"
    
    # Test data
    test_prompt = "Once upon a time in a land far away,"
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "I feel really happy today because I just got a promotion!"}
    ]
    
    # Run tests based on mode
    if args.mode in ["all", "completion"]:
        make_regular_completion_request(
            prompt=test_prompt,
            api_url=completion_url,
            api_key=args.key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            model=args.model
        )
    
    if args.mode in ["all", "stream-completion"]:
        make_streaming_completion_request(
            prompt=test_prompt,
            api_url=completion_url,
            api_key=args.key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            model=args.model
        )
    
    if args.mode in ["all", "chat"]:
        make_regular_chat_completion_request(
            messages=test_messages,
            api_url=chat_completion_url,
            api_key=args.key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            model=args.model
        )
    
    if args.mode in ["all", "stream-chat"]:
        make_streaming_chat_completion_request(
            messages=test_messages,
            api_url=chat_completion_url,
            api_key=args.key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            model=args.model
        )

if __name__ == "__main__":
    main()
