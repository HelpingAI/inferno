"""
Text generation module for Inferno.

This module provides functions for generating text from models.
"""

import torch
from typing import Dict, List, Optional, Any, Union, Tuple, Iterator
import time
import threading

from transformers import TextIteratorStreamer
from inferno.utils.logger import get_logger
from inferno.models.registry import ModelInfo

logger = get_logger(__name__)


def format_chat_messages(messages: List[Dict[str, str]], tokenizer) -> str:
    """
    Format chat messages for a model using the native chat template.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        tokenizer: The tokenizer to use

    Returns:
        Formatted prompt string
    """
    # Check if the tokenizer has a chat template
    if hasattr(tokenizer, 'apply_chat_template') and callable(tokenizer.apply_chat_template):
        try:
            # Apply the chat template directly using the native method
            # This will use the model's specific chat template
            prompt = tokenizer.apply_chat_template(
                messages,  # Pass messages directly
                tokenize=False,
                add_generation_prompt=True
            )
            logger.info(f"Using native chat template for model")

            # Check if the prompt is empty or None
            if not prompt:
                raise ValueError("Chat template returned empty prompt")

            return prompt
        except Exception as e:
            logger.warning(f"Error applying chat template: {e}")

    # Fallback to a simple format if no chat template is available
    logger.warning("No chat template available, using fallback format")
    prompt = ""
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")

        # Skip empty messages
        if not content:
            continue

        if role == "system":
            prompt += f"System: {content}\n\n"
        elif role == "user":
            prompt += f"User: {content}\n\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n\n"
        else:
            prompt += f"{role.capitalize()}: {content}\n\n"

    prompt += "Assistant: "
    return prompt


def generate_completion(model_info: ModelInfo, prompt: str, max_tokens: int = 100,
                       temperature: float = 0.7, top_p: float = 1.0,
                       frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
                       stop: Optional[Union[str, List[str]]] = None,
                       stream: bool = False) -> Union[Tuple[str, str], TextIteratorStreamer]:
    """
    Generate a completion from a model.

    Args:
        model_info: ModelInfo object containing the model and tokenizer
        prompt: The prompt to generate from
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        frequency_penalty: Frequency penalty
        presence_penalty: Presence penalty
        stop: Stop sequences
        stream: Whether to stream the output

    Returns:
        If stream=False: Tuple of (generated_text, finish_reason)
        If stream=True: TextIteratorStreamer that yields chunks of generated text
    """
    model = model_info.model
    tokenizer = model_info.tokenizer

    # Device-aware tensor placement
    device = getattr(model, 'device', None)
    if device is None:
        # Try to infer device from torch/xla
        try:
            import torch_xla.core.xla_model as xm # type: ignore[import]
            device = xm.xla_device()
        except ImportError:
            device = torch.device('cpu')

    # Check if it's a GGUF model (llama-cpp)
    if 'is_gguf' in model_info.metadata and model_info.metadata['is_gguf']:
        try:
            # Use llama-cpp API
            completion = model(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop
            )

            # Extract the generated text
            if isinstance(completion, dict) and 'choices' in completion:
                text = completion['choices'][0]['text']
                finish_reason = completion['choices'][0].get('finish_reason', 'length')
            else:
                text = completion.choices[0].text
                finish_reason = completion.choices[0].finish_reason or 'length'

            return text, finish_reason
        except Exception as e:
            logger.error(f"Error generating completion with GGUF model: {e}")
            # Fall back to a simple response
            return f"Error generating completion: {str(e)}", "error"

    # For Hugging Face models
    try:
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)

        # Set up generation parameters
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
        }

        # Add stop sequences if provided
        if stop:
            if isinstance(stop, str):
                stop = [stop]
            stop_token_ids = [tokenizer.encode(s, add_special_tokens=False)[-1] for s in stop]
            gen_kwargs["eos_token_id"] = stop_token_ids

        # Handle streaming
        if stream:
            # Create a streamer
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs["streamer"] = streamer

            # Start generation in a separate thread
            thread = threading.Thread(
                target=lambda: model.generate(input_ids, **gen_kwargs)
            )
            thread.start()

            return streamer
        else:
            # Generate
            with torch.no_grad():
                output = model.generate(input_ids, **gen_kwargs)

        # Decode the output
        generated_text = tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)

        return generated_text, "length"
    except Exception as e:
        logger.error(f"Error generating completion with HF model: {e}")
        return f"Error generating completion: {str(e)}", "error"


def generate_chat_completion(model_info: ModelInfo, messages: List[Dict[str, str]],
                            max_tokens: int = 100, temperature: float = 0.7,
                            top_p: float = 1.0, frequency_penalty: float = 0.0,
                            presence_penalty: float = 0.0,
                            stop: Optional[Union[str, List[str]]] = None,
                            stream: bool = False) -> Union[Tuple[str, str], TextIteratorStreamer]:
    """
    Generate a chat completion from a model.

    Args:
        model_info: ModelInfo object containing the model and tokenizer
        messages: List of message dictionaries with 'role' and 'content'
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        frequency_penalty: Frequency penalty
        presence_penalty: Presence penalty
        stop: Stop sequences
        stream: Whether to stream the output

    Returns:
        If stream=False: Tuple of (generated_text, finish_reason)
        If stream=True: TextIteratorStreamer that yields chunks of generated text
    """
    model = model_info.model
    tokenizer = model_info.tokenizer

    # Format the messages into a prompt
    prompt = format_chat_messages(messages, tokenizer)

    # Check if it's a GGUF model (llama-cpp)
    if 'is_gguf' in model_info.metadata and model_info.metadata['is_gguf']:
        try:
            # Check if the model has a create_chat_completion method
            if hasattr(model, 'create_chat_completion') and callable(model.create_chat_completion):
                logger.info("Using native llama-cpp chat completion API")

                # Use the messages directly with the llama-cpp API
                # This will use the model's native chat format handling
                completion = model.create_chat_completion(
                    messages=messages,  # Pass messages directly
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop
                )

                # Extract the generated text
                logger.debug(f"Chat completion response: {completion}")
                if isinstance(completion, dict) and 'choices' in completion:
                    # Handle dictionary response
                    choice = completion['choices'][0]
                    if 'message' in choice and choice['message']:
                        if isinstance(choice['message'], dict) and 'content' in choice['message']:
                            text = choice['message']['content']
                        else:
                            # Try to access as an object
                            text = getattr(choice['message'], 'content', '')
                    else:
                        # Some models might return text directly
                        text = choice.get('text', '')
                    finish_reason = choice.get('finish_reason', 'length')
                else:
                    # Handle object response
                    choice = completion.choices[0]
                    if hasattr(choice, 'message') and choice.message:
                        text = getattr(choice.message, 'content', '')
                    else:
                        # Some models might return text directly
                        text = getattr(choice, 'text', '')
                    finish_reason = getattr(choice, 'finish_reason', 'length') or 'length'

                return text, finish_reason
            else:
                logger.warning("llama-cpp model doesn't support create_chat_completion, falling back to regular completion")
                # Fall back to regular completion
                return generate_completion(
                    model_info, prompt, max_tokens, temperature,
                    top_p, frequency_penalty, presence_penalty, stop
                )
        except Exception as e:
            logger.error(f"Error generating chat completion with GGUF model: {e}", exc_info=True)
            # Log more details about the model and messages for debugging
            logger.debug(f"Model info: {model_info.model_id}")
            logger.debug(f"Messages: {messages}")
            # Fall back to a simple response
            return f"Error generating chat completion: {str(e)}", "error"

    # For Hugging Face models
    try:
        logger.info("Using Hugging Face model for chat completion")

        # Check if the model has a chat_model attribute or method
        if hasattr(model, 'chat') and callable(model.chat):
            # Some models have a direct chat method
            logger.info("Using model.chat() method")
            response = model.chat(messages)
            return response, "length"

        # Try using the pipeline approach if available
        try:
            from transformers import pipeline
            logger.info("Attempting to use transformers pipeline")

            # Create a text-generation pipeline with the model
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=model.device.index if hasattr(model.device, 'index') else -1)

            # Format the prompt using the chat template
            formatted_prompt = format_chat_messages(messages, tokenizer)
            logger.debug(f"Formatted prompt for pipeline: {formatted_prompt}")

            # Check if the formatted prompt is empty or None
            if not formatted_prompt:
                logger.warning("Formatted prompt is empty or None, using fallback format")
                # Use a simple fallback format
                formatted_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages if msg.get('content')]) + "\nassistant: "
                logger.debug(f"Fallback formatted prompt: {formatted_prompt}")

            # Handle streaming
            if stream:
                # Create a streamer
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

                # Start generation in a separate thread
                # Note: The first argument to the pipeline must be the text input as a positional argument
                # The rest can be keyword arguments
                generation_kwargs = {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": (temperature > 0),
                    "streamer": streamer
                }

                thread = threading.Thread(target=lambda: pipe(formatted_prompt, **generation_kwargs))
                thread.start()

                return streamer
            else:
                # Generate the response using the formatted prompt
                result = pipe(formatted_prompt, max_new_tokens=max_tokens, temperature=temperature, top_p=top_p, do_sample=(temperature > 0))

            # Extract the generated text
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and 'generated_text' in result[0]:
                    # Get the full generated text
                    full_text = result[0]['generated_text']
                    logger.debug(f"Full pipeline generated text: {full_text}")

                    # Extract only the new content (remove the prompt)
                    if full_text.startswith(formatted_prompt):
                        generated_text = full_text[len(formatted_prompt):].strip()
                    else:
                        # If we can't find the prompt, just return the full text
                        generated_text = full_text

                    logger.debug(f"Extracted generated text: {generated_text}")
                    return generated_text, "length"
        except Exception as pipe_error:
            logger.error(f"Pipeline approach failed: {pipe_error}", exc_info=True)
            logger.warning("Falling back to standard approach.")

        # Use the apply_chat_template approach
        logger.info("Using apply_chat_template approach")
        try:
            # Apply the chat template to get the formatted input
            inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")

            # Log the tokenized input for debugging
            logger.debug(f"Tokenized input shape: {inputs.shape}")

            # Decode the input to see what we're actually feeding to the model
            input_text = tokenizer.decode(inputs[0], skip_special_tokens=False)
            logger.debug(f"Input text: {input_text}")

            # Move to the appropriate device
            input_ids = inputs.to(model.device)

            # Set up generation parameters
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": temperature > 0,
            }

            # Add stop sequences if provided
            if stop:
                if isinstance(stop, str):
                    stop = [stop]
                stop_token_ids = [tokenizer.encode(s, add_special_tokens=False)[-1] for s in stop]
                gen_kwargs["eos_token_id"] = stop_token_ids

            # Handle streaming
            if stream:
                # Create a streamer
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                gen_kwargs["streamer"] = streamer

                # Start generation in a separate thread
                thread = threading.Thread(
                    target=lambda: model.generate(input_ids, **gen_kwargs)
                )
                thread.start()

                return streamer
            else:
                # Generate
                with torch.no_grad():
                    outputs = model.generate(input_ids, **gen_kwargs)

            # Decode the full output
            full_output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.debug(f"Full output text: {full_output_text}")

            # Decode the input portion to find where to start the generated text
            input_text_clean = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            logger.debug(f"Input text (clean): {input_text_clean}")

            # Extract only the new content
            if full_output_text.startswith(input_text_clean):
                generated_text = full_output_text[len(input_text_clean):].strip()
            else:
                # Decode the output, skipping the input portion by token count
                response = outputs[0][len(input_ids[0]):]
                generated_text = tokenizer.decode(response, skip_special_tokens=True)

            logger.debug(f"Generated text: {generated_text}")
            return generated_text, "length"
        except Exception as template_error:
            logger.error(f"apply_chat_template approach failed: {template_error}", exc_info=True)
            logger.warning("Falling back to standard approach.")

        # Otherwise use the standard approach with the formatted prompt
        logger.info("Using standard generation with chat template")

        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)

        # Log the prompt and tokenized input for debugging
        logger.debug(f"Standard approach prompt: {prompt}")
        logger.debug(f"Standard approach tokenized input shape: {input_ids.shape}")

        # Set up generation parameters
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
        }

        # Add stop sequences if provided
        if stop:
            if isinstance(stop, str):
                stop = [stop]
            stop_token_ids = [tokenizer.encode(s, add_special_tokens=False)[-1] for s in stop]
            gen_kwargs["eos_token_id"] = stop_token_ids

        # Handle streaming
        if stream:
            # Create a streamer
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs["streamer"] = streamer

            # Start generation in a separate thread
            thread = threading.Thread(
                target=lambda: model.generate(input_ids, **gen_kwargs)
            )
            thread.start()

            return streamer
        else:
            # Generate
            with torch.no_grad():
                output = model.generate(input_ids, **gen_kwargs)

        # Decode the full output
        full_output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.debug(f"Standard approach full output: {full_output_text}")

        # Decode the input portion to find where to start the generated text
        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        logger.debug(f"Standard approach input text: {input_text}")

        # Extract only the new content
        if full_output_text.startswith(input_text):
            generated_text = full_output_text[len(input_text):].strip()
        else:
            # Decode the output, skipping the input portion by token count
            generated_text = tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)

        logger.debug(f"Standard approach generated text: {generated_text}")
        return generated_text, "length"
    except Exception as e:
        logger.error(f"Error generating chat completion with HF model: {e}", exc_info=True)
        # Log more details about the model and messages for debugging
        logger.debug(f"Model info: {model_info.model_id}")
        logger.debug(f"Messages: {messages}")
        return f"Error generating chat completion: {str(e)}", "error"
