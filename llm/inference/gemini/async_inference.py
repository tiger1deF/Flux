"""
Asynchronous inference module for Google's Gemini API.

This module provides asynchronous functions for interacting with Gemini models,
handling model initialization and inference with thread-safe caching.
"""

import os
from functools import lru_cache
import asyncio
from functools import partial


@lru_cache(maxsize = 32)
def _initialize_model(
    model_name: str, 
    temperature: float, 
    max_tokens: int
):
    """
    Initialize and cache a Gemini model with specific configuration.
    
    :param model_name: Name of the Gemini model to initialize
    :type model_name: str
    :param temperature: Sampling temperature for generation
    :type temperature: float
    :param max_tokens: Maximum number of tokens in response
    :type max_tokens: int
    :return: Tuple of model instance and generation config
    :rtype: Tuple[genai.GenerativeModel, genai.GenerationConfig]
    """
    import google.generativeai as genai
    
    genai.configure(api_key = os.getenv('GEMINI_API_KEY'))
    generation_model = genai.GenerativeModel(model_name = model_name)
    
    generation_config = genai.GenerationConfig(
        temperature = float(temperature),       
        max_output_tokens = int(max_tokens)
    )
    
    return generation_model, generation_config


async def gemini_llm_inference(
    query: str,
    model_name: str = 'gemini-1.5-flash',
    temperature: float = 0.1,
    max_tokens: int = 4_000,
    system_prompt: str = None,
):
    """
    Perform asynchronous inference using Gemini API.
    
    :param query: Input text to send to the model
    :type query: str
    :param model_name: Name of the Gemini model to use
    :type model_name: str
    :param temperature: Sampling temperature
    :type temperature: float
    :param max_tokens: Maximum number of tokens in response
    :type max_tokens: int
    :param system_prompt: Optional system prompt to prepend
    :type system_prompt: str
    :return: Model's generated text response
    :rtype: str
    """
    prompt = f'{system_prompt}\n\n{query}' if system_prompt else query
    
    loop = asyncio.get_event_loop()
    generation_model, generation_config = await loop.run_in_executor(
        None,
        partial(_initialize_model, model_name, temperature, max_tokens)
    )
    
    response = await loop.run_in_executor(
        None,
        partial(
            generation_model.generate_content,
            prompt,
            generation_config = generation_config
        )
    )
    
    return response.text