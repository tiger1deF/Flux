"""
Asynchronous inference module for Pulse LLM API.

This module provides asynchronous functions for interacting with the Pulse LLM API,
including model listing and inference operations with support for various external models.
"""

import aiohttp
import asyncio
import json
import os
from typing import Optional, List
from functools import lru_cache
import time


external_models = {
    'gemini-1.5-flash': 'azure-openai',   
    'gemini-1.5-pro': 'azure-openai',
    'gemini-1.0-flash': 'azure-openai',
    'gpt-4o': 'azure-openai',
    'gpt-4o-mini': 'azure-openai',
    'gpt-4-0613': 'azure-openai',
    'gpt-4-32k-0613': 'azure-openai',
    'gpt-3.5-turbo-0613': 'azure-openai',
    'gpt-3.5-turbo-16k': 'azure-openai',
    'gpt-4-vision': 'azure-openai',
    
    'claude-3-5-sonnet-20240620': 'anthropic',
    'claude-3-haiku-20240307': 'anthropic',
    'claude-3-opus-20240620': 'anthropic',
    'claude-3-sonnet-20240229': 'anthropic'
}
"""Mapping of external model names to their API providers"""


# Cache for internal models
_INTERNAL_MODELS_CACHE = {
    'models': None,
    'timestamp': 0,
    'ttl': 100_000_000
}

async def _get_internal_model_list() -> List[str]:
    """
    Get list of available internal models with caching.
    
    :return: List of internal model names
    :rtype: List[str]
    :raises ValueError: If model listing fails and no cache is available
    """
    current_time = time.time()
    
    # Return cached result if valid
    if (_INTERNAL_MODELS_CACHE['models'] is not None and 
        current_time - _INTERNAL_MODELS_CACHE['timestamp'] < _INTERNAL_MODELS_CACHE['ttl']):
        return _INTERNAL_MODELS_CACHE['models']
    
    try:
        url = f"{os.environ.get('INFERENCE_URL')}/pulse/v4/pf/lp/llm/get_names"
        headers = {
            'Authorization': f'Apikey {os.environ.get("PULSE_API_KEY")}'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                model_names = await response.json()
                internal_models = model_names['open_models']
                internal_model_names = [model['model_name'] for model in internal_models]
                
                # Update cache
                _INTERNAL_MODELS_CACHE['models'] = internal_model_names
                _INTERNAL_MODELS_CACHE['timestamp'] = current_time
                
                return internal_model_names
    
    except Exception as e:
        if _INTERNAL_MODELS_CACHE['models'] is not None:
            # Return stale cache on error
            return _INTERNAL_MODELS_CACHE['models']
        raise ValueError(f"Failed to list models: {e}")


async def pulse_llm_inference(
    query: str,
    model_name: str = "gpt-4o",
    temperature: float = 0.1,
    max_tokens: int = 4_000,
    top_p: float = 0.9,
    system_prompt: str = None
):
    """
    Perform asynchronous inference using Pulse LLM API.
    
    :param query: Input text to send to the model
    :type query: str
    :param model_name: Name of the model to use
    :type model_name: str
    :param temperature: Sampling temperature
    :type temperature: float
    :param max_tokens: Maximum number of tokens in response
    :type max_tokens: int
    :param top_p: Top-p sampling parameter
    :type top_p: float
    :param system_prompt: Optional system prompt to prepend
    :type system_prompt: str
    :return: Model's generated text response
    :rtype: str
    :raises ValueError: If model is not accessible or API returns an error
    """
    internal_models = await _get_internal_model_list()
    
    payload = {
        "model_name": model_name,
        "temperature": str(temperature),
        "max_output_tokens": str(max_tokens),
        "prompt": f'{system_prompt}\n\n{query}' if system_prompt else query,
        "top_p": str(top_p)
    }
    
    if model_name not in internal_models:
        external = True
        try:
            api_type = external_models[model_name]
            payload['api_type'] = api_type
            
            if api_type == 'gemini':
                payload['api_key'] = os.getenv('GEMINI_API_KEY')
            elif api_type == 'azure-openai':
                payload['api_key'] = os.getenv('OPENAI_AZURE_API_KEY')
            elif api_type == 'anthropic':
                payload['api_key'] = os.getenv('ANTHROPIC_API_KEY')
                
        except KeyError:
            raise ValueError(f"Model {model_name} not an accessible model!")
    else:
        external = False
    payload['external'] = external
    
    payload_str = json.dumps(payload)
    form_data = aiohttp.FormData()
    form_data.add_field('data', payload_str, content_type='application/json')
    
    headers = {'Authorization': f'Apikey {os.getenv("PULSE_API_KEY")}'}
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{os.getenv('INFERENCE_URL')}/pulse/v4/pf/lp/llm/send",
            headers=headers,
            data=form_data
        ) as response:
            try:
                if external:
                    model_output = await response.text()
                else:
                    response_json = await response.json()
                    if isinstance(response_json, dict) and 'error' in response_json:
                        raise ValueError(f"API Error: {response_json['error']}")

                    if isinstance(response_json, list):
                        model_output = response_json[0]['model_output']['text']
                    elif isinstance(response_json, dict):
                        model_output = response_json['model_output']['text']
                    else:
                        raise ValueError(f"Unexpected response format: {response_json}")
                        
            except Exception as e:
                raise ValueError(f"Error in model response: {str(e)}")
    
    return model_output
