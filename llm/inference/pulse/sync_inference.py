"""
Synchronous inference module for Pulse LLM API.

This module provides synchronous functions for interacting with the Pulse LLM API,
including model listing and inference operations with support for various external models.
"""

import requests
import json
import os
from functools import lru_cache


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
"""
Mapping of external model names to their API providers.

:ivar external_models: Dictionary mapping model names to their provider services
:type external_models: Dict[str, str]
"""


@lru_cache(maxsize = 1)
def _get_internal_model_list():
    """
    Get cached list of available internal models.
    
    :return: List of internal model names
    :rtype: List[str]
    :raises ValueError: If model listing fails
    """
    try:
        url = f"{os.environ.get('INFERENCE_URL')}/pulse/v4/pf/lp/llm/get_names"
        headers = {
            'Authorization': f'Apikey {os.environ.get("PULSE_API_KEY")}'
        }
        
        response = requests.get(url, headers=headers)
        model_names = response.json()
        internal_models = model_names['open_models']
        internal_model_names = [model['model_name'] for model in internal_models]    
        return internal_model_names
    
    except Exception as e:
        raise ValueError(f"Failed to list models: {e}")


def pulse_llm_inference(
    query: str,
    model_name: str = "gpt-4o",
    temperature: float = 0.1,
    max_tokens: int = 4_000,
    top_p: float = 0.9,
    system_prompt: str = None
):
    """
    Perform synchronous inference using Pulse LLM API.
    
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
    
    internal_models = _get_internal_model_list()
    
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
    form_data = {
        'data': (None, payload_str, 'application/json')
    }

    headers = {'Authorization': f'Apikey {os.getenv("PULSE_API_KEY")}'}
    
    response = requests.post(
        f"{os.getenv('INFERENCE_URL')}/pulse/v4/pf/lp/llm/send",
        headers = headers,
        files = form_data
    )
    
    try:
        if external:
            model_output = response.text
        else:
            response_json = response.json()
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
