import os
import numpy as np
from typing import Union, List
import aiohttp
import asyncio

from llm.models import BaseEmbeddingFunction


async def pulse_embeddings(text: Union[str, list[str]]):
    """
    Generate embeddings for a given text using Pulse API.
    
    :param text: Text to embed
    :type text: Union[str, list[str]]
    :return: Embedding vector or list of embedding vectors
    :rtype: Union[List[float], List[List[float]]]
    """
    try:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Apikey {os.getenv("PULSE_API_KEY")}'
        }
        
        if not isinstance(text, list):
            text = [text]
            
        payload = {"text": text}
        url = f"{os.getenv('INFERENCE_URL')}/pulse/v4/pf/lp/llm/get_embeddings"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('embeddings', [])
                else:
                    raise RuntimeError(f"API request failed with status {response.status}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to fetch embeddings using Pulse: {str(e)}")


async def pulse_fetch_embedding(session, url: str, payload: dict, headers: dict):
    """
    Fetch an embedding from Pulse API.
    
    :param session: Aiohttp ClientSession
    :type session: aiohttp.ClientSession
    :param url: API endpoint URL
    :type url: str
    :param payload: Request payload
    :type payload: dict
    :param headers: Request headers
    :type headers: dict
    :return: Embedding vector
    :rtype: np.array
    """
    async with session.post(
        url, 
        json = payload, 
        headers = headers
    ) as response:
        if response.status == 200:
            data = await response.json()
            return np.array(data.get('embeddings', [])[0])
        else:
            raise RuntimeError(f"Error fetching embedding: {response.status}")


async def pulse_get_embeddings(text: Union[str, list[str]]):
    """
    Get embeddings from Pulse API.
    
    :param text: Text to embed
    :type text: Union[str, list[str]]
    :return: Embedding vector or list of embedding vectors
    :rtype: Union[List[float], List[List[float]]]
    """
    if isinstance(text, str):
        text = [text]
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Apikey {os.getenv("PULSE_API_KEY")}'
    }
    url = f"{os.getenv('INFERENCE_URL')}/pulse/v4/pf/lp/llm/get_embeddings"

    async with aiohttp.ClientSession() as session:
        tasks = []
        for t in text:
            payload = {"text": [t]}
            tasks.append(pulse_fetch_embedding(session, url, payload, headers))
        
        embeddings = await asyncio.gather(*tasks)
        return embeddings


async def pulse_batch_embeddings(text: List[str], max_concurrent_requests: int = 10):
    """
    Batch embeddings using Pulse API.
    
    :param text: List of texts to embed
    :type text: List[str]
    :param max_concurrent_requests: Maximum number of concurrent requests
    :type max_concurrent_requests: int
    :return: List of embedding vectors
    :rtype: List[List[float]]
    """
    batches = [text[i:i + max_concurrent_requests] 
               for i in range(0, len(text), max_concurrent_requests)]
    
    all_embeddings = []
    for batch in batches:
        embeddings = await pulse_get_embeddings(batch)
        all_embeddings.extend(embeddings)
    
    return all_embeddings


base_pulse_embedder = BaseEmbeddingFunction(pulse_batch_embeddings)

