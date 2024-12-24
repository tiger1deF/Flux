"""
Gemini embedding model implementation.

This module provides functions for generating embeddings using Google's Gemini API,
with thread-safe initialization and caching of the client.
"""

from typing import List
import os
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import google.auth

from llm.models import BaseEmbeddingFunction


_genai_lock = threading.Lock()
_genai = None
_executor = ThreadPoolExecutor(max_workers=1)


@lru_cache(maxsize = 1)
def _initialize_genai():
    """
    Initialize the Google Generative AI client.
    
    :return: Google Generative AI client
    :rtype: google.generativeai.GenerativeModel
    """
    global _genai
    if _genai is None:
        with _genai_lock:
            if _genai is None:
                import google.generativeai as genai
                genai.configure(api_key = os.getenv("GEMINI_API_KEY"))
                _genai = genai
    return _genai


async def gemini_generate_embedding(
    text: str,
    model_name: str = 'models/embedding-001',
    task_type: str = 'retrieval_document'
) -> List[float]:
    """
    Generate embeddings for a given text using Gemini API.
    
    :param text: Text to embed
    :type text: str
    :return: Embedding vector
    :rtype: List[float] 
    """
    genai = _initialize_genai()
    import os
   
    def _embed():
        try:
            response = genai.embed_content(
                model = model_name,
                content = text,
                task_type = task_type
            )
        except google.auth.exceptions.DefaultCredentialsError:
            raise Exception("Gemini API key is not set! Set it as GEMINI_API_KEY environment variable.")
        return response['embedding']
    
    return await asyncio.get_event_loop().run_in_executor(_executor, _embed)


base_gemini_embedder = BaseEmbeddingFunction(
    embedding_fn = gemini_generate_embedding,
    dimension = 768,
)
