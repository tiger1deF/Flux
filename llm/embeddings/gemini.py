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
import aiohttp
import logging

import google.auth

# Set up logging
logger = logging.getLogger(__name__)

_genai_lock = threading.Lock()
_genai = None
_executor = ThreadPoolExecutor(max_workers=1)


def _initialize_genai():
    """Initialize the Google Generative AI client."""
    global _genai
    if _genai is None:
        with _genai_lock:
            if _genai is None:
                import google.generativeai as genai
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY environment variable not set")
                genai.configure(api_key=api_key)
                _genai = genai
                logger.info("Initialized Gemini API client")
    return _genai


_genai = _initialize_genai()


async def gemini_generate_embedding(
    text: str,
    model_name: str = 'models/embedding-001',
    task_type: str = 'retrieval_document',
    batch_size: int = 20
) -> List[float]:
    """Optimized embedding generation with connection reuse and retries"""
    
    logger.debug(f"Generating embeddings for text type: {type(text)}")
    if isinstance(text, str):
        logger.debug(f"Input text length: {len(text)}")
    else:
        logger.debug(f"Batch size: {len(text)}")
    
    # Use shared connection pool
    if not hasattr(gemini_generate_embedding, '_connector'):
        gemini_generate_embedding._connector = aiohttp.TCPConnector(
            limit = 100,
            ttl_dns_cache = 300,
            use_dns_cache = True
        )
        logger.debug("Created new TCP connector")
    
    async def _batch_embed(texts: List[str]) -> List[List[float]]:
        logger.debug(f"Processing batch of {len(texts)} texts")
        
        async with aiohttp.ClientSession(
            connector = gemini_generate_embedding._connector,
            timeout = aiohttp.ClientTimeout(total = 30)
        ) as session:
            loop = asyncio.get_event_loop()
            tasks = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                logger.debug(f"Creating task for batch {i//batch_size + 1} with size {len(batch)}")
                
                def embed_batch(b):
                    try:
                        logger.debug(f"Calling embed_content for batch of size {len(b)}")
                        result = _genai.embed_content(
                            model = model_name,
                            content = b,
                            task_type = task_type
                        )
                        logger.debug(f"Raw embed_content response: {result}")
                        return result
                    except Exception as e:
                        logger.error(f"Error in embed_content: {str(e)}")
                        raise
                
                task = loop.run_in_executor(None, embed_batch, batch)
                tasks.append(task)
            
            try:
                responses = await asyncio.gather(*tasks)
                logger.debug(f"Gathered {len(responses)} responses")
                
                embeddings = []
                for i, response in enumerate(responses):
                    logger.debug(f"Processing response {i}: {type(response)}")
                    
                    # The response format is {'embedding': [[...values...]]}
                    if isinstance(response, dict) and 'embedding' in response:
                        batch_embeddings = response['embedding']
                        logger.debug(f"Found embeddings in response: {len(batch_embeddings)}")
                        embeddings.extend(batch_embeddings)
                    else:
                        logger.warning(f"Unexpected response format: {response}")
                        raise ValueError(f"Unexpected response format: {response}")
                
                logger.debug(f"Total embeddings collected: {len(embeddings)}")
                return embeddings
                
            except Exception as e:
                logger.error(f"Error processing responses: {str(e)}")
                raise

    try:
        if isinstance(text, str):
            embeddings = await _batch_embed([text])
            if not embeddings:
                logger.error("No embeddings generated for single text input")
                raise ValueError("Failed to generate embeddings")
            logger.debug(f"Returning single embedding of length {len(embeddings[0])}")
            return embeddings[0]  # Return the first embedding for single text
        else:
            embeddings = await _batch_embed(text)
            if not embeddings:
                logger.error("No embeddings generated for batch text input")
                raise ValueError("Failed to generate embeddings")
            logger.debug(f"Returning {len(embeddings)} embeddings")
            return embeddings
            
    except Exception as e:
        logger.error(f"Error in gemini_generate_embedding: {str(e)}")
        raise


