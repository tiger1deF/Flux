"""
Test suite for language model functionality.

This module provides tests for various language model configurations and inference modes,
including both Pulse and Gemini models with synchronous and asynchronous inference.
"""

import asyncio
import time
import logging

from llm import (
    LLM,
    pulse_llm_sync_inference,
    pulse_llm_async_inference,
    gemini_llm_sync_inference,
    gemini_llm_async_inference
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_llm():
    """Test language model configurations and inference."""
    
    # Test Pulse LLM (Async)
    logger.info("\n=== Testing Pulse LLM (Async) ===")
    start_time = time.time()
    llm = LLM(
        pulse_llm_async_inference, 
        max_tokens = 1000
    )
    logger.info(f"LLM initialization took {time.time() - start_time:.2f}s\n")
    
    async def run_async_tests():
        # Single async call
        start_time = time.time()
        response = await llm('How are you?')
        logger.info(f"Async call took {time.time() - start_time:.2f}s")
        logger.info(f"Response: {response}\n")
        
        # Batch operations
        texts = ["How are you?", "How are you?", "How are you?"]
        start_time = time.time()
        responses = await asyncio.gather(*[llm(text) for text in texts])
        logger.info(f"Async batch took {time.time() - start_time:.2f}s")
        logger.info(f"Number of responses: {len(responses)}\n")
        
        return responses

    # Run async tests
    asyncio.run(run_async_tests())
    
    # Test Gemini LLM (Sync)
    logger.info("\n=== Testing Gemini LLM (Sync) ===")
    start_time = time.time()
    llm = LLM(gemini_llm_sync_inference, model_name='gemini-1.5-pro')
    logger.info(f"LLM initialization took {time.time() - start_time:.2f}s\n")
    
    # Single sync call
    start_time = time.time()
    response = llm('Hello, how are you?')
    logger.info(f"Sync call took {time.time() - start_time:.2f}s")
    logger.info(f"Response: {response}\n")
    
    # Test token limiting
    logger.info("\n=== Testing Token Limiting ===")
    start_time = time.time()
    llm = LLM(
        pulse_llm_async_inference, 
        model_name = 'Mixtral-8x22B-Instruct-v0.1', 
        input_tokens = 1000, 
        max_tokens = 2000
    )
    logger.info(f"LLM initialization took {time.time() - start_time:.2f}s\n")
    
    # Test with long input
    long_input = "Hello, " * 1000  # Create a long input text
    start_time = time.time()
    response = llm(long_input)
    logger.info(f"Token-limited call took {time.time() - start_time:.2f}s")
    logger.info(f"Response length: {len(response)}\n")
    
    # Test system prompt
    logger.info("\n=== Testing System Prompt ===")
    start_time = time.time()
    llm = LLM(
        gemini_llm_sync_inference,
        model_name='gemini-1.5-pro',
        system_prompt="You are a helpful AI assistant."
    )
    logger.info(f"LLM initialization took {time.time() - start_time:.2f}s\n")
    
    start_time = time.time()
    response = llm('What are you?')
    logger.info(f"System prompt call took {time.time() - start_time:.2f}s")
    logger.info(f"Response: {response}\n")

