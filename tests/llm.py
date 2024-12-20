"""
Test suite for language model functionality.

This module provides tests for various language model configurations and inference modes,
including both Pulse and Gemini models with synchronous and asynchronous inference.
"""

import asyncio

from llm import (
    LLM,
    pulse_llm_sync_inference,
    pulse_llm_async_inference,
    gemini_llm_sync_inference,
    gemini_llm_async_inference
)

def test_llm():
    """
    Test language model configurations and inference.
    
    Tests the following configurations:
    1. Pulse LLM with async inference (GPT-4)
    2. Gemini LLM with sync inference (Gemini 1.5 Pro)
    3. Pulse LLM with token limits
    
    Each test verifies response generation and token handling.
    """
    print(f'Testing pulse llm...')
    llm = LLM(pulse_llm_async_inference, model_name = 'gpt-4o')
    response = asyncio.run(llm('How are you?'))
    print(response)
    
    
    llm = LLM(gemini_llm_sync_inference, model_name = 'gemini-1.5-pro')
    response = llm('Hello, how are you?')
    print(response)
    
    print('Testing max input tokens...')
    llm = LLM(
        pulse_llm_async_inference, 
        model_name = 'gpt-4o', 
        input_tokens = 1000, 
        max_tokens = 2000
    )
    response = asyncio.run(llm('Hello, how are you?'))
    print(response)

