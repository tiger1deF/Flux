"""
Shared tokenizer instance for text processing.

This module provides a shared tokenizer instance using the cl100k_base encoding
from tiktoken, which is compatible with most modern language models. It also
provides async-safe encoding and decoding functions to prevent event loop blocking.
"""

import asyncio
from typing import List, Union
from enum import Enum

import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")


async def encode_async(
    text: str,
    chunk_size: int = 512
) -> List[int]:
    """
    Asynchronously encode text into tokens while preventing event loop blocking.
    
    For small texts, processes directly. For large texts, splits into chunks
    and processes in parallel to maintain responsiveness.
    
    :param text: The text to encode into tokens
    :type text: str
    :param chunk_size: Maximum characters per chunk for large texts
    :type chunk_size: int
    :return: List of integer tokens representing the encoded text
    :rtype: List[int]
    """
    loop = asyncio.get_running_loop()
    
    if len(text) < chunk_size:
        return await loop.run_in_executor(None, tokenizer.encode, text)
    
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    token_chunks = await asyncio.gather(
        *[loop.run_in_executor(None, tokenizer.encode, chunk) 
          for chunk in chunks]
    )
    
    return [token for chunk in token_chunks for token in chunk]


async def decode_async(
    tokens: Union[List[int], bytes],
    chunk_size: int = 512
) -> str:
    """
    Asynchronously decode tokens back into text while preventing event loop blocking.
    
    :param tokens: List of integer tokens or bytes to decode
    :type tokens: Union[List[int], bytes]
    :param chunk_size: Maximum tokens per chunk for large sequences
    :type chunk_size: int
    :return: Decoded text string
    :rtype: str
    """
    loop = asyncio.get_running_loop()
    
    if isinstance(tokens, bytes) or len(tokens) < chunk_size:
        return await loop.run_in_executor(None, tokenizer.decode, tokens)
    
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    text_chunks = await asyncio.gather(
        *[loop.run_in_executor(None, tokenizer.decode, chunk) 
          for chunk in chunks]
    )
    
    return ''.join(text_chunks)


async def batch_token_lengths(
    texts: List[str]
) -> List[int]:
    """
    Get token counts for multiple texts in parallel.
    
    :param texts: List of texts to get token counts for
    :type texts: List[str]
    :return: List of token counts corresponding to each text
    :rtype: List[int]
    """
    token_lists = await asyncio.gather(*[
        encode_async(text) for text in texts
    ])
    
    return [len(tokens) for tokens in token_lists]


class SliceType(str, Enum):
    """
    Types of slicing to apply to item content
    
    :cvar START: Truncate the start of the message
    :cvar END: Truncate the end of the message
    :cvar MIDDLE: Truncate the middle of the message
    """
    START = "start"
    END = "end"
    MIDDLE = "middle"
    

async def slice_text(
    text: str, 
    slice_type: SliceType,
    max_tokens: int
    
) -> str:
    """
    Slice text to a maximum number of tokens.
    
    :param text: Text to slice
    :type text: str
    :param slice_type: Type of slice to apply
    :type slice_type: SliceType 
    :param max_tokens: Maximum number of tokens
    :type max_tokens: int
    :return: Truncated text
    :rtype: str
    """
    
    tokens = await encode_async(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    max_tokens -= 3 # For slice explanation
    if slice_type == SliceType.START:
        truncated_tokens = tokens[max_tokens:]
        truncated_text = await decode_async(truncated_tokens) 
        truncated_text = "[...TRUNCATED START] " + truncated_text
    
    elif slice_type == SliceType.END:
        truncated_tokens = tokens[:max_tokens]
        truncated_text = await decode_async(truncated_tokens) + " [TRUNCATED END...]"
    
    elif slice_type == SliceType.MIDDLE:
        start_tokens = tokens[:max_tokens // 2]
        end_tokens = tokens[-max_tokens // 2:]
        start_text = await decode_async(start_tokens)
        end_text = await decode_async(end_tokens)
        truncated_text = start_text + " [...TRUNCATED MIDDLE...] " + end_text
    
    else:
        raise ValueError(f"Invalid slice type: {slice_type}")
        
    return truncated_text