"""
Models for LLM parameter handling and embedding function wrappers.

This module provides dataclasses and base classes for handling LLM parameters
and embedding functions with support for both synchronous and asynchronous operations.
"""

from dataclasses import dataclass
from typing import Any, Type, List, Union, get_type_hints, Callable, Awaitable, Optional
import inspect
import asyncio
import numpy as np
import threading


@dataclass
class LLMParameter:
    """
    Dataclass representing a parameter for an LLM completion function.
    
    :ivar name: Name of the parameter
    :type name: str
    :ivar type: Type of the parameter
    :type type: Type
    :ivar required: Whether the parameter is required
    :type required: bool
    :ivar default: Default value for the parameter
    :type default: Any
    """
    name: str
    type: Type
    required: bool
    default: Any = None
    
    
class BaseEmbeddingFunction:
    """
    Base class for wrapping embedding functions with standardized interface.
    
    Provides a unified interface for both synchronous and asynchronous embedding functions,
    with support for batching, type inference, and output formatting.
    
    :param embedding_fn: The embedding function to wrap
    :type embedding_fn: Callable
    :param dimension: Optional fixed dimension of embeddings
    :type dimension: Optional[int]
    :param cleanup_fn: Optional cleanup function
    :type cleanup_fn: Optional[Callable]
    """
    
    def __init__(self, 
                 embedding_fn: Callable,
                 dimension: Optional[int] = None,
                 cleanup_fn: Optional[Callable] = None):
        """
        Initialize the embedding function wrapper.
        
        :param embedding_fn: The embedding function to wrap
        :type embedding_fn: Callable
        :param dimension: Optional fixed dimension of embeddings
        :type dimension: Optional[int]
        :param cleanup_fn: Optional cleanup function
        :type cleanup_fn: Optional[Callable]
        """
        self.is_async = inspect.iscoroutinefunction(embedding_fn)
        self.embedding_fn = embedding_fn if self.is_async else self._make_async(embedding_fn)
        self.cleanup_fn = cleanup_fn
        self._lock = threading.Lock()
        
        type_hints = get_type_hints(embedding_fn)
        self._accepts_list = any(t == List[str] or t == list for t in type_hints.values())
        self._accepts_text = any(t == str for t in type_hints.values())
        
        return_type = type_hints.get('return')
        
        if return_type:
            self._format_output = lambda x: x
            if hasattr(return_type, '__args__'):
                if List[float] in return_type.__args__:
                    self.dimension = dimension or 0  # Will be set on first call
                    return
                    
        if dimension:
            self.dimension = dimension
            self._format_output = lambda x: x
            return
            
        # No type hints or dimension - infer from test run
        test_result = asyncio.run(self.embedding_fn("test input"))
        if isinstance(test_result, (list, tuple)):
            self._output_is_nested = isinstance(test_result[0], (list, tuple, np.ndarray))
            if self._output_is_nested:
                self.dimension = len(test_result[0])
                self._format_output = lambda x: [list(emb) for emb in x]
            else:
                self.dimension = len(test_result)
                self._format_output = lambda x: [list(x)]
                
        elif isinstance(test_result, np.ndarray):
            self._output_is_nested = len(test_result.shape) > 1
            if self._output_is_nested:
                self.dimension = test_result.shape[1]
                self._format_output = lambda x: x.tolist()
            else:
                self.dimension = test_result.shape[0]
                self._format_output = lambda x: [x.tolist()]
    
    
    def _make_async(self, sync_fn: Callable) -> Callable:
        """
        Convert a synchronous function to an asynchronous one.
        
        :param sync_fn: Synchronous function to convert
        :type sync_fn: Callable
        :return: Asynchronous wrapper function
        :rtype: Callable
        """
        async def async_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: sync_fn(*args, **kwargs))
        
        return async_wrapper
    
    
    async def _batch_embed(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Batch process multiple texts for embedding.
        
        :param texts: List of texts to embed
        :type texts: List[str]
        :param batch_size: Size of batches to process
        :type batch_size: int
        :return: List of embedding vectors
        :rtype: List[List[float]]
        """
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        all_results = []
        
        for batch in batches:
            async with asyncio.Semaphore(batch_size):
                tasks = [self.embedding_fn(text) for text in batch]
                batch_results = await asyncio.gather(*tasks)
                all_results.extend(batch_results)
        
        return all_results
    
    
    async def __call__(self, text: str) -> List[float]:
        """
        Get embedding for a single text.
        
        :param text: Text to embed
        :type text: str
        :return: Embedding vector
        :rtype: List[float]
        """
        async with asyncio.Lock():
            result = await self.embedding_fn(text)
        
            if self.dimension == 0:
                self.dimension = len(result)
            return self._format_output(result)[0] if isinstance(result, list) else result
    
    
    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Get embeddings for one or more texts.
        
        :param text: Text or list of texts to embed
        :type text: Union[str, List[str]]
        :return: Single embedding vector or list of vectors
        :rtype: Union[List[float], List[List[float]]]
        """
        is_list_input = isinstance(text, list)
        
        if self._accepts_list and not self._accepts_text:
            if not is_list_input:
                text = [text]
            async with asyncio.Lock():
                result = await self.embedding_fn(text)
                formatted = self._format_output(result)
                return formatted[0] if not is_list_input else formatted
            
        elif self._accepts_text and not self._accepts_list:
            if is_list_input:
                results = await self._batch_embed(text)
                return self._format_output(results)
            async with asyncio.Lock():
                result = await self.embedding_fn(text)
                return self._format_output(result)[0]
            
        else:
            async with asyncio.Lock():
                result = await self.embedding_fn(text)
                formatted = self._format_output(result)
                return formatted[0] if not is_list_input else formatted
    
    
    async def cleanup(self):
        """
        Clean up resources used by the embedding function.
        """
        if self.cleanup_fn:
            if inspect.iscoroutinefunction(self.cleanup_fn):
                await self.cleanup_fn()
            else:
                await asyncio.get_event_loop().run_in_executor(None, self.cleanup_fn)
    
    
    async def __aenter__(self):
        """
        Async context manager entry.
        
        :return: Self
        :rtype: BaseEmbeddingFunction
        """
        return self
    
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit.
        
        :param exc_type: Exception type if an error occurred
        :param exc_val: Exception value if an error occurred
        :param exc_tb: Exception traceback if an error occurred
        """
        await self.cleanup()