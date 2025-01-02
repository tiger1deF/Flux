"""
Language model implementations and interfaces.

This module provides base classes and utilities for working with LLMs,
including parameter management, type checking, and thread-safe execution.
"""

from typing import (
    Any, List, Union, Callable, Optional, 
    Dict, ClassVar, Coroutine
)
import inspect
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pydantic import BaseModel, Field, PrivateAttr
import json

    
class BaseEmbeddingFunction(BaseModel):
    """Base class for embedding functions that automatically configures and optimizes embedding execution.
    
    :ivar embedding_fn: The embedding function to be wrapped
    :type embedding_fn: Callable
    :ivar dimension: The dimension of the embedding output
    :type dimension: Optional[int]
    :ivar accepts_list: Whether the embedding function accepts a list of texts
    :type accepts_list: bool
    :ivar _is_async: Whether the embedding function is asynchronous
    :type _is_async: bool
    :ivar _thread_pool: The thread pool executor for asynchronous execution
    :type _thread_pool: ClassVar[ThreadPoolExecutor]
    :ivar _lock: The lock for thread-safe execution
    :type _lock: ClassVar[threading.Lock]
    :ivar _fn_module: The module of the embedding function
    :type _fn_module: str
    :ivar _fn_name: The name of the embedding function
    :type _fn_name: str
    """
    
    embedding_fn: Callable
    dimension: Optional[int] = Field(gt=0, default=None)
    accepts_list: bool = True
    batch_size: int = 16
    
    # Private attributes must be declared before __init__
    _is_async: bool = PrivateAttr(default=False)
    _thread_pool: ClassVar[ThreadPoolExecutor] = ThreadPoolExecutor()
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _fn_module: str = PrivateAttr(default="")
    _fn_name: str = PrivateAttr(default="")
    

    def __init__(
        self, 
        embedding_fn: Callable, 
        dimension: Optional[int] = None, 
        batch_size: int = 5,
        **kwargs
    ):
        # First call parent's init to set up Pydantic
        super().__init__(
            embedding_fn = lambda x: embedding_fn(x, **kwargs),
            dimension = dimension,
            batch_size = batch_size
        )
        
        # Then set private attributes
        self._fn_module = embedding_fn.__module__
        self._fn_name = embedding_fn.__qualname__
        self._is_async = asyncio.iscoroutinefunction(embedding_fn)
        
        # Determine dimension through dry run if not provided
        if self.dimension is None:
            self.dimension = self._determine_dimension()


    async def serialize(self) -> str:
        """
        Serialize the embedding function configuration.
        
        :return: JSON string containing serialized configuration
        :rtype: str
        """
        return json.dumps({
            "module": self._fn_module,
            "name": self._fn_name,
            "dimension": self.dimension,
            "accepts_list": self.accepts_list,
            "is_async": self._is_async
        })


    @classmethod
    async def deserialize(cls, data: str) -> 'BaseEmbeddingFunction':
        """
        Deserialize an embedding function from its configuration.
        
        :param data: JSON string containing serialized configuration
        :type data: str
        :return: Reconstructed embedding function
        :rtype: BaseEmbeddingFunction
        :raises ImportError: If the module or function cannot be imported
        :raises ValueError: If the configuration is invalid
        """
        try:
            config = json.loads(data)
            
            # Import the module and get the function
            module = __import__(config["module"], fromlist=[config["name"]])
            embedding_fn = getattr(module, config["name"].split(".")[-1])
            
            # Create new instance with restored configuration
            instance = cls(
                embedding_fn=embedding_fn,
                dimension=config["dimension"]
            )
            
            # Restore additional attributes
            instance.accepts_list = config["accepts_list"]
            instance._is_async = config["is_async"]
            
            return instance
            
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not load embedding function: {str(e)}")
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid embedding function configuration: {str(e)}")


    def model_dump(self) -> Dict[str, Any]:
        """
        Custom model dump for Pydantic serialization.
        
        :return: Dictionary containing serializable configuration
        :rtype: Dict[str, Any]
        """
        return {
            "module": self._fn_module,
            "name": self._fn_name,
            "dimension": self.dimension,
            "accepts_list": self.accepts_list,
            "is_async": self._is_async
        }


    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> 'BaseEmbeddingFunction':
        """
        Custom model validation for Pydantic deserialization.
        
        :param data: Dictionary containing serialized configuration
        :type data: Dict[str, Any]
        :return: Reconstructed embedding function
        :rtype: BaseEmbeddingFunction
        """
        # Import the module and get the function
        module = __import__(data["module"], fromlist=[data["name"]])
        embedding_fn = getattr(module, data["name"].split(".")[-1])
        
        # Create new instance with restored configuration
        instance = cls(
            embedding_fn=embedding_fn,
            dimension=data["dimension"]
        )
        
        # Restore additional attributes
        instance.accepts_list = data["accepts_list"]
        instance._is_async = data["is_async"]
        
        return instance

    def _determine_dimension(self) -> int:
        """
        Performs a dry run to determine embedding dimension.
        
        Automatically handles both sync and async embedding functions by coercing
        them into the appropriate context for the test run.
        
        :return: The dimension of the embedding output
        :rtype: int
        :raises ValueError: If dimension cannot be determined from the output
        """
        try:
            sample_text = "This is a test sentence for dimension detection."
            
            # Get test embedding
            result = self.embedding_fn(sample_text)
            
            # Handle coroutine if async function
            if asyncio.iscoroutine(result):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(result)
                finally:
                    loop.close()
                    
            # Format and get dimension
            formatted = self._format_output(result)
            if not formatted or not formatted[0]:
                raise ValueError("Embedding function returned empty result")
            return len(formatted[0])
            
        except Exception as e:
            raise ValueError(f"Could not determine embedding dimension: {str(e)}")


    def __call__(self, text: Union[str, List[str]]) -> Union[List[List[float]], Coroutine[Any, Any, List[List[float]]]]:
        """
        Call implementation that works in both sync and async contexts.
        
        Automatically handles context switching between sync/async:
        - If called in async context: returns a coroutine
        - If called in sync context: returns the result directly
        
        The embedding function will be coerced into the appropriate context:
        - Async embedding_fn in sync context: runs in event loop
        - Sync embedding_fn in async context: runs in thread pool
        
        :param text: Single text or list of texts to embed
        :type text: Union[str, List[str]]
        :return: Embeddings result or coroutine that will return embeddings
        :rtype: Union[List[List[float]], Coroutine[Any, Any, List[List[float]]]]
        """
        if isinstance(text, str):
            text = [text]
            
        try:
            # We're in an async context
            loop = asyncio.get_running_loop()
            if asyncio.iscoroutinefunction(self.embedding_fn):
                # Both async - direct return
                return self._async_impl(text)
            else:
                # Sync fn in async context - wrap in thread pool
                return loop.run_in_executor(
                    self._thread_pool,
                    self._sync_impl,
                    text
                )
        except RuntimeError:
            # We're in a sync context
            if asyncio.iscoroutinefunction(self.embedding_fn):
                # Async fn in sync context - run in new loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self._async_impl(text))
                finally:
                    loop.close()
            # Both sync - direct return
            return self._sync_impl(text)


    async def acall(self, text: Union[str, List[str]]) -> List[List[float]]:
        """Asynchronous call implementation."""
        if isinstance(text, str):
            text = [text]
            
        if asyncio.iscoroutinefunction(self.embedding_fn):
            return await self._async_impl(text)
        else:
            # If the function is sync but called asynchronously
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._thread_pool,
                self._sync_impl,
                text
            )

    async def _async_impl(self, texts: List[str]) -> List[List[float]]:
        """Optimized async implementation with batching."""
        if not self.accepts_list:
            results = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_results = await asyncio.gather(
                    *[self.embedding_fn(t) for t in batch]
                )
                results.extend([self._format_output(r)[0] for r in batch_results])
            return results
        
        result = await self.embedding_fn(texts)
        return self._format_output(result)

    def _sync_impl(self, texts: List[str]) -> List[List[float]]:
        """Thread-safe synchronous implementation."""
        with self._lock:
            if not self.accepts_list:
                results = []
                for t in texts:
                    result = self.embedding_fn(t)
                    # Handle case where embedding_fn returns a coroutine
                    if asyncio.iscoroutine(result):
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            result = loop.run_until_complete(result)
                        finally:
                            loop.close()
                    results.append(self._format_output(result)[0])
                return results
            
            result = self.embedding_fn(texts)
            # Handle case where embedding_fn returns a coroutine
            if asyncio.iscoroutine(result):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(result)
                finally:
                    loop.close()
            return self._format_output(result)

    def _format_output(self, result: Union[List[float], Dict, np.ndarray]) -> List[List[float]]:
        """Format the embedding output consistently."""
        if isinstance(result, dict):
            embeddings = result.get('embedding', result.get('embeddings', result))
            if not isinstance(embeddings, (list, np.ndarray)):
                raise ValueError(f"Expected list or array of embeddings, got {type(embeddings)}")
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            if not isinstance(embeddings[0], (list, np.ndarray)):
                embeddings = [embeddings]
            return embeddings
        elif isinstance(result, (list, np.ndarray)):
            if isinstance(result, np.ndarray):
                result = result.tolist()
            if not isinstance(result[0], (list, np.ndarray)):
                return [result]
            return result
        else:
            raise ValueError(f"Unexpected embedding format: {type(result)}")


class EmbeddingPool:
    def __init__(self, max_size: int = 1000):
        self.pool = {}
        self.max_size = max_size
        self._lock = asyncio.Lock()
        
    async def get(self, key: str) -> Optional[np.ndarray]:
        async with self._lock:
            return self.pool.get(key)
            
    async def add(self, key: str, embedding: np.ndarray):
        async with self._lock:
            if len(self.pool) >= self.max_size:
                # Evict oldest
                oldest = next(iter(self.pool))
                del self.pool[oldest]
            self.pool[key] = embedding