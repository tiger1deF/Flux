"""
Language model implementations and interfaces.

This module provides base classes and utilities for working with LLMs,
including parameter management, type checking, and thread-safe execution.
"""

from dataclasses import dataclass
from typing import Any, Type, List, Union, get_type_hints, Callable, Awaitable, Optional, Coroutine, get_origin, get_args, Sequence, Iterable
import inspect
import asyncio
from pydantic import BaseModel, Field
import numpy as np


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
    
    
class BaseEmbeddingFunction(BaseModel):
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
    
    embedding_fn: Callable = Field(description = "Function that generates embeddings")
    dimension: Optional[int] = Field(default = None, description = "Dimension of embeddings")
    cleanup_fn: Optional[Callable] = Field(default = None, description = "Optional cleanup function")
    batch_size: Optional[int] = Field(default = None, description = "Batch size for processing")
    
    # Private fields determined from function signature or sample runs
    accepts_list: Optional[bool] = Field(default = None, exclude = True)
    accepts_text: Optional[bool] = Field(default = None, exclude = True)
    is_async: bool = Field(default = False, exclude = True)
    output_formatter: Optional[Callable] = Field(default = None, exclude = True)
    has_determined_format: bool = Field(default = False, exclude = True)
    input_param: Optional[inspect.Parameter] = Field(default=None, exclude=True)
    
    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    def __init__(self, embedding_fn: Callable, **kwargs):
        super().__init__(embedding_fn=embedding_fn, **kwargs)
        self.is_async = inspect.iscoroutinefunction(embedding_fn)
        
        # Inspect function signature and type hints
        sig = inspect.signature(embedding_fn)
        type_hints = get_type_hints(embedding_fn)
        
        # Get primary input parameter
        self.input_param = next(
            (param for param in sig.parameters.values() 
             if param.name in {'text', 'input', 'query', 'x', 'data'}),
            next(iter(sig.parameters.values()))  # fallback to first parameter
        )
        
        # Try to determine accepts_list and accepts_text from type hints
        if self.input_param.name in type_hints:
            hint = type_hints[self.input_param.name]
            origin = get_origin(hint)
            args = get_args(hint)
            
            self.accepts_list = origin in {list, List, Sequence, Iterable}
            self.accepts_text = str in args if args else hint == str
        
        # Handle initialization based on context
        if self.dimension is None or self.accepts_list is None or self.accepts_text is None:
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - defer initialization
                self._pending_init = True
            except RuntimeError:
                # We're in a sync context - initialize now
                self._sync_run_sample_inference()

    def _sync_run_sample_inference(self):
        """Run sample inference synchronously"""
        sample_text = "Sample text for inference"
        
        try:
            if self.is_async:
                # Create a new event loop for initialization only
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self.embedding_fn(sample_text))
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)
            else:
                result = self.embedding_fn(sample_text)
            
            formatted = self._format_output(result)
            if self.dimension is None:
                self.dimension = len(formatted[0])
                
        except Exception as e:
            raise ValueError(f"Could not determine embedding dimension: {str(e)}")

    async def _async_run_sample_inference(self):
        """Run sample inference asynchronously"""
        sample_text = "Sample text for inference"
        
        try:
            if self.is_async:
                result = await self.embedding_fn(sample_text)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.embedding_fn, sample_text)
            
            formatted = self._format_output(result)
            if self.dimension is None:
                self.dimension = len(formatted[0])
                
        except Exception as e:
            raise ValueError(f"Could not determine embedding dimension: {str(e)}")

    def _determine_format(self, output: Any) -> None:
        """One-time format determination"""
        if isinstance(output, (list, tuple)):
            if all(isinstance(x, (list, np.ndarray)) for x in output):
                self.output_formatter = lambda x: [
                    y.tolist() if isinstance(y, np.ndarray) else y 
                    for y in x
                ]
            else:
                self.output_formatter = lambda x: [
                    x.tolist() if isinstance(x, np.ndarray) else x
                ]
        elif isinstance(output, np.ndarray):
            if len(output.shape) == 2:
                self.output_formatter = lambda x: x.tolist()
            else:
                self.output_formatter = lambda x: [x.tolist()]
        else:
            raise ValueError(f"Unsupported output format: {type(output)}")
        self.has_determined_format = True

    def _format_output(self, output: Any) -> List[List[float]]:
        """Fast path once format is determined"""
        if not self.has_determined_format:
            self._determine_format(output)
        return self.output_formatter(output)

    async def _async_impl(self, text: Union[str, List[str]]) -> List[List[float]]:
        """Handle async execution with proper output formatting"""
        if isinstance(text, list) and not self.accepts_list:
            # More efficient batching
            batch_size = self.batch_size or len(text)
            results = []
            for i in range(0, len(text), batch_size):
                batch = text[i:i + batch_size]
                if self.is_async:
                    batch_results = await asyncio.gather(*[self.embedding_fn(t) for t in batch])
                else:
                    loop = asyncio.get_event_loop()
                    batch_results = await asyncio.gather(*[
                        loop.run_in_executor(None, self.embedding_fn, t) 
                        for t in batch
                    ])
                results.extend(batch_results)
            return [self._format_output(r)[0] for r in results]
        
        input_text = text if self.accepts_text else [text]
        if self.is_async:
            result = await self.embedding_fn(input_text)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.embedding_fn, input_text)
        
        formatted = self._format_output(result)
        return formatted if self.accepts_list else formatted[:1]

    def _sync_impl(self, text: Union[str, List[str]]) -> List[List[float]]:
        """Handle sync execution with proper output formatting"""
        if isinstance(text, list) and not self.accepts_list:
            if self.is_async:
                # Create loop once for all items
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Process all items in one loop
                    results = loop.run_until_complete(
                        asyncio.gather(*[self.embedding_fn(t) for t in text])
                    )
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)
            else:
                results = [self.embedding_fn(t) for t in text]
            return [self._format_output(r)[0] for r in results]
        
        input_text = text if self.accepts_text else [text]
        if self.is_async:
            # Create a new event loop for sync execution
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.embedding_fn(input_text))
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        else:
            result = self.embedding_fn(input_text)
        
        formatted = self._format_output(result)
        return formatted if self.accepts_list else formatted[:1]

    def __call__(self, text: Union[str, List[str]]) -> Union[
        List[List[float]], 
        Coroutine[Any, Any, List[List[float]]]
    ]:
        """Smart caller that properly handles all input/output combinations"""
        # Handle pending initialization
        if getattr(self, '_pending_init', False):
            async def _init_and_run():
                await self._async_run_sample_inference()
                self._pending_init = False
                return await self._async_impl(text)
            return _init_and_run()

        async def _async_wrapper():
            return await self._async_impl(text)

        if getattr(self, '_force_async', False):
            return _async_wrapper()

        try:
            loop = asyncio.get_running_loop()
            return _async_wrapper()
        except RuntimeError:
            return self._sync_impl(text)

    async def __aenter__(self):
        self._force_async = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._force_async = False

    def _inspect_input_types(self):
        """Determine input type acceptance through inspection or testing"""
        # Try type hints first
        sig = inspect.signature(self.embedding_fn)
        type_hints = get_type_hints(self.embedding_fn)
        
        input_param = next(
            (param for param in sig.parameters.values() 
             if param.name in {'text', 'input', 'query', 'x', 'data'}),
            next(iter(sig.parameters.values()))
        )
        
        if input_param.name in type_hints:
            hint = type_hints[input_param.name]
            origin = get_origin(hint)
            args = get_args(hint)
            
            self.accepts_list = origin in {list, List, Sequence, Iterable}
            self.accepts_text = str in args if args else hint == str
            return

        # If no type hints, try sample runs
        sample_text = "Sample text"
        sample_list = ["Sample 1", "Sample 2"]
        
        try:
            if self.is_async:
                asyncio.run(self.embedding_fn(sample_text))
            else:
                self.embedding_fn(sample_text)
            self.accepts_text = True
        except:
            self.accepts_text = False
            
        try:
            if self.is_async:
                asyncio.run(self.embedding_fn(sample_list))
            else:
                self.embedding_fn(sample_list)
            self.accepts_list = True
        except:
            self.accepts_list = False