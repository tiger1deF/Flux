from typing import Any, Callable, Dict, Type, get_type_hints, Union, Coroutine
import inspect
from functools import lru_cache
import weakref
import asyncio
import threading
import types

from dataclasses import dataclass


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
    
    
class LLM:
    """
    Flux wrapper class for LLM completion functions.

    This class provides a standardized interface for both synchronous and asynchronous
    completion functions, handling parameter validation, type checking, thread safe,
    and supports token limiting.

    :param completion_fn: The completion function to wrap
    :type completion_fn: Callable
    :param default_params: Default parameters to pass to the completion function
    :type default_params: Dict[str, Any]

    Note - The following parameters are stored defaults, and won't be used unless the input function
    has the same parameter name.
    
    :ivar top_p: Top-p sampling parameter
    :type top_p: float
    :ivar temperature: Temperature parameter for controlling randomness
    :type temperature: float
    :ivar max_tokens: Maximum number of tokens in the response
    :type max_tokens: int
    :ivar input_tokens: Maximum number of tokens in the input
    :type input_tokens: int
    :ivar system_prompt: System prompt to prepend to queries
    :type system_prompt: str
    """

    _function_cache: Dict[Callable, Dict[str, Any]] = weakref.WeakKeyDictionary()
    _thread_local = threading.local()
    _lock = threading.Lock()
    _call_impl: Callable = None
    
    # Standard parameters
    top_p: float = 0.9
    temperature: float = 0.1
    max_tokens: int = 2_000
    system_prompt: str = None
    context_length: int = 10_000

    # Parameter mapping for normalization
    CONTEXT_LENGTH_PARAMS = {
        "context_length",
        "input_tokens",
        "context_window",
        "max_input_tokens",
        "max_context_length",
        "context_window_size",
    }

    MAX_TOKENS_PARAMS = {
        "max_tokens",
        "response_tokens",
        "output_tokens",
        "max_output_tokens",
        "max_completion_tokens",
        "completion_tokens",
    }
    

    def __init__(self, completion_fn: Callable, **default_params):
        """
        Initialize the LLM wrapper.

        :param completion_fn: The completion function to wrap
        :type completion_fn: Callable
        :param default_params: Default parameters to pass to the completion function. Can be anything.
        :type default_params: Dict[str, Any]
        """
        self.completion_fn = completion_fn  
        self.is_async = inspect.iscoroutinefunction(completion_fn)
        
        # Normalize parameters first
        self.default_params = self._normalize_parameters(default_params)
        
        cached_inspection = self._function_cache.get(completion_fn)
        if cached_inspection:
            self.parameters = cached_inspection['parameters']
            self.input_param_name = cached_inspection['input_param_name']
            self._async_call_impl = cached_inspection['async_impl']
            self._sync_call_impl = cached_inspection['sync_impl']
        else:
            # Inspect function and build implementations
            self._inspect_function()
            self._build_call_implementation()
            
            # Cache the results
            self._function_cache[completion_fn] = {
                'parameters': self.parameters,
                'input_param_name': self.input_param_name,
                'async_impl': self._async_call_impl,
                'sync_impl': self._sync_call_impl
            }
       
        self._validate_types(**self.default_params)

    def _normalize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parameter names while preserving originals."""
        normalized = params.copy()
        
        # Handle context length parameters
        for param in params:
            if param in self.CONTEXT_LENGTH_PARAMS and param != "context_length":
                normalized["context_length"] = params[param]
                
        # Handle max tokens parameters
        for param in params:
            if param in self.MAX_TOKENS_PARAMS and param != "max_tokens":
                normalized["max_tokens"] = params[param]
        
        return normalized

    def _build_call_implementation(self):
        """Build optimized call implementations for both sync and async."""
        valid_params = {param.name for param in self.parameters}
        
        # Pre-compute filtered default args
        self._filtered_defaults = {
            k: v for k, v in self.default_params.items() 
            if k in valid_params
        }
        
        # Pre-compute token limit check
        self._has_token_limit = any(
            param in self.CONTEXT_LENGTH_PARAMS.union(self.MAX_TOKENS_PARAMS)
            for param in valid_params
        )
        
        # Create instance methods
        self._async_call_impl = types.MethodType(self._create_async_impl(valid_params), self)
        self._sync_call_impl = types.MethodType(self._create_sync_impl(valid_params), self)


    def _create_async_impl(self, valid_params):
        """Create the async implementation function"""
        async def _async_call_impl(self, query: str, **kwargs):
            # Start with pre-filtered defaults
            call_args = self._filtered_defaults.copy()
            
            # Update with filtered kwargs
            call_args.update({
                k: v for k, v in kwargs.items() 
                if k in valid_params
            })
            
            # Normalize any new parameters from kwargs
            normalized_kwargs = self._normalize_parameters(kwargs)
            call_args.update({
                k: v for k, v in normalized_kwargs.items()
                if k in valid_params and k not in kwargs
            })
            
            # Prepare input text
            input_text = f"{self.system_prompt}\n\n{query}" if self.system_prompt else query
            call_args[self.input_param_name] = input_text
            
            return await self.completion_fn(**call_args)
        return _async_call_impl


    def _create_sync_impl(self, valid_params):
        """Create the sync implementation function"""
        def _sync_call_impl(self, query: str, **kwargs):
            # Start with pre-filtered defaults
            call_args = self._filtered_defaults.copy()
            
            # Update with filtered kwargs
            call_args.update({
                k: v for k, v in kwargs.items() 
                if k in valid_params
            })
            
            # Normalize any new parameters from kwargs
            normalized_kwargs = self._normalize_parameters(kwargs)
            call_args.update({
                k: v for k, v in normalized_kwargs.items()
                if k in valid_params and k not in kwargs
            })
            
            # Prepare input text
            input_text = f"{self.system_prompt}\n\n{query}" if self.system_prompt else query
            call_args[self.input_param_name] = input_text
            
            with self._lock:  # Thread safety for sync calls
                result = self.completion_fn(**call_args)
                # Handle case where completion_fn returns a coroutine
                if asyncio.iscoroutine(result):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(result)
                    finally:
                        loop.close()
                return result
        return _sync_call_impl


    def __call__(self, query: str, **kwargs) -> Union[str, Coroutine[Any, Any, str]]:
        """
        Call implementation that works in both sync and async contexts.
        
        Automatically handles context switching between sync/async:
        - If called in async context: returns a coroutine
        - If called in sync context: returns the result directly
        
        The completion function will be coerced into the appropriate context:
        - Async completion_fn in sync context: runs in event loop
        - Sync completion_fn in async context: runs in thread pool
        
        :param query: The query text to process
        :type query: str
        :param kwargs: Additional parameters to pass to the completion function
        :type kwargs: Any
        :return: Completion result or coroutine that will return completion
        :rtype: Union[str, Coroutine[Any, Any, str]]
        """
        try:
            # We're in an async context
            loop = asyncio.get_running_loop()
            if asyncio.iscoroutinefunction(self.completion_fn):
                # Both async - direct return
                return self._async_call_impl(query, **kwargs)
            else:
                # Sync fn in async context - wrap in thread pool
                return loop.run_in_executor(
                    self._thread_pool,
                    self._sync_call_impl,
                    query,
                    **kwargs
                )
        except RuntimeError:
            # We're in a sync context
            if asyncio.iscoroutinefunction(self.completion_fn):
                # Async fn in sync context - run in new loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self._async_call_impl(query, **kwargs))
                finally:
                    loop.close()
            # Both sync - direct return
            return self._sync_call_impl(query, **kwargs)


    @staticmethod
    @lru_cache(maxsize = 128)
    def _get_type_hints(func: Callable) -> Dict[str, Type]:
        """
        Get cached type hints for a function.

        :param func: The function to get type hints for
        :type func: Callable
        :return: Dictionary of parameter names to their types
        :rtype: Dict[str, Type]
        """
        return get_type_hints(func)


    def _inspect_function(self) -> None:
        """
        Inspect the completion function to determine its parameters and input parameter.
        
        This method analyzes the function signature and type hints to set up proper
        parameter handling and identify the main input parameter.

        :raises ValueError: If no suitable input parameter is found
        """
        signature = inspect.signature(self.completion_fn)
        type_hints = self._get_type_hints(self.completion_fn)
        
        self.parameters = []
        self.input_param_name = None
        
        input_param_priorities = {
            "text", "prompt", "query", "input", "message", "content"
        }

        for param_name, param in signature.parameters.items():
            param_type = type_hints.get(param_name, Any)
            required = param.default == param.empty
            default = None if param.default == param.empty else param.default

            if param_name in self.default_params:
                required = False
                default = self.default_params[param_name]

            self.parameters.append(LLMParameter(
                name = param_name,
                type = param_type,
                required = required,
                default = default
            ))

            if self.input_param_name is None and param_type == str:
                if param_name.lower() in input_param_priorities:
                    self.input_param_name = param_name
                elif self.input_param_name is None:
                    self.input_param_name = param_name

        if self.input_param_name is None:
            raise ValueError("No suitable input parameter found in completion function")


    def _validate_types(self, **kwargs) -> None:
        """
        Validate and convert parameter types according to the function signature.

        :param kwargs: Parameters to validate
        :type kwargs: Dict[str, Any]
        :raises TypeError: If parameter types don't match and can't be converted
        :raises ValueError: If a required parameter is missing
        """
        for param in self.parameters:
            if param.name == self.input_param_name:
                continue
            
            if param.name in kwargs:
                value = kwargs[param.name]
                if not isinstance(value, param.type):
                    try:
                        kwargs[param.name] = param.type(value)
                    except (ValueError, TypeError):
                        raise TypeError(
                            f"Parameter '{param.name}' expects type {param.type.__name__}, "
                            f"got {type(value).__name__}"
                        )
            elif param.required and param.name not in self.default_params:
                if hasattr(self, param.name):
                    self.default_params[param.name] = getattr(self, param.name)
                else:
                    raise ValueError(f"Required parameter '{param.name}' not provided")


    @property
    def signature(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the signature information for the wrapped completion function.

        :return: Dictionary containing parameter information including types, requirements,
                default values, and whether it's the input parameter
        :rtype: Dict[str, Dict[str, Any]]
        """
        return {
            param.name: {
                "type": param.type.__name__,
                "required": param.required,
                "default": param.default if param.name not in self.default_params 
                    else self.default_params[param.name],
                "is_input": param.name == self.input_param_name
            }
            for param in self.parameters
        }
