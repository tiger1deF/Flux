from typing import Any, Callable, Dict, Type, get_type_hints, Union, Coroutine
import inspect
from functools import lru_cache
import weakref
import asyncio
import threading
import contextlib
import types

from llm.models import LLMParameter
from utils.summarization import truncate_text


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
    :ivar system_prompt: System prompt to prepend to queries
    :type system_prompt: str
    """

    _function_cache: Dict[Callable, Dict[str, Any]] = weakref.WeakKeyDictionary()
    _thread_local = threading.local()
    _lock = threading.Lock()
    
    top_p: float = 0.9
    temperature: float = 0.1
    max_tokens: int = 1_000
    system_prompt: str = None
    

    def __init__(self, completion_fn: Callable, **default_params):
        """
        Initialize the LLM wrapper.

        :param completion_fn: The completion function to wrap
        :type completion_fn: Callable
        :param default_params: Default parameters to pass to the completion function. Can be anything.
        :type default_params: Dict[str, Any]
        """
        self.completion_fn = completion_fn  
        self.default_params = default_params
        self.is_async = inspect.iscoroutinefunction(completion_fn)
        
        cached_inspection = self._function_cache.get(completion_fn)
        if cached_inspection:
            self.parameters = cached_inspection['parameters']
            self.input_param_name = cached_inspection['input_param_name']
            self._call_impl = cached_inspection['call_impl']
            
        else:
            self._inspect_function()
            self._build_call_implementation()
            self._function_cache[completion_fn] = {
                'parameters': self.parameters,
                'input_param_name': self.input_param_name,
                'call_impl': self._call_impl
            }
       
        self._validate_types(**default_params)


    def _build_call_implementation(self):
        """
        Builds the appropriate call implementation based on the completion function's signature.
        
        This method creates either a synchronous or asynchronous implementation depending on
        the completion function type and whether token limiting is required.
        """
        token_limit_params = ["input_tokens", "max_input_tokens"]
        has_token_limit = any(
            param.name in token_limit_params 
            for param in self.parameters
        ) or any(
            param in token_limit_params 
            for param in self.default_params
        )
        
        token_param = next(
            (param for param in token_limit_params if param in self.default_params),
            next(
                (param.name for param in self.parameters if param.name in token_limit_params),
                None
            )
        )

        # Get valid parameter names from the completion function
        valid_params = {param.name for param in self.parameters}
        
        if has_token_limit:
            if self.is_async:
                async def _call_impl(self, query: str, **kwargs):
                    # Filter kwargs to only include valid parameters
                    call_args = {
                        k: v for k, v in self.default_params.items() 
                        if k in valid_params
                    }
                    call_args.update({
                        k: v for k, v in kwargs.items() 
                        if k in valid_params
                    })
                    
                    input_text = f"{self.system_prompt}\n\n{query}" if self.system_prompt else query
                    max_tokens = call_args.get(token_param)
                    input_text = truncate_text(input_text, max_tokens) if max_tokens else input_text
                    call_args[self.input_param_name] = input_text
                    
                    async with self._async_lock():
                        return await self.completion_fn(**call_args)
            else:
                def _call_impl(self, query: str, **kwargs):
                    # Filter kwargs to only include valid parameters
                    call_args = {
                        k: v for k, v in self.default_params.items() 
                        if k in valid_params
                    }
                    call_args.update({
                        k: v for k, v in kwargs.items() 
                        if k in valid_params
                    })
                    
                    input_text = f"{self.system_prompt}\n\n{query}" if self.system_prompt else query
                    max_tokens = call_args.get(token_param)
                    input_text = truncate_text(input_text, max_tokens) if max_tokens else input_text
                    call_args[self.input_param_name] = input_text
                    
                    with self._lock:
                        return self.completion_fn(**call_args)
        else:
            if self.is_async:
                async def _call_impl(self, query: str, **kwargs):
                    # Filter kwargs to only include valid parameters
                    call_args = {
                        k: v for k, v in self.default_params.items() 
                        if k in valid_params
                    }
                    call_args.update({
                        k: v for k, v in kwargs.items() 
                        if k in valid_params
                    })
                    
                    input_text = f"{self.system_prompt}\n\n{query}" if self.system_prompt else query
                    call_args[self.input_param_name] = input_text
                    
                    async with self._async_lock():
                        return await self.completion_fn(**call_args)
            else:
                def _call_impl(self, query: str, **kwargs):
                    # Filter kwargs to only include valid parameters
                    call_args = {
                        k: v for k, v in self.default_params.items() 
                        if k in valid_params
                    }
                    call_args.update({
                        k: v for k, v in kwargs.items() 
                        if k in valid_params
                    })
                    
                    input_text = f"{self.system_prompt}\n\n{query}" if self.system_prompt else query
                    call_args[self.input_param_name] = input_text
                    
                    with self._lock:
                        return self.completion_fn(**call_args)

        self._call_impl = types.MethodType(_call_impl, self)


    def __call__(self, query: str, **kwargs) -> Union[str, Coroutine[Any, Any, str]]:
        """
        Call the wrapped completion function with the given query and parameters.

        :param query: The input text to send to the language model
        :type query: str
        :param kwargs: Additional parameters to pass to the completion function
        :type kwargs: Dict[str, Any]
        :return: The completion result (or coroutine for async functions)
        :rtype: Union[str, Coroutine[Any, Any, str]]
        """
        if self.is_async:
            return self._call_impl(query, **kwargs)
        return self._call_impl(query, **kwargs)


    @contextlib.asynccontextmanager
    async def _async_lock(self):
        """
        Async context manager for thread safety in async operations.

        :yield: The acquired lock
        """
        loop = asyncio.get_event_loop()
        with self._lock:
            yield


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
        
        input_param_priorities = [
            "text", "prompt", "query", "input", "message", "content"
        ]

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
