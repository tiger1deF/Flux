"""
Tool models for agent function wrapping and parameter handling.

This module provides models for wrapping functions as tools that can be used by agents,
with support for both synchronous and asynchronous execution.
"""

from typing import List, Any, Optional, Type, Callable, Dict
from pydantic import BaseModel, Field
from functools import partial
import inspect
import asyncio


class ToolParameter(BaseModel):
    """
    Model representing a parameter for a tool function.
    
    :ivar name: Name of the parameter
    :type name: str
    :ivar type: Type of the parameter
    :type type: Type
    :ivar required: Whether the parameter is required
    :type required: bool
    :ivar default: Default value for the parameter
    :type default: Any
    :ivar description: Description of the parameter
    :type description: Optional[str]
    """
    name: str
    type: Type
    required: bool = True
    default: Any = None
    description: Optional[str] = None

    def __repr__(self):
        """
        Get string representation of parameter.
        
        :return: String representation
        :rtype: str
        """
        return f"ToolParameter(name={self.name}, type={self.type}, required={self.required}, default={self.default}, description={self.description})"
    

class Tool(BaseModel):
    """
    Model for wrapping functions as agent tools, allowing them to be called like functions while
    maintaining static parameters, validation, and metadata.
    
    Handles both synchronous and asynchronous function execution with parameter validation.
    
    :ivar name: Name of the tool
    :type name: str
    :ivar description: Description of what the tool does
    :type description: Optional[str]
    :ivar parameters: List of tool parameters
    :type parameters: List[ToolParameter]
    :ivar static_params: Static parameters to always pass to function
    :type static_params: Dict[str, Any]
    :ivar function: The wrapped function
    :type function: Callable
    :ivar return_type: Expected return type
    :type return_type: Optional[Type]
    :ivar _is_async: Whether the function is async
    :type _is_async: bool
    """
    name: str
    description: Optional[str] = None
    parameters: List[ToolParameter] = Field(default_factory=list)
    static_params: Dict[str, Any] = Field(default_factory=dict)
    
    function: Callable
    return_type: Optional[Type] = None
    
    _is_async: bool = False
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        """
        Initialize tool and detect if function is async.
        
        :param data: Tool configuration data
        """
        super().__init__(**data)
        self._is_async = inspect.iscoroutinefunction(self.function)

    def __repr__(self):
        """
        Get string representation of tool.
        
        :return: String representation
        :rtype: str
        """
        return f"Tool(name={self.name}, description={self.description}, parameters={self.parameters}, static_params={self.static_params}, function={self.function}, return_type={self.return_type})"

    async def __call__(self, *args, **kwargs) -> Any:
        """
        Async call implementation for the tool.
        
        Handles both async and sync functions by running sync functions in executor.
        
        :param args: Positional arguments for the function
        :param kwargs: Keyword arguments for the function
        :return: Function result
        :rtype: Any
        """
        all_kwargs = {**self.static_params, **kwargs}
        func = self.function
        
        if inspect.iscoroutinefunction(func):
            return await func(*args, **all_kwargs)
        else:
            # Run sync functions in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                partial(func, *args, **all_kwargs)
            )

    def __sync_call__(self, *args, **kwargs) -> Any:
        """
        Synchronous call implementation for the tool.
        
        Handles both async and sync functions by running async functions in new event loop.
        
        :param args: Positional arguments for the function
        :param kwargs: Keyword arguments for the function
        :return: Function result
        :rtype: Any
        """
        all_kwargs = {**self.static_params, **kwargs}
        func = self.function
        
        if inspect.iscoroutinefunction(func):
            # For async functions, run them in a new event loop
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(func(*args, **all_kwargs))
            finally:
                loop.close()
        else:
            return func(*args, **all_kwargs)

    def __get__(self, obj, objtype = None):
        """
        Descriptor protocol implementation.
        
        Returns appropriate call method based on whether function is async.
        
        :param obj: Instance that the descriptor is accessed from
        :param objtype: Type of the instance
        :return: Call method
        :rtype: Callable
        """
        if obj is None:
            return self
        return self.__call__ if self._is_async else self.__sync_call__
