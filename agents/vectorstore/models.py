"""
Base vector store implementation for embedding storage and retrieval.

This module provides an abstract base class for vector stores with thread-safe
operations and resource management.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional
import numpy as np
from pathlib import Path
import asyncio
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import os
from datetime import datetime
import contextvars

from agents.messages.message import Message


class WorkerThreadPoolExecutor(ThreadPoolExecutor):
    """
    Thread pool executor that preserves context variables across threads.
    
    This executor ensures that context variables (like request context, security context, etc.)
    are properly propagated to worker threads.
    
    :ivar _max_workers: Maximum number of worker threads
    :type _max_workers: int
    """
    
    def submit(self, fn, *args, **kwargs):
        """
        Submit a callable to be executed with the current context.
        
        Captures the current context and ensures it is restored when the 
        function executes in the worker thread.
        
        :param fn: The callable object to be executed
        :type fn: Callable
        :param args: Positional arguments for the callable
        :type args: Any
        :param kwargs: Keyword arguments for the callable
        :type kwargs: Any
        :return: A future representing pending execution
        :rtype: concurrent.futures.Future
        """
        context = contextvars.copy_context()
        return super().submit(context.run, fn, *args, **kwargs)


class BaseVectorStore(ABC):
    """
    Abstract base class for vector storage implementations.
    
    Provides thread-safe executor management and abstract methods for
    embedding storage and retrieval operations.
    
    :cvar _instance_lock: Lock for thread-safe executor management
    :type _instance_lock: Lock
    :cvar _executor: Thread pool executor for async operations
    :type _executor: ThreadPoolExecutor
    """
    _instance_lock = Lock()
    _executor = None

    data: Dict[Any, Any] = {}

    def __init__(self):
        """
        Initialize vector store with context-aware thread pool executor.
        """
        with self.__class__._instance_lock:
            if self.__class__._executor is None:
                self.__class__._executor = WorkerThreadPoolExecutor(
                    max_workers = min(32, (os.cpu_count() or 1) + 4)
                )
                
    async def filter_recent(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        filter: Union[Dict[str, Any], None] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter stored items by their addition date.
        
        :return: List of metadata entries within the date range
        :rtype: List[Dict[str, Any]]
        """
        loop = asyncio.get_running_loop()
        
        async def search_dates():
            if end_date or start_date:
                filtered = [
                    val for _, val in self.data.items() 
                    if (not start_date or datetime.fromisoformat(val.date) > start_date) and
                       (not end_date or datetime.fromisoformat(val.date) < end_date)
                ]
                return await loop.run_in_executor(None,
                    lambda: sorted(filtered, key = lambda x: datetime.fromisoformat(x.date))
                )
            else:
                return await loop.run_in_executor(None,
                    lambda: sorted(self.data.values(), key = lambda x: datetime.fromisoformat(x.date))
                )
        data = await search_dates()
        if filter:
            data = [item for item in data if all(item[attr] == filter[attr] for attr in filter)]
        
        return data

    async def search_attribute(
        self, 
        attribute: str, 
        values: Union[List[str], str]
    ) -> List[Any]:
        """
        Filter stored items by a specific attribute.
        
        :param attribute: Attribute name to filter on
        :type attribute: str
        :param values: Value(s) to match against
        :type values: Union[List[str], str]
        :return: List of matching metadata entries
        :rtype: List[Any]
        """
    
        loop = asyncio.get_running_loop()  
        async def filter_attrs():
            if isinstance(values, str):
                value_list = [values]
            else:
                value_list = values
          
            return await loop.run_in_executor(None, lambda: [
                val for _, val in self.data.items() 
                if hasattr(val, attribute) and getattr(val, attribute) in value_list
            ])
            
        return await filter_attrs()
    
    
    @classmethod
    async def acleanup_executor(cls):
        """
        Asynchronously clean up thread pool executor.
        
        Ensures graceful shutdown of executor in async contexts.
        Waits for pending tasks to complete before shutdown.
        
        :raises RuntimeError: If cleanup fails
        """
        with cls._instance_lock:
            if cls._executor is not None:
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, cls._executor.shutdown, True)
                finally:
                    cls._executor = None

    @classmethod
    def cleanup_executor(cls):
        """
        Synchronously clean up thread pool executor.
        
        Used for cleanup in __del__ and synchronous contexts.
        Attempts graceful shutdown but will force if needed.
        """
        with cls._instance_lock:
            if cls._executor is not None:
                try:
                    cls._executor.shutdown(wait = True, cancel_futures = True)
                finally:
                    cls._executor = None

    def __del__(self):
        """
        Clean up resources on deletion.
        
        Handles both async and sync contexts appropriately.
        """
        try:
            loop = asyncio.get_running_loop()
            if not loop.is_closed():
                # We're in an async context but can't await in __del__
                # Schedule cleanup for later
                loop.call_soon_threadsafe(self.__class__.cleanup_executor)
        except RuntimeError:
            # No event loop - use sync cleanup
            self.__class__.cleanup_executor()

    @abstractmethod
    async def add(
        self, 
        embeddings: Union[np.ndarray, list[float], list[Union[np.ndarray, list[float]]]], 
        metadata: Union[List[Dict[str, Any]], Dict[str, Any]]
    ) -> List[int]:
        """
        Add embeddings with metadata to the vector store.
        
        :param embeddings: Single or multiple embeddings to store
        :type embeddings: Union[np.ndarray, list[float], list[Union[np.ndarray, list[float]]]]
        :param metadata: Metadata associated with the embeddings
        :type metadata: Union[List[Dict[str, Any]], Dict[str, Any]]
        :return: List of assigned IDs for the embeddings
        :rtype: List[int]
        """
        pass
    
    @abstractmethod
    async def search_relevant(
        self, 
        query: Union[str, list[str]], 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings in the vector store.
        
        :param query: Query string or list of queries
        :type query: Union[str, list[str]]
        :param k: Number of results to return
        :type k: int
        :return: List of metadata for similar embeddings
        :rtype: List[Dict[str, Any]]
        """
        pass
    
    @abstractmethod
    async def delete(self, ids: Union[List[int], int]) -> None:
        """
        Delete embeddings from the vector store.
        
        :param ids: Single ID or list of IDs to delete
        :type ids: Union[List[int], int]
        """
        pass
    
    @abstractmethod
    async def save(self, path: Union[str, Path]) -> None:
        """
        Save the vector store to disk.
        
        :param path: Path to save the vector store
        :type path: Union[str, Path]
        """
        pass
    
    @abstractmethod
    async def load(self, path: Union[str, Path]) -> None:
        """
        Load the vector store from disk.
        
        :param path: Path to load the vector store from
        :type path: Union[str, Path]
        """
        pass

    @abstractmethod
    async def serialize(self, path: Union[str, Path]) -> None:
        """
        Serialize the vector store to disk.
        
        Should save both index and metadata in a format that can be loaded later.
        
        :param path: Directory path to save serialized data
        :type path: Union[str, Path]
        """
        pass

    @abstractmethod
    async def deserialize(self, path: Union[str, Path]) -> None:
        """
        Deserialize the vector store from disk.
        
        Should restore both index and metadata to their previous state.
        
        :param path: Directory path containing serialized data
        :type path: Union[str, Path]
        """
        pass


    async def __len__(self) -> int:
        """
        Return the number of elements in the vector store.
        
        :return: Number of elements
        :rtype: int
        """
        return len(self.data)