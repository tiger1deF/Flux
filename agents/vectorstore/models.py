"""
Base vector store implementation for embedding storage and retrieval.

This module provides an abstract base class for vector stores with thread-safe
operations and resource management.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import numpy as np
from pathlib import Path
import asyncio
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import os


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

    def __init__(self):
        """
        Initialize vector store with thread pool executor.
        """
        with self.__class__._instance_lock:
            if self.__class__._executor is None:
                self.__class__._executor = ThreadPoolExecutor(
                    max_workers=min(32, (os.cpu_count() or 1) + 4)
                )

    @classmethod
    async def acleanup_executor(cls):
        """
        Asynchronously clean up thread pool executor.
        
        Used for cleanup in async contexts.
        """
        with cls._instance_lock:
            if cls._executor is not None:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, cls._executor.shutdown, False)
                cls._executor = None

    @classmethod
    def cleanup_executor(cls):
        """
        Synchronously clean up thread pool executor.
        
        Used for cleanup in __del__.
        """
        with cls._instance_lock:
            if cls._executor is not None:
                cls._executor.shutdown(wait=False)
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
    async def search(
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

    