"""
HNSW vector store implementation for efficient local embeddings.

This module provides a concrete implementation of BaseVectorStore using HNSW
(Hierarchical Navigable Small World) graphs for fast approximate nearest neighbor search.
"""

from typing import List, Dict, Any, Optional, Set, Union, Callable
import numpy as np
import hnswlib
import os
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextvars
from threading import Lock
import orjson

from agents.vectorstore.models import BaseVectorStore
from llm import BaseEmbeddingFunction, base_gemini_embedder


class ContextAwareThreadPoolExecutor(ThreadPoolExecutor):
    """
    Thread pool executor that preserves context variables across threads.
    """
    def submit(self, fn, *args, **kwargs):
        """
        Submit a callable to be executed with the current context.
        
        :param fn: The callable object to be executed
        :param args: Positional arguments for the callable
        :param kwargs: Keyword arguments for the callable
        :return: A future representing pending execution
        """
        context = contextvars.copy_context()
        return super().submit(context.run, fn, *args, **kwargs)


class HNSWStore(BaseVectorStore):
    """
    Vector store implementation using HNSW algorithm for efficient similarity search.
    
    :ivar dim: Dimensionality of the embedding space
    :type dim: int
    :ivar max_elements: Maximum number of elements in the index
    :type max_elements: int
    :ivar deleted_count: Number of deleted elements
    :type deleted_count: int
    :ivar internal_embed: Function for generating embeddings
    :type internal_embed: BaseEmbeddingFunction
    :ivar index: HNSW index for similarity search
    :type index: hnswlib.Index
    :ivar metadata: Storage for element metadata
    :type metadata: Dict[int, Dict[str, Any]]
    :ivar deleted_ids: Set of deleted element IDs
    :type deleted_ids: Set[int]
    """
    
    _instance_lock = Lock()
    _executor = None

    def __init__(
        self,
        embedding_function: Optional[BaseEmbeddingFunction] = base_gemini_embedder,
        max_elements: int = 100_000,
        ef_construction: int = 1000,
        M: int = 128,
        allow_replace_deleted: bool = False,
    ):
        """
        Initialize HNSW vector store.
        
        :param embedding_function: Function to generate embeddings
        :type embedding_function: Optional[BaseEmbeddingFunction]
        :param max_elements: Maximum number of elements in index
        :type max_elements: int
        :param ef_construction: Size of dynamic candidate list for construction
        :type ef_construction: int
        :param M: Number of bi-directional links created for each element
        :type M: int
        :param allow_replace_deleted: Whether to allow replacing deleted elements
        :type allow_replace_deleted: bool
        """
        super().__init__()
        
        self.dim = embedding_function.dimension
        self.max_elements = max_elements
        self.deleted_count = 0
        
        self.internal_embed = embedding_function
        
        # Initialize HNSW index
        self.index = hnswlib.Index(
            space = 'cosine', 
            dim = self.dim
        )
        self.index.init_index(
            max_elements = max_elements,
            ef_construction = ef_construction,
            M = M,
            allow_replace_deleted = allow_replace_deleted
        )
        
        # Metadata storage
        self.metadata: Dict[int, Dict[str, Any]] = {}
        self.deleted_ids: Set[int] = set()
        
        # Thread-safe singleton executor initialization
        with HNSWStore._instance_lock:
            if HNSWStore._executor is None:
                HNSWStore._executor = ContextAwareThreadPoolExecutor(
                    max_workers=min(32, (os.cpu_count() or 1) + 4)
                )
    
    
    async def add(
        self, 
        text: Optional[Union[str, List[str]]] = None,
        embedding_function: Optional[BaseEmbeddingFunction] = None,
        embeddings: Optional[Union[np.ndarray, list[float], list[Union[np.ndarray, list[float]]]]] = None, 
        metadata: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = [],
    ) -> List[int]:
        """
        Add items to the vector store.
        
        :param text: Text to embed and store
        :type text: Optional[Union[str, List[str]]]
        :param embedding_function: Custom embedding function
        :type embedding_function: Optional[BaseEmbeddingFunction]
        :param embeddings: Pre-computed embeddings to store
        :type embeddings: Optional[Union[np.ndarray, list[float], list[Union[np.ndarray, list[float]]]]]
        :param metadata: Metadata for the items
        :type metadata: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]
        :return: List of assigned IDs
        :rtype: List[int]
        """
    
        # Handles embedding if needed
        if embedding_function is None:
            embedding_function = self.internal_embed
        
        if text is not None:
            if isinstance(text, str):
                text = [text]
            embeddings = await embedding_function(text)
        
        if isinstance(metadata, dict):
            metadata = [metadata]
                        
        if text is not None:
            if not metadata:
                metadata = [{"query": query} for query in text]
            
            else:
                metadata = [
                    {**meta, "query": txt}
                    for meta, txt in zip(metadata, text)
                ]
        
        if len(embeddings) != len(metadata):
            raise ValueError(f"Number of embeddings must match number of metadata items. Len Text: {len(text)}, Len Metadata: {len(metadata)}, Len Embeddings: {len(embeddings)}")
        
        start_id = self.index.get_current_count()
        ids = list(range(start_id, start_id + len(embeddings)))
        
        # Add to HNSW index - this needs to be sequential
        self.index.add_items(embeddings, ids)
        
        async def store_metadata(id_, meta):
            return id_, {
                **meta,
                "_added": datetime.now().isoformat(),
                "_id": id_
            }
        
        tasks = [
            store_metadata(id_, meta) 
            for id_, meta in zip(ids, metadata)
        ]
        results = await asyncio.gather(*tasks)
        
        for id_, meta in results:
            self.metadata[id_] = meta
        
        return ids
    
    
    async def search(
        self, 
        query: Union[str, List[str]], 
        embedding_function: Optional[BaseEmbeddingFunction] = None,
        k: int = 5,
        ef_search: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar items in the vector store.
        
        :param query: Query text or list of queries
        :type query: Union[str, List[str]]
        :param embedding_function: Custom embedding function
        :type embedding_function: Optional[BaseEmbeddingFunction]
        :param k: Number of nearest neighbors to return
        :type k: int
        :param ef_search: Size of dynamic candidate list for search
        :type ef_search: Optional[int]
        :return: List of metadata for similar items with distances
        :rtype: List[Dict[str, Any]]
        """
        # Get current count of items (excluding deleted)
        current_count = self.index.get_current_count() - self.deleted_count
        
        # Dynamic ef scaling based on dataset size
        if ef_search is None:
            if current_count < 20:
                ef_search = max(k * 100, current_count * 10)
            elif current_count < 100:
                ef_search = max(k * 50, 100)
            else:
                ef_search = max(k * 40, 100)
        
        self.index.set_ef(ef_search)
        
        if isinstance(query, str):
            query = [query]
            
        if embedding_function is None:
            embedding_function = self.internal_embed
            
        embeddings = await embedding_function(query)
        
        # Adjust k if we have fewer items than requested
        actual_k = min(k, current_count)
        
        if actual_k == 0:
            return []
            
        labels, distances = self.index.knn_query(embeddings, k=actual_k)
        
        results = []
        for idx, dist in zip(labels[0], distances[0]):
            if idx not in self.deleted_ids:
                results.append({
                    **self.metadata[idx],
                    "_distance": float(dist)
                })
                
        return results
    
    
    async def delete(self, ids: List[int]) -> None:
        """
        Delete items from the vector store.
        
        :param ids: List of IDs to delete
        :type ids: List[int]
        """
        for id_ in ids:
            if isinstance(id_, list):
                id_ = id_[0]
                
            if id_ in self.metadata:
                self.index.mark_deleted(id_)
                self.deleted_ids.add(id_)
                self.deleted_count += 1
    
    
    async def save(self, path: str) -> None:
        """
        Save the vector store to disk.
        
        Saves both the HNSW index and metadata concurrently.
        
        :param path: Directory path to save to
        :type path: str
        """
        os.makedirs(path, exist_ok = True)
        
        # Save index and metadata concurrently
        async def save_index():
            """Helper function to save HNSW index"""
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                self.index.save_index,
                os.path.join(path, "index.bin")
            )
            
        async def save_metadata():
            """Helper function to save metadata"""
            metadata_dict = {
                "metadata": self.metadata,
                "deleted_ids": list(self.deleted_ids),
                "current_count": self.index.get_current_count(),
                "deleted_count": self.deleted_count,
                "dim": self.dim,
                "max_elements": self.max_elements
            }
            
            async def write_metadata():
                with open(os.path.join(path, "metadata.json"), "wb") as f:
                    f.write(orjson.dumps(metadata_dict))
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, write_metadata)
        
        await asyncio.gather(save_index(), save_metadata())
    

    async def load(self, path: str) -> None:
        """
        Load the vector store from disk.
        
        Loads both the HNSW index and metadata.
        
        :param path: Directory path to load from
        :type path: str
        """
        with open(os.path.join(path, "metadata.json"), "rb") as f:
            data = orjson.loads(f.read())
            self.metadata = data["metadata"]
            self.deleted_ids = set(data["deleted_ids"])
            self.deleted_count = data["deleted_count"]
            self.dim = data["dim"]
            self.max_elements = data["max_elements"]
        
        # Reinitialize and load index
        self.index = hnswlib.Index(space = 'cosine', dim = self.dim)
        self.index.load_index(
            os.path.join(path, "index.bin"),
            max_elements = self.max_elements
        )
        
    async def __aenter__(self):
        """
        Async context manager entry.
        
        :return: Self
        :rtype: HNSWStore
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit.
        
        Cleans up executor on exit.
        
        :param exc_type: Exception type if an error occurred
        :param exc_val: Exception value if an error occurred
        :param exc_tb: Exception traceback if an error occurred
        """
        await self.__class__.acleanup_executor()

    def __enter__(self):
        """
        Sync context manager entry.
        
        :return: Self
        :rtype: HNSWStore
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Sync context manager exit.
        
        Cleans up executor on exit.
        
        :param exc_type: Exception type if an error occurred
        :param exc_val: Exception value if an error occurred
        :param exc_tb: Exception traceback if an error occurred
        """
        self.__class__.cleanup_executor()
    