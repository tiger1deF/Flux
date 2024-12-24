"""
HNSW vector store implementation for efficient local embeddings.

This module provides a concrete implementation of BaseVectorStore using HNSW
(Hierarchical Navigable Small World) graphs for fast approximate nearest neighbor search.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
import hnswlib
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextvars
from threading import Lock
import orjson

from agents.vectorstore.models import BaseVectorStore

from llm import BaseEmbeddingFunction, base_local_embedder

from agents.messages.message import Message
from agents.messages.file import File
from agents.messages.metadata import Metadata


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
    :ivar internal_embed: Function for generating embeddings
    :type internal_embed: BaseEmbeddingFunction
    :ivar index: HNSW index for similarity search
    :type index: hnswlib.Index
    :ivar metadata: Storage for element metadata
    :type metadata: Dict[int, Dict[str, Any]]
    """
    
    _instance_lock = Lock()
    _executor = None

    def __init__(
        self,
        embedding_function: Optional[BaseEmbeddingFunction] = base_local_embedder,
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
        self.data: Dict[int, Dict[str, Any]] = {}
        
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
        metadata: Optional[Union[List[Any], Any]] = [],
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
    
        if not embeddings and text:
            if isinstance(text, str):
                text = [text]
                
            if embedding_function is None:
                embedding_function = self.internal_embed
        
            embeddings = await embedding_function(text)
    
            if metadata is None:
                if isinstance(text, str):
                    metadata = Message(content = text)
                elif isinstance(text, list):
                    metadata = [Message(content = text) for text in text]
               
        # Process metadata
        if not isinstance(metadata, list):
            id = metadata.id
            metadata = [metadata]
            ids = [id]  
        else:
            ids = [meta.id for meta in metadata]
         
        if len(embeddings) != len(metadata):
            raise ValueError(f"Number of embeddings ({len(embeddings)}) must match metadata ({len(metadata)})")
        
        # Add embeddings to HNSW index - must be sequential
        self.index.add_items(embeddings, ids)
        
        # Store processed metadata
        for id_, meta in zip(ids, metadata):
            self.data[id_] = meta
        
        if len(ids) == 1:
            return ids[0]
        else:
            return ids
    
    
    async def search_relevant(
        self,
        query: Union[str, List[str]],
        embedding_function: Optional[BaseEmbeddingFunction] = None,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        ef_search: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar items in the vector store.
        
        Uses dynamic ef scaling for optimal performance based on dataset size.
        Processes queries concurrently when possible.
        
        :param query: Query text or list of queries
        :type query: Union[str, List[str]]
        :param embedding_function: Custom embedding function to override default
        :type embedding_function: Optional[BaseEmbeddingFunction]
        :param k: Number of nearest neighbors to return
        :type k: int
        :param ef_search: Size of dynamic candidate list for search. If None, scales automatically
        :type ef_search: Optional[int]
        :return: List of metadata for similar items with distances
        :rtype: List[Dict[str, Any]]
        """
        # Dynamic ef scaling based on dataset size
        current_count = self.index.get_current_count()
        if current_count == 0:
            return []
        
        if ef_search is None:
            if current_count < 20:
                ef_search = max(k * 100, current_count * 10)
            elif current_count < 100:
                ef_search = max(k * 50, 100)
            else:
                ef_search = max(k * 40, 100)
        
        self.index.set_ef(ef_search)
        
        # Handle query embedding
        if isinstance(query, str):
            query = [query]
        
        if embedding_function is None:
            embedding_function = self.internal_embed
        
        embeddings = await embedding_function(query)
        
        # Adjust k if needed
        results = []
        if filter:
            # Overfetch to ensure we get enough results
            actual_k = min(k * 100, current_count)

            # Perform search
            labels, distances = self.index.knn_query(embeddings, k=actual_k)

            for idx, dist in zip(labels[0], distances[0]):
                meta = self.data[idx]
                if all(meta[attr] == filter[attr] for attr in filter.keys()):
                    meta.score = 1 - dist
                    results.append(meta)
                else:
                    continue
        else:
            actual_k = min(k, current_count)
   
            # Extracts results
            for idx, dist in zip(labels[0], distances[0]):
                meta = self.data[idx]
                meta.score = 1 - dist
                results.append(meta)
                
        return results
    
    
    async def delete(
        self, 
        ids: Union[int, List[int]]
    ) -> None:
        """
        Permanently delete items by rebuilding the index without them.
        
        :param ids: Single ID or list of IDs to delete
        :type ids: Union[int, List[int]]
        """
        # Convert single ID to list
        if isinstance(ids, int):
            ids = [ids]
        
        # Convert to set for O(1) lookup
        delete_set = set(ids)
        
        # Get all existing embeddings and metadata
        remaining_embeddings = []
        remaining_metadata = []
        
        # Collect remaining items
        current_count = self.index.get_current_count()
        for old_id in range(current_count):
            if old_id not in delete_set:
                # Get embedding for this ID
                embedding = self.index.get_items([old_id])[0]
                remaining_embeddings.append(embedding)
                remaining_metadata.append(self.data[old_id])
        
        # Create new index with same parameters
        new_index = hnswlib.Index(space='cosine', dim=self.dim)
        new_index.init_index(
            max_elements = self.max_elements,
            ef_construction = self.index.ef_construction,
            M = self.index.M
        )
        
        # Add remaining items to new index
        if remaining_embeddings:
            new_index.add_items(
                data = np.array(remaining_embeddings),
                ids = range(len(remaining_embeddings))
            )
        
        # Update instance variables
        self.index = new_index
        self.data = {
            new_id: meta 
            for new_id, meta in enumerate(remaining_metadata)
        }

    async def reset(self) -> None:
        """
        Reset the vector store to an empty state.
        
        Reinitializes the index with the same parameters but no data.
        """
        # Create fresh empty index with same parameters
        new_index = hnswlib.Index(space='cosine', dim=self.dim)
        new_index.init_index(
            max_elements = self.max_elements,
            ef_construction = self.index.ef_construction,
            M = self.index.M
        )
        
        # Reset instance variables
        self.index = new_index
        self.data = {}
    
    
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
            """
            Helper function to save HNSW index.
            
            :ivar path: Path to save the index file
            :type path: str
            """
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                self.index.save_index,
                os.path.join(path, "index.bin")
            )
            
        async def save_metadata():
            """Save metadata with proper serialization"""
            # Serialize each metadata item
            serialized_data = {}
            for id_, item in self.data.items():
                if hasattr(item, 'serialize'):
                    serialized_data[id_] = await item.serialize()
                else:
                    serialized_data[id_] = item
                    
            # Save store configuration
            metadata_dict = {
                "metadata": serialized_data,
                "dim": self.dim,
                "max_elements": self.max_elements,
                "ef_construction": self.index.ef_construction,
                "M": self.index.M
            }
            
            # Write metadata using executor to avoid blocking
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._executor,
                lambda: self._write_metadata(path / "metadata.json", metadata_dict)
            )

        def _write_metadata(self, filepath, metadata_dict):
            """Helper method to write metadata synchronously"""
            with open(filepath, "wb") as f:
                f.write(orjson.dumps(metadata_dict))
        
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
            self.data = data["metadata"]
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
    
    def write_metadata(self, filepath: Path, metadata_dict: dict) -> None:
        """Write metadata to file synchronously"""
        with open(filepath, "wb") as f:
            f.write(orjson.dumps(metadata_dict))

    async def serialize(self, path: Path) -> None:
        """
        Serialize the vector store to disk.
        
        :param path: Directory path to save serialized data
        :type path: Path
        """
        path.mkdir(parents=True, exist_ok=True)
        
        # Save index and metadata concurrently
        async def save_index():
            if self.index:
                self.index.save_index(str(path / "index.bin"))
                
        async def save_metadata():
            # Serialize the data items first
            serialized_data = {}
            for k, v in self.data.items():
                if hasattr(v, 'serialize'):
                    # Handle async serializable objects
                    serialized_data[str(k)] = await v.serialize()
                else:
                    # Handle basic JSON-serializable objects
                    serialized_data[str(k)] = v
            
            # Prepare metadata dict with serialized data
            metadata_dict = {
                "metadata": serialized_data,
                "dim": self.dim,
                "max_elements": self.max_elements,
                "ef_construction": self.index.ef_construction if self.index else None,
                "M": self.index.M if self.index else None
            }
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.write_metadata(path / "metadata.json", metadata_dict)
            )
            
        await asyncio.gather(save_index(), save_metadata())

    async def deserialize(self, path: Path) -> None:
        """
        Deserialize the vector store from disk.
        
        :param path: Directory path containing serialized data
        :type path: Path
        """
        # Load metadata first to get parameters
        if (path / "metadata.json").exists():
            with open(path / "metadata.json", 'rb') as f:
                data = orjson.loads(f.read())
                
                # Deserialize the data items
                self.data = {}
                for k, v in data["metadata"].items():
                    if isinstance(v, dict) and v.get("_type") == "Message":
                        # Handle Message objects
                        self.data[k] = await Message.deserialize(v)
                    else:
                        # Handle basic JSON objects
                        self.data[k] = v
                        
                self.dim = data["dim"]
                self.max_elements = data["max_elements"]
                
                # Initialize index with parameters from metadata
                if data["ef_construction"] and data["M"] and (path / "index.bin").exists():
                    self.index = hnswlib.Index(
                        space='cosine', 
                        dim=self.dim
                    )
                    # Set construction parameters before loading
                    self.index.init_index(
                        max_elements=self.max_elements,
                        ef_construction=data["ef_construction"],
                        M=data["M"]
                    )
                    # Now load the index data
                    self.index.load_index(
                        str(path / "index.bin"),
                        max_elements=self.max_elements
                    )
    