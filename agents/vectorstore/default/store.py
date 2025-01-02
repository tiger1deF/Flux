"""
HNSW vector store implementation for efficient local embeddings.

This module provides a concrete implementation of BaseVectorStore using HNSW
(Hierarchical Navigable Small World) graphs for fast approximate nearest neighbor search.
"""
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
import hnswlib
import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextvars
from threading import Lock
import json
import aiofiles

from agents.vectorstore.models import BaseVectorStore

from llm import BaseEmbeddingFunction

from agents.storage.message import Message
from agents.storage.file import File
from agents.storage.context import Context
from agents.storage.models import Chunk


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
    
    :ivar max_elements: Maximum number of elements in the index
    :type max_elements: int
    :ivar embedding_function: Function for generating embeddings
    :type embedding_function: BaseEmbeddingFunction
    :ivar index: HNSW index for similarity search
    :type index: hnswlib.Index
    :ivar data: Storage for element metadata
    :type data: Dict[int, Any]
    """
    
    _instance_lock = Lock()
    _executor = None
    embedding_function: BaseEmbeddingFunction = None
    max_elements: int = 100_000
    
    def __init__(
        self,
        embedding_function: Optional[BaseEmbeddingFunction] = None,
        ef_construction: int = 1000,
        M: int = 128,
        allow_replace_deleted: bool = False,
        lazy_indexing: bool = False
    ):
        """
        Initialize HNSW vector store.
        
        :param embedding_function: Function to generate embeddings
        :param ef_construction: Size of dynamic list for construction
        :param M: Number of bi-directional links created for every new element
        :param allow_replace_deleted: Whether to allow replacing deleted elements
        :param lazy_indexing: Whether to enable lazy indexing of chunks
        """
        super().__init__(embedding_function = embedding_function)
        
        if not embedding_function or not embedding_function.dimension:
            raise ValueError("Embedding function with dimension required")
            
        # Initialize HNSW index
        self.index = hnswlib.Index(
            space = 'cosine', 
            dim = embedding_function.dimension
        )
        self.index.init_index(
            max_elements = self.max_elements,
            ef_construction = ef_construction,
            M = M,
            allow_replace_deleted = allow_replace_deleted
        )
        
        # Store metadata directly using same IDs as index
        self.data = {}
        
        # Lazy indexing setup
        self.lazy_indexing = lazy_indexing
        self._pending_index = set()
        self._indexed = set()
        self._index_lock = asyncio.Lock()
    
    
    async def add(
        self, 
        text: Optional[Union[str, List[str]]] = None,
        embedding_function: Optional[BaseEmbeddingFunction] = None,
        embeddings: Optional[Union[np.ndarray, list[float], list[Union[np.ndarray, list[float]]]]] = None, 
        metadata: Optional[Union[List[Any], Any]] = None,
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
        if not embedding_function:
            embedding_function = self.embedding_function
        
        # Generate embeddings if needed
        if embeddings is None:
            if text:
                if isinstance(text, str):
                    text = [text]
                embeddings = await embedding_function(text)
                embeddings = np.array(embeddings)
                
                if metadata is None:
                    metadata = [Context(content = t) for t in text]
            
            elif metadata is not None:
                try:
                    if isinstance(metadata, list):
                        text = [await meta.content for meta in metadata]
                    else:
                        text = [await metadata.content]
                    embeddings = await embedding_function(text)
                    embeddings = np.array(embeddings)
                except Exception as e:
                    raise ValueError("No text or metadata provided to add in vector store!")
            else:
                raise ValueError("Either text, embeddings, or metadata with content must be provided")
                
        if not isinstance(metadata, list):
            metadata = [metadata]
            
        ids = [meta.id for meta in metadata]
    
        if len(embeddings) != len(metadata):
            raise ValueError(f"Number of embeddings ({len(embeddings)}) must match metadata ({len(metadata)})")
            
        # Add items using their own IDs directly
        with self._instance_lock:
            self.index.add_items(
                data = embeddings,
                ids = ids
            )
            
            # Store metadata with same IDs
            for id_, meta in zip(ids, metadata):
                self.data[id_] = meta
        
        return ids[0] if len(ids) == 1 else ids
    
    
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
            embedding_function = self.embedding_function
        
        embeddings = await embedding_function(query)
        
        # Adjust k if needed
        results = []
        if filter:
            # Overfetch to ensure we get enough results
            actual_k = min(k * 100, current_count)

            # Perform search
            labels, distances = self.index.knn_query(
                embeddings, 
                k = actual_k
            )
            
            for idx, dist in zip(labels[0], distances[0]):
                meta = self.data[idx]  # Use index ID directly to get metadata
                if all(getattr(meta, attr) == filter[attr] for attr in filter.keys()):
                    meta.score = 1 - dist
                    results.append(meta)
        else:
            actual_k = min(k, current_count)

            labels, distances = self.index.knn_query(
                embeddings, 
                k = actual_k
            )
        
            # Extracts results
            for idx, dist in zip(labels[0], distances[0]):
                meta = self.data[idx]  # Use index ID directly to get metadata
                meta.score = 1 - dist
                results.append(meta)
                
        return results
    
    
    async def delete(
        self, 
        ids: Union[str, List[str]]
    ) -> None:
        """
        Delete items by rebuilding index without them.
        
        :param ids: Single ID or list of IDs to delete
        :type ids: Union[str, List[str]]
        """
        if not isinstance(ids, list):
            ids = [ids]
        delete_set = set(ids)
        
        with self._instance_lock:
            # Get items to keep
            remaining_embeddings = []
            remaining_metadata = []
            remaining_ids = []
            
            valid_ids = set(self.index.get_ids_list())
            
            for id_ in self.data.keys():
                if id_ not in delete_set and id_ in valid_ids:
                    embedding = self.index.get_items([id_])[0]
                    remaining_embeddings.append(embedding)
                    remaining_metadata.append(self.data[id_])
                    remaining_ids.append(id_)
            
            # Create new index
            dimension = self.index.dim
            new_index = hnswlib.Index(space='cosine', dim=dimension)
            new_index.init_index(
                max_elements=self.max_elements,
                ef_construction=self.index.ef_construction,
                M=self.index.M
            )
            
            if remaining_embeddings:
                new_index.add_items(
                    data=np.array(remaining_embeddings),
                    ids=remaining_ids  # Use original IDs
                )
            
            self.index = new_index
            self.data = {
                id_: meta for id_, meta in zip(remaining_ids, remaining_metadata)
            }
    
    
    async def reset(self) -> None:
        """
        Reset the vector store to an empty state.
        
        Reinitializes the index with the same parameters but no data.
        """
        # Create fresh empty index with same parameters
        new_index = hnswlib.Index(
            space = 'cosine', 
            dim = self.dim
        )
        new_index.init_index(
            max_elements = self.max_elements,
            ef_construction = self.index.ef_construction,
            M = self.index.M
        )
        
        # Reset instance variables
        self.index = new_index
        self.data = {}
        
        
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
        with open(filepath, "w") as f:
            f.write(json.dumps(metadata_dict))


    async def serialize(self, path: Union[str, Path]) -> None:
        """
        Serialize the vector store to disk.
        
        :param path: Directory path to save to
        :type path: Union[str, Path]
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "metadata": {},
            "types": {},
            "config": {
                "max_elements": self.max_elements,
                "dimension": self.embedding_function.dimension,
                "ef_construction": self.index.ef_construction,
                "M": self.index.M
            }
        }

        for id_, item in self.data.items():
            if hasattr(item, 'serialize'):
                serialized = await item.serialize()
                metadata["metadata"][id_] = serialized
                metadata["types"][id_] = item.__class__.__name__
            else:
                metadata["metadata"][id_] = item
        
        async with aiofiles.open(path / "data.json", "w") as f:
            await f.write(json.dumps(metadata))

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            self.index.save_index,
            str(path / "index.bin")
        )


    async def deserialize(self, path: Union[str, Path]) -> None:
        """
        Deserialize the vector store from disk.
        
        :param path: Directory path containing serialized data
        :type path: Union[str, Path]
        """
        path = Path(path)
        
        data_path = path / "data.json"
        index_path = path / "index.bin"
        
        if not data_path.exists():
            print(f"No data file found at {data_path}, initializing empty store")
            self.data = {}
            return
        
        async with aiofiles.open(data_path, "r") as f:
            content = await f.read()
            metadata = json.loads(content)
            
            config = metadata["config"]
            self.max_elements = config["max_elements"]
            dimension = config["dimension"]
            
            self.data = {}
            for id_, item in metadata["metadata"].items():
                if isinstance(item, str):
                    if "File" in metadata["types"].get(id_, ""):
                        self.data[id_] = await File.deserialize(item)
                    elif "Message" in metadata["types"].get(id_, ""):
                        self.data[id_] = await Message.deserialize(item)
                    elif "Chunk" in metadata["types"].get(id_, ""):
                        self.data[id_] = await Chunk.deserialize(item)
                    else:
                        self.data[id_] = item
                else:
                    self.data[id_] = item
        
        if index_path.exists():
            self.index = hnswlib.Index(space='cosine', dim=dimension)
            
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self.index.load_index(
                    str(index_path),
                    max_elements=self.max_elements
                )
            )
        else:
            logger.warning(f"No index file found at {index_path}, initializing empty index")
            self.index = None
    
    
    async def update(
        self,
        id: str,
        text: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        metadata: Any = None
    ) -> None:
        """
        Update an existing item in the store by marking old as deleted and adding new.
        
        :param id: ID of item to update
        :type id: str
        :param text: New text content (will be embedded if provided)
        :type text: Optional[str]
        :param embedding: New pre-computed embedding
        :type embedding: Optional[np.ndarray]
        :param metadata: New metadata
        :type metadata: Any
        """
        if text is not None:
            # Generate new embedding
            embedding = (await self.internal_embed([text]))[0]
        
        if embedding is not None:
            # Mark old item as deleted and add new embedding at same position
            index_id = self.id_to_index[id]
            self.index.mark_deleted(index_id)
            self.index.add_items(
                data = embedding,
                ids = [index_id]
            )
        
        if metadata is not None:
            # Update metadata
            self.data[id] = metadata


 