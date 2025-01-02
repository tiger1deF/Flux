"""
State management models for agents.

This module provides state management functionality including message history,
context storage, file management, and metadata handling.
"""
import os
from typing import Dict, Any, List, Tuple, ClassVar, Optional, Callable, Union
import tempfile
from pathlib import Path
from pydantic import BaseModel, Field
import json
import aiofiles
import asyncio
from contextlib import asynccontextmanager
from difflib import SequenceMatcher, unified_diff
import polars as pl
import numpy as np
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial, cached_property
import multiprocessing
from uuid import uuid4

from llm import BaseEmbeddingFunction

# Import message and file types
from agents.storage.message import Message, MessageType
from agents.storage.metadata import Metadata
from agents.storage.file import File
from agents.storage.models import Chunk

from agents.vectorstore.models import BaseVectorStore
from agents.vectorstore.default.store import HNSWStore

from agents.config.models import ContextConfig, RetrievalType

from agents.vectorstore.truncation import truncate_items

from agents.storage.filestore import FileStore
from agents.storage.context import Context, ContextType

from utils.serialization import decompress_and_deserialize


class AgentState(BaseModel):
    """
    State management model for agents.
    
    Handles message history, context storage, file management, and metadata
    with thread-safe operations.
    
    :cvar _metadata_lock: Lock for thread-safe metadata access
    :type _metadata_lock: ClassVar[threading.RLock]
    

    :ivar context: Context variable store
    :type context: Dict[str, Any]
    :ivar files: File store
    :type files: Dict[str, File]
    :ivar metadata: Metadata store
    :type metadata: Dict[str, Any]
    :ivar temp_dir: Temporary directory path
    :type temp_dir: Optional[str]
    :ivar session_id: Unique session identifier
    :type session_id: str
    """
    _metadata_lock: ClassVar[threading.RLock] = threading.RLock()
    
    session_id: str = Field(default = str(uuid4()), exclude = True)
    temp_dir: Optional[str] = Field(default = None, exclude = True)
   
    # Context configuration
    context_config: ContextConfig = Field(
        default_factory = ContextConfig,
        description = "Configuration for context handling and retrieval"
    )
    
    embedding_function: Optional[BaseEmbeddingFunction] = Field(
        default = None,
        description = "Shared embedding function for all stores"
    )
    
    # These will be initialized in __init__ with the embedding function
    context_store: Optional[BaseVectorStore] = None
    file_store: Optional[FileStore] = None 
    metadata_store: Optional[BaseVectorStore] = None
    
    context: Dict[str, Any] = Field(
        default_factory = dict,
        description = "Runtime context storage"
    )
    
    def __init__(
        self,
        embedding_function: Optional[BaseEmbeddingFunction] = None,
        **kwargs
    ):
        """Initialize agent state."""
        # First initialize the base model with the embedding function
        super().__init__(embedding_function=embedding_function, **kwargs)
        
        # Then set up the embedding function and stores
        if not embedding_function:
            from llm.embeddings.default import generate_static_embeddings
            embedding_function = BaseEmbeddingFunction(
                generate_static_embeddings,
                dimension=768
            )
            
        # Store the embedding function
        self.embedding_function = embedding_function
            
        # Initialize stores with the embedding function
        self.context_store = HNSWStore(embedding_function=embedding_function)
        self.file_store = FileStore(embedding_function=embedding_function)
        self.metadata_store = HNSWStore(embedding_function=embedding_function)
    
    class Config:
        arbitrary_types_allowed = True 

    async def search_file_chunks(self, query: str) -> List[Chunk]:
        return await self.file_store.search_chunks(
            query = query,
            k = self.context_config.item_count,
            chunk_size = self.context_config.chunk_size,
            chunk_overlap = self.context_config.chunk_overlap
        )
        
    
    @cached_property 
    def _executor(self) -> ThreadPoolExecutor:
        """
        Get or create thread pool executor for metadata operations.
        
        :return: Thread pool executor
        :rtype: ThreadPoolExecutor
        """
        try:
            worker_threads = len(threading.enumerate())
            thread_count = max(2, min(worker_threads, multiprocessing.cpu_count()))
        except:
            thread_count = multiprocessing.cpu_count()
            
        return ThreadPoolExecutor(
            max_workers = thread_count,
            thread_name_prefix = 'agent_metadata_'
        )
    
    async def _run_in_executor(self, func, *args):
        """
        Run a function in the thread pool executor.
        
        :param func: Function to execute
        :type func: Callable
        :param args: Arguments for the function
        :return: Function result
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(func, *args)
        )
    
    @property
    def temp_directory(self) -> Path:
        """
        Get or create temporary directory for file operations.
        
        :return: Path to temporary directory
        :rtype: Path
        """
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix = f"agent_workspace_{self.session_id}_{uuid4()}_")
        return Path(self.temp_dir)
    
    async def cleanup(self):
        """
        Clean up temporary directory and resources.
        """
        if self.temp_dir and Path(self.temp_dir).exists():
            await asyncio.to_thread(self._cleanup_sync)
    
    def _cleanup_sync(self):
        """
        Synchronous cleanup of temporary directory.
        """
        import shutil
        shutil.rmtree(self.temp_dir)
        self.temp_dir = None
    
    async def __adel__(self):
        """
        Async cleanup on deletion.
        """
        await self.cleanup()


    async def create_file(
        self,
        filename: str,
        content: str = "",
        description: str = "",
        annotations: Dict[str, Any] = {}
    ) -> File:
        """
        Create a new file in the workspace.
        
        :param filename: Name of file to create
        :type filename: str
        :param content: Initial file content
        :type content: str
        :param description: File description
        :type description: str
        :param annotations: Additional metadata
        :type annotations: Dict[str, Any]
        :return: Created file object
        :rtype: File
        """
        file_path = self.temp_directory / filename
        
        # If content is str, use text mode
        if isinstance(content, str):
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(content)
        else:
            # For bytes/binary content
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            
        file = File(
            data = content,
            path = str(file_path),
            description = description,
            annotations = annotations
        )
        
        await self.file_store.add_file(file)
        return file


    async def read_file(self, filename: str) -> str:
        """
        Read contents of a file.
        
        :param filename: Name of file to read
        :type filename: str
        :return: File contents
        :rtype: str
        """
        file_path = self.temp_directory / filename
        # Default to text mode for reading
        async with aiofiles.open(file_path, 'r') as f:
            return await f.read()


    async def delete_file(self, filename: str) -> None:
        """
        Delete a file.
        
        :param filename: Name of file to delete
        :type filename: str
        """
        file_path = self.temp_directory / filename
        if file_path.exists():
            await asyncio.to_thread(file_path.unlink)
            await self.file_store.remove_file(filename)


    async def read_directory(self, directory: Path) -> List[File]:
        """
        Read all files from a directory recursively.
        
        :param directory: Directory to read
        :type directory: Path
        :return: List of imported files
        :rtype: List[File]
        """
        return await self.file_store.read_from_directory(directory)


    async def write_to_directory(self, directory: Path) -> Dict[str, Path]:
        """
        Write all files to a directory.
        
        :param directory: Target directory
        :type directory: Path
        :return: Mapping of file IDs to written paths
        :rtype: Dict[str, Path]
        """
        return await self.file_store.write_to_directory(directory)


    async def add_file(
        self,
        files: Union[File, List[File]]
    ) -> None:
        """
        Add file(s) to the file store.
        
        :param files: Single file or list of files to add
        :type files: Union[File, List[File]]
        """
        if not isinstance(files, list):
            files = [files]

        async def add_file(file):   
            await self.file_store.add_file(file)

        await asyncio.gather(*[add_file(file) for file in files])


    async def add_metadata(
        self,
        metadata: Union[Metadata, List[Metadata]]
    ) -> None:
        """
        Add metadata item(s) to the metadata store.
        
        :param metadata: Single metadata item or list of items to add
        :type metadata: Union[Metadata, List[Metadata]]
        """
        if not isinstance(metadata, list):
            metadata = [metadata]
        for meta in metadata:
            await self.metadata_store.add(
                metadata = meta,
                text = await meta.content
            )


    async def obtain_message_context(
        self,
        query: Message,
        context_config: ContextConfig,
        retrieval_strategy: RetrievalType = None,
        filter: Optional[Dict[str, Any]] = None,
        truncate: bool = True
    ) -> List[Message]:
        """
        Obtain message context from message store.
        
        :param query: Query message
        :type query: Message
        :param context_config: Context configuration
        :type context_config: ContextConfig
        :param filter: Optional filter criteria
        :type filter: Optional[Dict[str, Any]]
        :param truncate: Whether to truncate results
        :type truncate: bool
        :return: List of relevant messages
        :rtype: List[Message]
        """
        return await self._obtain_context(
            query = query,
            store = self.message_store,
            context_config = context_config,
            retrieval_strategy = retrieval_strategy,
            filter = filter,
            truncate = truncate
        )


    async def obtain_file_context(
        self,
        query: str,
        context_config: ContextConfig,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[File]:
        """
        Get relevant complete files using FileStore's file search.
        
        :param query: Search query
        :type query: str
        :param context_config: Context configuration
        :type context_config: ContextConfig 
        :param retrieval_strategy: Retrieval strategy
        :type retrieval_strategy: RetrievalType
        :param filter: Optional filter criteria
        :type filter: Optional[Dict[str, Any]]
        :return: List of relevant files
        :rtype: List[File]
        """
        return await self.file_store.search_files(
            query = query,
            k = context_config.item_count
        )


    async def obtain_metadata_context(
        self,
        query: Metadata,
        context_config: ContextConfig,
        retrieval_strategy: RetrievalType = None,
        filter: Optional[Dict[str, Any]] = None,
        truncate: bool = True
    ) -> List[Metadata]:
        """
        Obtain metadata context from metadata store.
        
        :param query: Query metadata
        :type query: Metadata
        :param context_config: Context configuration
        :type context_config: ContextConfig
        :param retrieval_strategy: Retrieval strategy
        :type retrieval_strategy: RetrievalType
        :param filter: Optional filter criteria
        :type filter: Optional[Dict[str, Any]]
        :param truncate: Whether to truncate results
        :type truncate: bool
        :return: List of relevant metadata items
        :rtype: List[Metadata]
        """
        return await self._obtain_context(
            query = query,
            store = self.metadata_store,
            context_config = context_config,
            retrieval_strategy = retrieval_strategy,
            filter = filter,
            truncate = truncate
        )


    async def _obtain_context(
        self,
        query: Union[Message, File, Metadata],
        store: Any,
        context_config: ContextConfig,
        retrieval_strategy: RetrievalType = None,
        filter: Optional[Dict[str, Any]] = None,
        truncate: bool = True
    ) -> List[Any]:
        """
        Internal method to obtain context from a store.
        
        :param query: Query item
        :param store: Store to search in
        :param context_config: Context configuration
        :param retrieval_strategy: Retrieval strategy
        :param filter: Optional filter criteria
        :param truncate: Whether to truncate results
        :return: List of items fetched from store
        """
        if not retrieval_strategy:
            retrieval_strategy = context_config.retrieval_strategy
        
        if retrieval_strategy == RetrievalType.NONE:
            return []
        
        elif retrieval_strategy == RetrievalType.SIMILARITY:
            retrieved_items = await store.search_relevant(
                query = query,
                filter = filter,
                k = context_config.item_count
            )
            if truncate:
                retrieved_items = await truncate_items(
                    items = retrieved_items,
                    context_config = context_config
                )
        elif retrieval_strategy == RetrievalType.CHRONOLOGICAL:
            retrieved_items = await store.search_history(
                query = query,
                filter = filter,
                k = context_config.item_count
            )
            
            if truncate:
                retrieved_items = await truncate_items(
                    items = retrieved_items,
                    context_config = context_config
                )
                
        elif retrieval_strategy == RetrievalType.ALL:
            relevant_items = await store.search_relevant(
                query = query,
                filter = filter,
                k = context_config.item_count
            )
            
            history_items = await store.search_history(
                query = query,
                filter = filter,
                k = context_config.item_count
            )
            
            if truncate:
                modified_config = context_config / 2  # Split config for each part
                relevant_items = await truncate_items(
                    items=relevant_items,
                    context_config=modified_config
                )
                history_items = await truncate_items(
                    items = history_items,
                    context_config = modified_config
                )
            
            retrieved_items = relevant_items + history_items
            
        return retrieved_items
    
    
    ###################
    # File Management # 
    ###################
    
    async def write_file(
        self, 
        filename: str, 
        content: str,
        annotations: Dict[str, Any] = {},
        description: str = ""
    ) -> File:  
        """
        Write content to a file in the agent's workspace.
        
        :param filename: Name of the file to write
        :type filename: str
        :param content: Content to write
        :type content: str
        :param annotations: Metadata annotations for the file
        :type annotations: Dict[str, Any], optional
        :param description: File description
        :type description: str, optional
        :return: Created/updated file object
        :rtype: File
        """
        return await self.create_file(
            filename, 
            content, 
            annotations, 
            description
        )
    
    
    async def append_file(
        self, 
        filename: str, 
        content: str,
        annotations: Dict[str, Any] = {},
        description: str = ""
    ) -> File:
        """
        Append content to an existing file.
        
        :param filename: Name of the file to append to
        :type filename: str
        :param content: Content to append
        :type content: str
        :param annotations: Additional metadata annotations
        :type annotations: Dict[str, Any], optional
        :param description: Additional description
        :type description: str, optional
        :return: Updated file object
        :rtype: File
        """
        file_path = self.temp_directory / filename
        async with aiofiles.open(file_path, 'a') as f:
            await f.write(content)
        
        file = self.files[filename]
        file.data = file.data + content
        file.annotations = file.annotations | annotations
        file.description = file.description + description
        
        self.files[filename] = file
        
        return file
    
    
    async def edit_file(
        self,
        filename: str,
        new_content: str,
        edit_description: str = "",
        annotations: Dict[str, Any] = {}
    ) -> File:
        """
        Edit an existing file with new content and track changes.
        """
        # First remove old file from stores
        await self.file_store.remove_file(filename)
        
        # Create new file object
        file_path = self.temp_directory / filename
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(new_content)
        
        file = File(
            content = new_content,
            path = str(file_path),
            description = edit_description,
            annotations = annotations
        )
        
        # Add updated file to store
        await self.file_store.add_file(file)
        
        return file
    
    
    async def list_files(self, pattern: str = "*") -> List[Path]:
        """
        List files in the agent's workspace matching a pattern.
        
        :param pattern: Glob pattern for matching files
        :type pattern: str
        :return: List of matching file paths
        :rtype: List[Path]
        """
        return list(self.temp_directory.glob(pattern))
    
    
    @asynccontextmanager
    async def temp_file(self, suffix: str = None, prefix: str = None):
        """
        Create a temporary file in the agent's workspace.
        
        :param suffix: Optional suffix for the filename
        :type suffix: str, optional
        :param prefix: Optional prefix for the filename
        :type prefix: str, optional
        :yield: Path to the temporary file
        :rtype: Path
        """
        temp_file = tempfile.NamedTemporaryFile(
            dir = self.temp_directory,
            suffix = suffix,
            prefix = prefix,
            delete = False
        )
        try:
            yield Path(temp_file.name)
        finally:
            await self.delete_file(Path(temp_file.name).name)
    
    
    async def save_json(self, filename: str, data: Any):
        """
        Save data as JSON to a file.
        
        :param filename: Name of the file to save
        :type filename: str
        :param data: Data to serialize as JSON
        :type data: Any
        """
        async with aiofiles.open(self.temp_directory / filename, 'w') as f:
            await f.write(json.dumps(data, indent = 2))
    
    
    ######################
    # Metadata Management # 
    #######################
    
    @asynccontextmanager
    async def access_metadata(self, key: str = None) -> Any:
        """
        Thread-safe access to metadata store.
        
        :param key: Optional key to access specific metadata
        :type key: str, optional
        :yield: Metadata dictionary or specific value
        :rtype: Any
        """
        async with self._async_lock():
            try:
                if key is None:
                    yield self.metadata
                else:
                    yield self.metadata.get(key)
            finally:
                pass  


    async def get_metadata(self, key: str = None) -> Any:
        """
        Get metadata value(s).
        
        :param key: Optional key to get specific metadata
        :type key: str, optional
        :return: Metadata value or entire metadata dict
        :rtype: Any
        """
        async with self.access_metadata(key) as metadata:
            return metadata


    async def update_metadata(self, 
        metadata: Dict[str, Any], 
        merge: bool = True
    ) -> None:
        """
        Update metadata with optional merging of values.
        
        :param metadata: New metadata to update
        :type metadata: Dict[str, Any]
        :param merge: Whether to merge with existing values
        :type merge: bool, optional
        """
        async with self.access_metadata() as current_metadata:
            updates = []
            direct_updates = {}
            
            for key, value in metadata.items():
                if key not in current_metadata or not merge:
                    direct_updates[key] = value
                else:
                    updates.append((key, value, current_metadata[key]))
            
            current_metadata.update(direct_updates)
            
            if updates:
                merge_tasks = [
                    self._merge_value(k, v, e) 
                    for k, v, e in updates
                ]
                results = await asyncio.gather(*merge_tasks, return_exceptions=True)
                
                for result in results:
                    if not isinstance(result, Exception):
                        key, merged_value = result
                        current_metadata[key] = merged_value


    async def _merge_value(self, key: str, value: Any, existing: Any) -> Tuple[str, Any]:
        """
        Merge two values based on their types.
        
        :param key: Metadata key
        :type key: str
        :param value: New value
        :type value: Any
        :param existing: Existing value
        :type existing: Any
        :return: Tuple of key and merged value
        :rtype: Tuple[str, Any]
        """
        if isinstance(value, pl.DataFrame) and isinstance(existing, pl.DataFrame):
            if set(value.columns) == set(existing.columns):
                merged_df = await self._run_in_executor(
                    lambda: pl.concat([existing, value], how="vertical"),
                    ignore_index=True
                )
                return key, merged_df
            return f"{key}_new", value
            
        elif isinstance(value, (dict, Dict)) and isinstance(existing, (dict, Dict)):
            return key, await self._deep_merge_dicts(existing, value)
            
        elif isinstance(value, str) and isinstance(existing, str):
            if not re.search(r'<[^>]+>', value + existing):
                ratio = await self._run_in_executor(
                    lambda: SequenceMatcher(None, existing, value).ratio()
                )
                if ratio > 0.3:
                    if len(value) + len(existing) > 10000:
                        merged = await self._run_in_executor(
                            self._merge_strings, existing, value
                        )
                    else:
                        merged = self._merge_strings(existing, value)
                    return key, merged
            return f"{key}_new", value
            
        elif isinstance(value, (list, tuple)) and isinstance(existing, (list, tuple)):
            if len(value) + len(existing) > 10000:
                merged = await self._run_in_executor(
                    lambda: list(existing) + list(value)
                )
                return key, merged
            return key, list(existing) + list(value)
            
        elif isinstance(value, set) and isinstance(existing, set):
            if len(value) + len(existing) > 10000:
                merged = await self._run_in_executor(
                    lambda: existing.union(value)
                )
                return key, merged
            return key, existing.union(value)
            
        elif isinstance(value, np.ndarray) and isinstance(existing, np.ndarray):
            try:
                merged = await self._run_in_executor(
                    np.concatenate, [existing, value]
                )
                return key, merged
            except ValueError:
                return f"{key}_new", value
                
        return f"{key}_new", value


    def _merge_strings(self, existing: str, value: str) -> str:
        """
        Merge two strings using sequence matcher.
        
        :param existing: Existing string
        :type existing: str
        :param value: New string
        :type value: str
        :return: Merged string
        :rtype: str
        """
        merged = []
        matcher = SequenceMatcher(None, existing, value)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                merged.append(existing[i1:i2])
            else:
                merged.append(existing[i1:i2])
                merged.append(value[j1:j2])
        return ''.join(merged)


    async def _deep_merge_dicts(self, dict1: Dict, dict2: Dict) -> Dict:
        """
        Recursively merge two dictionaries.
        
        :param dict1: First dictionary
        :type dict1: Dict
        :param dict2: Second dictionary
        :type dict2: Dict
        :return: Merged dictionary
        :rtype: Dict
        """
        merged = dict1.copy()
        merge_tasks = []
        
        for key, value in dict2.items():
            if (
                key in merged and 
                isinstance(merged[key], dict) and 
                isinstance(value, dict)
            ):
                merge_tasks.append((key, self._deep_merge_dicts(merged[key], value)))
            else:
                merged[key] = value
                
        if merge_tasks:
            results = await asyncio.gather(
                *(task[1] for task in merge_tasks)
            )
            for (key, _), result in zip(merge_tasks, results):
                merged[key] = result
                
        return merged


    async def filter_metadata(self, filter_fn: Callable[[str, Any], bool]) -> Dict[str, Any]:
        """
        Filter metadata using a predicate function.
        
        :param filter_fn: Function to filter metadata items
        :type filter_fn: Callable[[str, Any], bool]
        :return: Filtered metadata dictionary
        :rtype: Dict[str, Any]
        """
        async with self.access_metadata() as metadata:
            filter_tasks = [
                self._run_in_executor(filter_fn, k, v)
                for k, v in metadata.items()
            ]
            results = await asyncio.gather(*filter_tasks)
            return {
                k: v for (k, v), keep 
                in zip(metadata.items(), results) 
                if keep
            }


    async def transform_metadata(self, transform_fn: Callable[[str, Any], Any]) -> None:
        """
        Transform metadata values using a function.
        
        :param transform_fn: Function to transform metadata values
        :type transform_fn: Callable[[str, Any], Any]
        """
        async with self.access_metadata() as metadata:
            transform_tasks = [
                self._run_in_executor(transform_fn, k, v)
                for k, v in metadata.items()
            ]
            results = await asyncio.gather(*transform_tasks)
            metadata.clear()
            metadata.update(dict(zip(metadata.keys(), results)))


    @asynccontextmanager 
    async def _async_lock(self):
        """
        Context manager for thread-safe metadata access.
        
        :yield: None
        """
        with self._metadata_lock:
            yield


    def to_dict(self) -> dict:
        """
        Convert agent state to dictionary.
        
        :return: Dictionary representation of state
        :rtype: dict
        """
        return {
            'messages': self.message_store.to_dict(),
            'metadata': self.metadata_store.to_dict(),
            'files': self.file_store.to_dict(),
            'temp_directory': str(self.temp_directory),
            'temp_files_count': len(self.list_files())
        }


    async def write_files_to_temp(self) -> Dict[str, Path]:
        """Write all files from store to temporary directory"""
        return await self.file_store.write_to_directory(self.temp_directory)


    async def read_files_from_directory(self, directory: Path) -> List[File]:
        """Read all files from directory into store"""
        return await self.file_store.read_from_directory(directory)


    async def serialize(self, path: Union[str, Path]) -> None:
        """
        Serialize agent state to disk.
        
        :param path: Directory path to save serialized data
        :type path: Union[str, Path]
        """
        path = Path(path)
        os.makedirs(path, exist_ok=True)
        
        # Save core state data
        state_data = {
            "session_id": self.session_id,
            "temp_dir": self.temp_dir,
            "embedding_function": await self.embedding_function.serialize() if self.embedding_function else None
        }
        
        # Save JSON data in text mode
        async with aiofiles.open(path / "state.json", "w") as f:
            await f.write(json.dumps(state_data))
        
        # Serialize stores concurrently
        await asyncio.gather(
            self.context_store.serialize(path / "context"),
            self.metadata_store.serialize(path / "metadata"),
            self.file_store.serialize(path / "files")
        )

    async def deserialize(self, path: Union[str, Path]) -> None:
        """
        Deserialize agent state from disk.
        
        :param path: Directory path containing serialized data
        :type path: Union[str, Path]
        """
        path = Path(path)
        
        # Read JSON data in text mode
        async with aiofiles.open(path / "state.json", "r") as f:
            state_data = json.loads(await f.read())
            self.session_id = state_data["session_id"]
            self.temp_dir = state_data["temp_dir"]
            if state_data.get("embedding_function"):
                self.embedding_function = await BaseEmbeddingFunction.deserialize(
                    state_data["embedding_function"]
                )
        
        # Load stores concurrently
        await asyncio.gather(
            self.context_store.deserialize(path / "context"),
            self.metadata_store.deserialize(path / "metadata"),
            self.file_store.deserialize(path / "files")
        )


    async def obtain_file_chunk_context(
        self,
        query: str,
        context_config: ContextConfig,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Get formatted file chunks with context.
        
        :param query: Search query
        :type query: str
        :param context_config: Configuration for context retrieval
        :type context_config: ContextConfig
        :param filter: Optional filter criteria
        :type filter: Optional[Dict[str, Any]]
        :return: List of formatted file chunks
        :rtype: List[str]
        """
        chunks = await self.file_store.search_chunks(
            query = query,
            k = context_config.item_count,
            chunk_size = context_config.chunk_size,
            chunk_overlap = context_config.chunk_overlap
        )
        
        # Format chunks with file context
        formatted_chunks = []
        for chunk in chunks:
            file_path = chunk.path
            chunk_text = chunk.content
            formatted = f"From file {file_path}:\n{chunk_text}"
            formatted_chunks.append(formatted)
                
        return formatted_chunks


    async def ingest_message_data(self, message: Message) -> None:
        """
        Ingest files and metadata from a message into their respective stores.
        Handles both serialized data and direct objects.
        
        :param message: Message containing data to ingest
        :type message: Message
        """
        tasks = []
        
        # Process files
        for file_data in message.files:
            if isinstance(file_data, (str, bytes)):  # Serialized data
                file_obj = await decompress_and_deserialize(file_data, File)
            elif isinstance(file_data, File):  # Already a File object
                file_obj = file_data
            else:  # Create new File object from raw data
                file_obj = await File.create(data = file_data)
                
            tasks.append(self.file_store.add_file(file_obj))
      
        for meta_data in message.metadata:
            if isinstance(meta_data, (str, bytes)):  # Serialized data
                meta_obj = await decompress_and_deserialize(meta_data, Metadata)
            elif isinstance(meta_data, Metadata):  # Already a Metadata object
                meta_obj = meta_data
            else:  # Create new Metadata object from raw data
                meta_obj = Metadata(data = meta_data)
                
            tasks.append(self.metadata_store.add(metadata = meta_obj))
        
        # Run all ingestion tasks in parallel
        if tasks:
            await asyncio.gather(*tasks)


    async def add_context(self, context: Context) -> None:
        """
        Add a context entry to the store.
        
        :param context: Context entry to add
        :type context: Context
        """
        await self.context_store.add(
            text=context.content,
            metadata=context
        )

    async def get_context(
        self,
        query: str,
        context_type: Optional[ContextType] = None,
        limit: int = 10
    ) -> List[Context]:
        """
        Retrieve relevant context entries.
        
        :param query: Search query
        :type query: str
        :param context_type: Optional type filter
        :type context_type: Optional[ContextType]
        :param limit: Maximum number of entries to return
        :type limit: int
        :return: List of relevant context entries
        :rtype: List[Context]
        """
        filter_dict = {}
        if context_type:
            filter_dict["type"] = context_type

        return await self.context_store.search_relevant(
            query=query,
            filter=filter_dict,
            k=limit
        )
