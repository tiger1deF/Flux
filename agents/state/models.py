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
import datetime
from difflib import SequenceMatcher, unified_diff
import polars as pl
import numpy as np
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial, cached_property
import multiprocessing
from uuid import uuid4

# Import message and file types
from agents.messages.message import Message, MessageType
from agents.messages.metadata import Metadata
from agents.messages.file import File

from agents.vectorstore.default.store import HNSWStore

from agents.config.models import ContextConfig, RetrievalType

from agents.vectorstore.truncation import truncate_items


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
   
    # Context stores
    message_store: HNSWStore = Field(
        default_factory = lambda: HNSWStore(),
        description = "Store for message history and retrieval"
    )
    file_store: HNSWStore = Field(
        default_factory = lambda: HNSWStore(),
        description = "Store for files"
    )
    metadata_store: HNSWStore = Field(
        default_factory = lambda: HNSWStore(),
        description = "Store for metadata"
    )
    context: Dict[str, Any] = Field(
        default_factory = dict,
        description = "Runtime context storage"
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
        path: Union[str, Path] = None,
        content: str = "",
        annotations: Dict[str, Any] = {},
        description: str = ""
    ):
        """
        Create a new file in the agent's workspace.
        
        :param filename: Name of the file to create
        :type filename: str
        :param path: Optional custom path, defaults to temp directory
        :type path: Union[str, Path], optional
        :param content: Initial file content
        :type content: str, optional
        :param annotations: Metadata annotations for the file
        :type annotations: Dict[str, Any], optional
        :param description: File description
        :type description: str, optional
        :return: Created file object
        :rtype: File
        """
        if path is None:
            file_path = self.temp_directory / filename
        else:
            file_path = Path(path) / filename
        
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(content)
        
        file = File(
            content = content,
            path = str(file_path),
            description = description,
            annotations = annotations
        )
        await self.add_file(file)
        
    
    async def read_file(self, filename: str) -> str:
        """
        Read contents of a file from the agent's workspace.
        
        :param filename: Name of the file to read
        :type filename: str
        :return: File contents
        :rtype: str
        """
        file_path = self.temp_directory / filename
        async with aiofiles.open(file_path, 'r') as f:
            return await f.read()
    

    async def add_message(
        self,
        messages: Union[Message, List[Message]]
    ) -> None:
        """
        Add message(s) to the message store.
        
        :param messages: Single message or list of messages to add
        :type messages: Union[Message, List[Message]]
        """
        if not isinstance(messages, list):
            messages = [messages]
        for message in messages:
            await self.message_store.add(
                metadata = message,
                text = message.content
            )


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
    
        for file in files:
            await self.file_store.add(
                metadata = file,
                text = file.content
            )


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
            filter = filter,
            truncate = truncate
        )


    async def obtain_file_context(
        self,
        query: File,
        context_config: ContextConfig,
        filter: Optional[Dict[str, Any]] = None,
        truncate: bool = True
    ) -> List[File]:
        """
        Obtain file context from file store.
        
        :param query: Query file
        :type query: File 
        :param context_config: Context configuration
        :type context_config: ContextConfig
        :param filter: Optional filter criteria
        :type filter: Optional[Dict[str, Any]]
        :param truncate: Whether to truncate results
        :type truncate: bool
        :return: List of relevant files
        :rtype: List[File]
        """
        return await self._obtain_context(
            query = query,
            store = self.file_store,
            context_config = context_config,
            filter = filter,
            truncate = truncate
        )


    async def obtain_metadata_context(
        self,
        query: Metadata,
        context_config: ContextConfig,
        filter: Optional[Dict[str, Any]] = None,
        truncate: bool = True
    ) -> List[Metadata]:
        """
        Obtain metadata context from metadata store.
        
        :param query: Query metadata
        :type query: Metadata
        :param context_config: Context configuration
        :type context_config: ContextConfig
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
            filter = filter,
            truncate = truncate
        )


    async def _obtain_context(
        self,
        query: Union[Message, File, Metadata],
        store: Any,
        context_config: ContextConfig,
        filter: Optional[Dict[str, Any]] = None,
        truncate: bool = True
    ) -> List[Any]:
        """
        Internal method to obtain context from a store.
        
        :param query: Query item
        :param store: Store to search in
        :param context_config: Context configuration
        :param filter: Optional filter criteria
        :param truncate: Whether to truncate results
        :return: List of relevant items
        """
        if context_config.strategy == RetrievalType.DISABLED:
            return []
        
        elif context_config.strategy == RetrievalType.RELEVANT:
            retrieved_items = store.search_relevant(
                query,
                filter = filter,
                k = context_config.item_count
            )
            if truncate:
                retrieved_items = await truncate_items(
                    items = retrieved_items,
                    context_config = context_config
                )
           
        elif context_config.strategy == RetrievalType.HISTORY:
            retrieved_items = store.search_history(
                query,
                filter = filter,
                k = context_config.item_count
            )
            if truncate:
                retrieved_items = await truncate_items(
                    items = retrieved_items,
                    context_config = context_config
                )

        elif context_config.strategy == RetrievalType.ALL:
            relevant_items = store.search_relevant(
                query,
                filter = filter,
                k = context_config.item_count // 2
            )
            history_items = store.search_history(
                query,
                filter = filter,
                k = context_config.item_count // 2
            )
            if truncate:
                retrieved_items = await truncate_items(
                    items = relevant_items + history_items,
                    context_config = context_config
                )
        else:
            raise ValueError(f"Invalid retrieval strategy: {context_config.strategy}")
      
        return retrieved_items
    
    
    def flush_context(self, **kwargs) -> None:
        """
        Remove items from context store.
        
        :param kwargs: Optional keys to remove, if none provided clears all context
        """
        if not kwargs:
            self.context.clear()
            return
        
        for key in kwargs.keys():
            if key in self.context:
                self.context.pop(key)
    
    
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
        file.content = file.content + content
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
        
        :param filename: Name of the file to edit
        :type filename: str
        :param new_content: New content to write
        :type new_content: str
        :param edit_description: Description of the edit
        :type edit_description: str, optional
        :param annotations: Additional metadata annotations
        :type annotations: Dict[str, Any], optional
        :return: Updated file object with edit history
        :rtype: File
        """
        if filename not in self.files:
            return await self.create_file(
                filename,
                new_content,
                annotations,
                edit_description
            )

        file = self.files[filename]
        old_content = file.content
        differ = SequenceMatcher(None, old_content, new_content)
        changes = []
        
        for tag, i1, i2, j1, j2 in differ.get_opcodes():
            if tag != 'equal':
                changes.append({
                    'type': tag,
                    'old_start': i1,
                    'old_end': i2,
                    'new_start': j1, 
                    'new_end': j2,
                    'old_text': old_content[i1:i2],
                    'new_text': new_content[j1:j2]
                })

        diff_text = '\n'.join(unified_diff(
            old_content.splitlines(True),
            new_content.splitlines(True),
            fromfile = 'previous',
            tofile = 'new'
        ))

        file_path = self.temp_directory / filename
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(new_content)  
        file.content = new_content
        
        if edit_description:
            if file.description:
                file.description += f"\n{edit_description}"
            else:
                file.description = edit_description

        change_meta = {
            "changes": changes,
            "diff": diff_text,
            "last_edited": str(datetime.datetime.now())
        }
        
        file.annotations = file.annotations | annotations | {"edit_history": change_meta}
        if "edit_count" in file.annotations:
            file.annotations["edit_count"] += 1
        else:
            file.annotations["edit_count"] = 1

        self.files[filename] = file
        
        return file
    
    
    async def delete_file(self, filename: str):
        """
        Delete a file from the agent's workspace.
        
        :param filename: Name of the file to delete
        :type filename: str
        """
        file_path = self.temp_directory / filename
        if file_path.exists():
            await asyncio.to_thread(file_path.unlink)
            del self.files[filename]
    
    
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


    def __repr__(self):
        """
        Get string representation of agent state.
        
        :return: String representation
        :rtype: str
        """
        return f"AgentState(num_messages={len(self.messages)}, num_context_vars={len(self.context)}, num_files={len(self.files)})"


    def to_dict(self) -> dict:
        """
        Convert agent state to dictionary.
        
        :return: Dictionary representation of state
        :rtype: dict
        """
        return {
            'messages': [message.to_dict() for message in self.messages],
            'metadata': self.metadata.to_dict(),
            'context': self.context,
            'temp_directory': str(self.temp_directory),
            'temp_files': [str(file) for file in self.list_files()],
            'temp_files_count': len(self.list_files())
        }


    async def write_files_to_temp(self) -> Dict[str, Path]:
        """
        Asynchronously write all files from messages to temporary directory.
        Preserves original paths and creates necessary subdirectories.
        
        :return: Dictionary mapping file IDs to their written paths
        :rtype: Dict[str, Path]
        """
        async def write_single_file(file_id: str, file: File) -> Tuple[str, Path]:
            if not file.path:
                return file_id, None
            
            # Create full path in temp directory
            rel_path = Path(file.path).relative_to(file.path.anchor)
            temp_path = self.temp_directory / rel_path
            
            # Create parent directories
            os.makedirs(temp_path.parent, exist_ok=True)
            
            # Write file content
            if isinstance(file.content, bytes):
                async with aiofiles.open(temp_path, 'wb') as f:
                    await f.write(file.content)
            else:
                async with aiofiles.open(temp_path, 'w') as f:
                    await f.write(str(file.content))
                
            return file_id, temp_path

        # Gather all files from messages
        file_tasks = []
        for file_id, file in self.files.items():
            if file.path:  # Only process files with paths
                file_tasks.append(write_single_file(file_id, file))
            
        # Execute all file writes in parallel
        results = await asyncio.gather(*file_tasks, return_exceptions=True)
        
        # Filter out failed writes and create result dictionary
        written_files = {
            file_id: path 
            for file_id, path in results 
            if isinstance(path, Path)
        }
        
        return written_files


    async def serialize(self, path: Union[str, Path]) -> None:
        """
        Serialize agent state to disk.
        
        :param path: Directory path to save serialized data
        :type path: Union[str, Path]
        """
        path = Path(path)
        os.makedirs(path, exist_ok=True)
        
        # Serialize stores concurrently
        async def save_messages_store():
            await self.message_store.serialize(path / "messages_store")
            
        async def save_files():
            serialized_files = {}
            for name, file in self.file_store.data.items():
                serialized_files[name] = await file.serialize()
            
            async with aiofiles.open(path / "files.json", "w") as f:
                await f.write(json.dumps(serialized_files))
            
        async def save_metadata():
            serialized_metadata = {}
            for key, meta in self.metadata_store.data.items():
                serialized_metadata[key] = await meta.serialize()
            
            async with aiofiles.open(path / "metadata.json", "w") as f:
                await f.write(json.dumps(serialized_metadata))
            
        # Save core state data
        state_data = {
            "session_id": self.session_id,
            "temp_dir": self.temp_dir
        }
        
        async with aiofiles.open(path / "state.json", "w") as f:
            await f.write(json.dumps(state_data))
            
        await asyncio.gather(
            save_messages_store(),
            save_files(),
            save_metadata()
        )

    async def deserialize(self, path: Union[str, Path]) -> None:
        """
        Deserialize agent state from disk.
        
        :param path: Directory path containing serialized data
        :type path: Union[str, Path]
        """
        path = Path(path)
        
        # Load core state data
        async with aiofiles.open(path / "state.json", "r") as f:
            state_data = json.loads(await f.read())
            self.session_id = state_data["session_id"]
            self.temp_dir = state_data["temp_dir"]
        
        # Load stores concurrently
        async def load_messages_store():
            # Ensure we create proper HNSWStore instance
            if not isinstance(self.message_store, HNSWStore):
                self.message_store = HNSWStore()
            await self.message_store.deserialize(path / "messages_store")
            
        async def load_files():
            # Ensure we create proper HNSWStore instance
            if not isinstance(self.file_store, HNSWStore):
                self.file_store = HNSWStore()
                
            async with aiofiles.open(path / "files.json", "r") as f:
                file_data = json.loads(await f.read())
            
            # Deserialize files and add to store
            for name, serialized_file in file_data.items():
                file = await File.deserialize(serialized_file)
                await self.file_store.add(
                    text=str(file.content),
                    metadata=file
                )
            
        async def load_metadata():
            # Ensure we create proper HNSWStore instance
            if not isinstance(self.metadata_store, HNSWStore):
                self.metadata_store = HNSWStore()
                
            async with aiofiles.open(path / "metadata.json", "r") as f:
                metadata_data = json.loads(await f.read())
            
            # Deserialize metadata and add to store
            for key, serialized_meta in metadata_data.items():
                metadata = await Metadata.deserialize(serialized_meta)
                await self.metadata_store.add(
                    text=str(metadata.data),
                    metadata=metadata
                )
        
        await asyncio.gather(
            load_messages_store(),
            load_files(),
            load_metadata()
        )

    class Config:
        """Pydantic model configuration"""
        arbitrary_types_allowed = True  # Allow custom classes like HNSWStore