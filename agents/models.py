"""
Core agent models and state management.

This module provides the base models for agent configuration, state management,
and file handling with support for both synchronous and asynchronous operations.
"""

from typing import List, Dict, Any, Optional, Callable, Union, Tuple, ClassVar
import tempfile
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field
import json
import aiofiles
import asyncio
from contextlib import asynccontextmanager
import datetime
from difflib import SequenceMatcher, unified_diff
import pandas as pd
import numpy as np
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial, cached_property
import multiprocessing
import contextlib
from uuid import uuid4

from agents.messages.models import Message, File


class AgentStatus(Enum):
    """
    Enumeration of possible agent execution states.
    
    :cvar IDLE: Agent is waiting for work
    :cvar RUNNING: Agent is currently executing
    :cvar COMPLETED: Agent has finished successfully
    :cvar FAILED: Agent encountered an error
    :cvar CANCELLED: Agent execution was cancelled
    """
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Logging(Enum):
    """
    Enumeration of logging modes for agents.
    
    :cvar LANGFUSE: Use Langfuse for logging
    :cvar ENABLED: Use basic logging
    :cvar DISABLED: Disable logging
    """
    LANGFUSE = "langfuse"
    ENABLED = "enabled"
    DISABLED = "disabled"


class AgentConfig(BaseModel):
    """
    Configuration model for agents.
    
    :ivar task_prompt: Main task prompt for the agent
    :type task_prompt: Optional[str]
    :ivar prompts: Dictionary of named prompts
    :type prompts: Dict[str, str]
    :ivar attributes: Additional configuration attributes
    :type attributes: Dict[str, Any]
    :ivar description: Agent description
    :type description: Optional[str]
    :ivar logging: Logging mode
    :type logging: Logging
    """
    task_prompt: Optional[str] = None
    prompts: Dict[str, str] = Field(default_factory = dict)
    attributes: Dict[str, Any] = Field(default_factory = dict)
    description: Optional[str] = None
    logging: Logging = Logging.LANGFUSE
    
    def __repr__(self):
        return f"AgentConfig(system_prompt={self.system_prompt}, prompts={self.prompts}, attributes={self.attributes})"

    
class AgentState(BaseModel):
    """
    State management model for agents.
    
    Handles message history, context storage, file management, and metadata
    with thread-safe operations.
    
    :cvar _metadata_lock: Lock for thread-safe metadata access
    :type _metadata_lock: ClassVar[threading.RLock]
    
    :ivar input_messages: Input message store
    :type input_messages: Dict[str, Message]
    :ivar output_messages: Output message store
    :type output_messages: Dict[str, Message]
    :ivar error_messages: Error message store
    :type error_messages: Dict[str, Message]
    :ivar messages: Complete message history
    :type messages: List[Message]
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
    
    # Message stores
    input_messages: Dict[str, Message] = Field(default_factory = dict)
    output_messages: Dict[str, Message] = Field(default_factory = dict)
    error_messages: Dict[str, Message] = Field(default_factory = dict)
    messages: List[Message] = Field(default_factory = list)
    
    # Context stores
    context: Dict[str, Any] = Field(default_factory = dict)
    files: Dict[str, File] = Field(default_factory = dict)
    metadata: Dict[str, Any] = Field(default_factory = dict)
    temp_dir: Optional[str] = Field(default = None, exclude = True)
    session_id: str = Field(default = str(uuid4()), exclude = True)
    
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
            max_workers=thread_count,
            thread_name_prefix='agent_metadata_'
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
            self.temp_dir = tempfile.mkdtemp(prefix = f"agent_workspace_{self.metadata.name}_")
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
    ) -> File:
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
        self.files[filename] = file
        
        return file
    
    
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
    
    
    ######################
    # Context Management # 
    ######################
    
    def obtain_context(self, key: str = None) -> Any:
        """
        Get value(s) from the context store.
        
        :param key: Optional key to retrieve specific value
        :type key: str, optional
        :return: Context value or formatted string of all context
        :rtype: Any
        """
        if key is None:
            context_str = ""
            for key, value in self.context.items():
                context_str += f"{key}: {value}\n"
            return context_str
        
        else:
            return self.context.get(key, None)
    
    
    def update_context(self, **kwargs) -> None:
        """
        Update context store with new key-value pairs.
        
        :param kwargs: Key-value pairs to update
        """
        self.context.update(kwargs)

    
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
    
    @contextlib.asynccontextmanager
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
        if isinstance(value, pd.DataFrame) and isinstance(existing, pd.DataFrame):
            if set(value.columns) == set(existing.columns):
                merged_df = await self._run_in_executor(
                    pd.concat, [existing, value], 
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


    @contextlib.asynccontextmanager 
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


