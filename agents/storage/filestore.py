from typing import List, Dict, Any, Optional, Type, TYPE_CHECKING, Tuple
from pydantic import BaseModel, Field, PrivateAttr
import asyncio
from pathlib import Path
import tempfile
import shutil
import json
import aiofiles
import os
import time
from difflib import unified_diff
from functools import cached_property
from typing import Set
import numpy as np

from agents.vectorstore.models import BaseVectorStore
from agents.vectorstore.default.store import HNSWStore
from agents.vectorstore.truncation import truncate_items

from agents.config.models import ContextConfig

from agents.storage.models import Chunk
from agents.storage.file import File
from agents.storage.models import FileType

# Only import RetrievalType enum
from agents.config.models import RetrievalType

if TYPE_CHECKING:
    from agents.state.models import AgentState

from llm import BaseEmbeddingFunction

from utils.shared.tokenizer import encode_async, decode_async


class FileStore(BaseModel):
    """
    Manages file storage and vector embeddings for search functionality.
    
    Handles both whole file storage and chunked content with vector embeddings
    for efficient similarity search.
    
    :param embedding_function: Function to generate embeddings
    :type embedding_function: BaseEmbeddingFunction
    :param temp_dir: Temporary directory for file operations
    :type temp_dir: Optional[Path]
    :param lazy_indexing: Whether to index chunks lazily
    :type lazy_indexing: bool
    """
    
    embedding_function: BaseEmbeddingFunction = Field(
        default = None,
        description = "Function used to generate embeddings"
    )
    _temp_dir: Optional[Path] = PrivateAttr(default = None)
    
    @cached_property
    def temp_dir(self) -> Path:
        """
        Lazily create and return temporary directory.
        Only created when first accessed.
        """
        if not self._temp_dir:
            self._temp_dir = Path(tempfile.mkdtemp(prefix = "filestore_"))
        return self._temp_dir
    
    lazy_indexing: bool = Field(
        default = True,
        description = "Whether to index chunks lazily"
    )
    
    # Store class to use (will be instantiated in __init__)
    vector_store_class: Type[BaseVectorStore] = Field(
        default = HNSWStore,
        description = "Vector store class to use for storage"
    )
    
    # Stores - these will be initialized in __init__
    file_handler: Optional[BaseVectorStore] = Field(
        default = None,
        description="Vector store for whole files"
    )
    chunk_store: Optional[BaseVectorStore] = Field(
        default = None,
        description="Vector store for file chunks"
    )
    
    # Mappings
    chunk_to_file_map: Dict[str, str] = Field(
        default_factory = dict,
        description = "Maps chunk IDs to their parent file IDs"
    )
    file_chunks_map: Dict[str, List[str]] = Field(
        default_factory = dict,
        description = "Maps file IDs to their chunk IDs"
    )

    _state: Optional["AgentState"] = PrivateAttr(default = None)

    # Tracking sets for lazy indexing
    _pending_index: Set[str] = PrivateAttr(default_factory = set)  
    _index_lock: asyncio.Lock = PrivateAttr(default_factory = asyncio.Lock)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.temp_dir:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="filestore_"))
        
        # Initialize stores with embedding function using the specified class
        self.file_handler = self.vector_store_class(
            embedding_function = self.embedding_function
        )
        self.chunk_store = self.vector_store_class(
            embedding_function = self.embedding_function
        )

    class Config:
        arbitrary_types_allowed = True


    async def add_file(
        self,
        file: File
    ) -> None:
        """
        Add a file to both stores.
        
        :param file: File to add
        :type file: File
        """
        # Generate content summary before adding to store
        await file.get_content()
   
        # Update file path to use store's temp directory if needed
        if file.path:
            original_path = Path(file.path)
            file.path = str(self.temp_dir / original_path.name)
        
        # Add to store with summary as searchable text
        await self.file_handler.add(
            text = file.content,  
            metadata = file
        )
     
        if not self.lazy_indexing:
            await self._index_file(file)
        else:
            self._pending_index.add(file.id)
        

    async def remove_file(self, file_id: str) -> None:
        """
        Remove file and its chunks from stores.
        
        :param file_id: ID of file to remove
        :type file_id: str
        """
        # Remove from file store
        await self.file_handler.delete(file_id)
        
        # Remove chunks if they exist
        if file_id in self.file_chunks_map:
            chunk_ids = self.file_chunks_map[file_id]
            await self.chunk_store.delete(chunk_ids)
            
            # Clean up mappings
            for chunk_id in chunk_ids:
                del self.chunk_to_file_map[chunk_id]
            del self.file_chunks_map[file_id]
            

    async def get_file(self, file_id: str) -> Optional[File]:
        """
        Retrieve a file by ID.
        
        :param file_id: ID of file to retrieve
        :type file_id: str
        :return: File object if found, None otherwise
        :rtype: Optional[File]
        """
        result = await self.file_handler.get(file_id)
        if isinstance(result, File):
            return result
        
        return None


    async def search_files(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        retrieval_strategy: Optional[RetrievalType] = None,
        config: Optional[ContextConfig] = None
    ) -> List[File]:
        """
        Search for relevant files.
        
        :param query: Search query
        :type query: str
        :param k: Maximum number of files to return
        :type k: Optional[int]
        :param filter: Optional filter criteria
        :type filter: Optional[Dict[str, Any]]
        :param retrieval_strategy: Strategy for retrieving files
        :type retrieval_strategy: Optional[RetrievalType]
        :param config: Context configuration
        :type config: Optional[ContextConfig]
        :return: List of relevant files
        :rtype: List[File]
        """
        if not config and self._state:
            config = self._state.context_config
        
        # Use config's strategy if none provided
        if retrieval_strategy is None and config:
            retrieval_strategy = config.retrieval_strategy
        elif retrieval_strategy is None:
            retrieval_strategy = RetrievalType.SIMILARITY

        # Set k from config if not provided
        if k is None and config:
            k = config.item_count
        elif k is None:
            k = 10

        if retrieval_strategy == RetrievalType.CHRONOLOGICAL:
            # Get most recent files
            files = await self.list_files()
            files.sort(key=lambda x: x.date, reverse=True)
            return files[:k]
        
        elif retrieval_strategy == RetrievalType.SIMILARITY:
            # Get semantically similar files
            return await self.file_handler.search_relevant(
                query=query,
                k=k,
                filter=filter
            )
        
        elif retrieval_strategy == RetrievalType.ALL:
            # Get all files
            return await self.list_files()
        
        else:  # NONE
            return []


    async def search_chunks(
        self,
        query: str,
        
        k: Optional[int] = None,
        file_id: Optional[str] = None, # Allows for chunks from a single file to be retrieved
        filter: Optional[Dict[str, Any]] = None,
        retrieval_strategy: Optional[RetrievalType] = RetrievalType.SIMILARITY,
        config: Optional[ContextConfig] = None,
        truncate: bool = True
    ) -> List[Chunk]:
       
        """
        Search for relevant chunks.
        
        :param query: Search query
        :type query: str
        :param k: Maximum number of chunks
        :type k: Optional[int] 
        :param filter: Optional filter criteria
        :type filter: Optional[Dict[str, Any]]
        :param retrieval_strategy: Strategy for retrieving chunks
        :type retrieval_strategy: Optional[RetrievalType]
        :param config: Context configuration
        :type config: Optional[ContextConfig]
        :param truncate: Whether to truncate chunks
        :type truncate: bool
        :return: List of relevant chunks
        :rtype: List[Chunk]
        """
        if self.lazy_indexing:
            async with self._index_lock:
                files_to_index = set()
                if file_id:
                    if file_id in self._pending_index:
                        files_to_index = set([file_id])
                else:
                    files_to_index = self._pending_index

                for file_idx in files_to_index:
                    file = await self.get_file(file_idx)
                    if file_idx in self.file_chunks_map:
                        chunk_ids = self.file_chunks_map[file_idx]
                        for chunk_id in chunk_ids:
                            await self.chunk_store.delete(chunk_id)
                            self.chunk_to_file_map.pop(chunk_id, None)
                        self.file_chunks_map.pop(file_idx)
                    
                    # Index file
                    await self._index_file(file)
            
            self._pending_index -= files_to_index
        
        if file_id:
            if filter:
                filter.update({
                    "parent_id": file_id
                })
            else:
                filter = {
                    "parent_id": file_id
                }

        # Now perform the search
        if not config and self._state:
            config = self._state.context_config
        
        # Use config's strategy if none provided
        if config:
            retrieval_strategy = config.retrieval_strategy
        if retrieval_strategy not in [RetrievalType.SIMILARITY, RetrievalType.CHRONOLOGICAL]:
            retrieval_strategy = RetrievalType.SIMILARITY
        
        # Set k from config if not provided
        if k is None and config:
            k = config.item_count        
        
        if retrieval_strategy == RetrievalType.CHRONOLOGICAL: # Get chunks in chronological order            
            chunks = []
            if not file_id:
                files = await self.list_files()
                files.sort(key = lambda x: x.date, reverse = True)
                
                for file in files[:k]:
                    if file.id in self.file_chunks_map:
                        chunk_ids = self.file_chunks_map[file.id]
                        for chunk_id in chunk_ids:
                            chunk = await self.chunk_store.get(chunk_id)
                            if chunk:
                                chunks.append(chunk)
                                if len(chunks) >= k:
                                    break
                        if len(chunks) >= k:
                            break
            
            else:
                chunk_ids = self.file_chunks_map[file_id]
                for chunk_id in chunk_ids:
                    chunk = await self.chunk_store.get(chunk_id)
                    if chunk:
                        chunks.append(chunk)
                        if len(chunks) >= k:
                            break
                        
        elif retrieval_strategy == RetrievalType.SIMILARITY:  # Get semantically similar chunks 
            if not file_id:
                chunks = await self.chunk_store.search_relevant(
                    query = query,
                    k = k,
                    filter = filter
                )
            else:
                store_length = await self.chunk_store.length
                chunks = await self.chunk_store.search_relevant(
                    query = query,
                    k = store_length,
                    filter = filter
                )
                chunks = chunks[:k]
            
        # Apply truncation if requested
        if truncate:
            if not config:
                raise ValueError("Context configuration is required for truncation!")
            chunks = await truncate_items(
                items = chunks,
                context_config = config
            )
       
        return chunks


    async def write_to_directory(self, directory: Path) -> Dict[str, Path]:
        """
        Write files from store to disk.
        
        :param directory: Directory to write files to
        :type directory: Path
        :return: Map of file IDs to written paths
        :rtype: Dict[str, Path]
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written_files = {}
        
        for file_id, file in self.file_handler.data.items():
            if isinstance(file, File):
                rel_path = Path(file.path).name if file.path else f"{file_id}.txt"
                full_path = directory / rel_path
                
                async with aiofiles.open(full_path, 'w') as f:
                    await f.write(str(file.data))
                    
                written_files[file_id] = full_path
                
        return written_files


    async def read_from_directory(self, directory: Path) -> List[File]:
        """
        Load files from disk into store.
        
        :param directory: Directory to read files from
        :type directory: Path
        :return: List of loaded files
        :rtype: List[File]
        """
        directory = Path(directory)
        loaded_files = []
        
        async def load_file(path: Path):
            async with aiofiles.open(path, 'r') as f:
                content = await f.read()
                
            file = File(
                data = content,
                path = str(path),
                type = FileType.from_extension(path.suffix)
            )
            await self.add_file(file)
            loaded_files.append(file)
            
        tasks = []
        for path in directory.rglob('*'):
            if path.is_file():
                tasks.append(load_file(path))
                
        await asyncio.gather(*tasks)
        return loaded_files


    async def serialize(self, path: Path) -> None:
        """
        Serialize the file store to disk.
        
        Saves vector stores, mappings, and configuration in parallel.
        
        :param path: Directory path to save serialized data
        :type path: Path
        """
        path.mkdir(parents=True, exist_ok=True)
        
        # Save stores and mappings concurrently
        async def save_stores():
            # Save both vector stores
            await asyncio.gather(
                self.file_handler.serialize(path / "file_handler"),
                self.chunk_store.serialize(path / "chunk_store")
            )
        
        async def save_mappings():
            # Save all mappings and configuration
            mappings = {
                "chunk_to_file": self.chunk_to_file_map,
                "file_chunks": self.file_chunks_map,
                "config": {
                    "lazy_indexing": self.lazy_indexing,
                    "embedding_function": await self.embedding_function.serialize(),
                    "pending_index": list(self._pending_index)
                }
            }
            
            # Also save the actual files data
            files_data = {}
            for file in self.file_handler.data.values():
                if isinstance(file, File):
                    serialized = await file.serialize()
                    files_data[file.id] = serialized
            
            mappings["files"] = files_data
            
            # Save mappings and config as JSON (text)
            async with aiofiles.open(path / "mappings.json", "w") as f:
                await f.write(json.dumps(mappings))
            
        await asyncio.gather(save_stores(), save_mappings())


    @classmethod
    async def deserialize(cls, path: Path) -> 'FileStore':
        """
        Deserialize the file store from disk.
        
        :param path: Directory path containing serialized data
        :type path: Path
        :return: Deserialized FileStore instance
        :rtype: FileStore
        """
        # Load mappings first to get configuration and files
        async with aiofiles.open(path / "mappings.json", "r") as f:
            content = await f.read()
            mappings = json.loads(content)
            config = mappings["config"]
            
            # Create new instance
            instance = cls(embedding_function = await BaseEmbeddingFunction.deserialize(
                config["embedding_function"]
            ))
                
            instance.chunk_to_file_map = mappings["chunk_to_file"]
            instance.file_chunks_map = mappings["file_chunks"]
            
            # Restore configuration
            instance.lazy_indexing = config["lazy_indexing"]
            
            # Restore files
            files_data = mappings.get("files", {})
            for file_id, serialized in files_data.items():
                file = await File.deserialize(serialized)
                await instance.add_file(file)
        
        # Load vector stores in parallel
        await asyncio.gather(
            instance.file_handler.deserialize(path / "file_handler"),
            instance.chunk_store.deserialize(path / "chunk_store")
        )
        
        # Restore only pending_index state
        instance._pending_index = set(config.get("pending_index", []))
        
        return instance


    async def _index_file(
        self,
        file: File,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> None:
        """
        Index a file by chunking and adding to chunk store.
        
        :param file: File to index
        :type file: File
        :param chunk_size: Optional chunk size override
        :type chunk_size: Optional[int]
        :param chunk_overlap: Optional chunk overlap override
        :type chunk_overlap: Optional[int]
        """
        # Skip chunking for binary files
        if file.type == FileType.BINARY:
            return []

        # Get default values from state if available
        if self._state and not chunk_size:
            chunk_size = self._state.context_config.chunk_size
        if self._state and not chunk_overlap:
            chunk_overlap = self._state.context_config.chunk_overlap
        
        # Use reasonable defaults if still not set
        chunk_size = chunk_size or 50  # Default to 50 tokens
        chunk_overlap = chunk_overlap or 10  # Default to 10 token overlap
        
        # Get file content and tokens
        content = str(file.data)
        tokens = await encode_async(content)
        
        # Calculate stride
        stride = chunk_size - chunk_overlap
        if stride <= 0:
            stride = 1
        
        # Create chunks with overlap
        chunk_texts = []
        chunks = []
        
        # First, create all chunks in parallel
        async def create_chunk(start: int) -> Tuple[str, Chunk]:
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            if not chunk_tokens:
                return None, None
            
            # Convert back to text
            chunk_text = await decode_async(chunk_tokens)

            # Create chunk object
            chunk = Chunk(
                content = chunk_text,
                path = file.path,
                parent_id = file.id,
                file_metadata = {
                    "type": file.type,
                    "description": file.description,
                    **file.annotations
                }
            )
            
            return chunk_text, chunk

        # Create all chunks in parallel
        chunk_tasks = [
            create_chunk(start) 
            for start in range(0, len(tokens), stride)
        ]
        chunk_results = await asyncio.gather(*chunk_tasks)
        
        chunk_texts = []
        chunks = []
        for text, chunk in chunk_results:
            if text and chunk:
                chunk_texts.append(text)
                chunks.append(chunk)
        
        if not chunks:
            return []

        embeddings = await self.embedding_function(chunk_texts)
       
        chunk_ids = await self.chunk_store.add(
            embeddings = embeddings,
            metadata = chunks
        )
        
        # Update mappings
        for chunk_id in chunk_ids:
            self.chunk_to_file_map[chunk_id] = file.id
            if file.id not in self.file_chunks_map:
                self.file_chunks_map[file.id] = []
            self.file_chunks_map[file.id].append(chunk_id)
        
        return chunks


    async def _create_chunks(self, file: File) -> List[Chunk]:
        """
        Create chunks from file content.
        
        :param file: File to chunk
        :type file: File
        :return: List of chunks
        :rtype: List[Chunk]
        """
        # Skip chunking for binary files
        if file.type.is_binary():
            return []

        content = str(file.data)
        config = ContextConfig()  # Default config for chunking

        tokens = await encode_async(content)
        chunk_size = config.chunk_size
        chunk_overlap = config.chunk_overlap
        stride = chunk_size - chunk_overlap
        
        num_chunks = max(1, (len(tokens) - chunk_overlap) // stride)

        async def create_chunk(chunk_idx: int) -> Chunk:
            start_idx = chunk_idx * stride
            end_idx = min(start_idx + chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = await decode_async(chunk_tokens)
            
            return Chunk(
                content = chunk_text,
                path = file.path,
                parent_id = file.id
            )

        chunks = await asyncio.gather(*[
            create_chunk(i) for i in range(num_chunks)
        ])
        
        return chunks


    async def edit_file(
        self,
        file_id: str,
        new_content: str,
        edit_description: str = "",
        create_new: bool = False
    ) -> File:
        """
        Edit an existing file or create a new one with edits.
        
        :param file_id: ID of file to edit
        :param new_content: New/edited content
        :param edit_description: Description of the edit
        :param create_new: Whether to create new file or edit in place
        :return: Edited or new file
        """
        # Get existing file
        original_file = await self.get_file(file_id)
        if not original_file:
            raise ValueError(f"File {file_id} not found")

        # Generate diff for edit description if none provided
        if not edit_description:
            diff = list(unified_diff(
                str(original_file.data).splitlines(),
                new_content.splitlines(),
                fromfile='original',
                tofile='edited'
            ))
            edit_description = "\n".join(diff)

        if create_new:
            # Create new file with edits
            edited_file = File(
                data = new_content,
                path = original_file.path,
                type = original_file.type,
                description = f"{original_file.description}\nEdit: {edit_description}",
                annotations = {
                    **original_file.annotations,
                    "edited_from": file_id,
                    "edit_description": edit_description,
                    "edit_time": time.time()
                }
            )
            await self.add_file(edited_file)
            return edited_file
        else:
            # Remove old chunks
            if file_id in self.file_chunks_map:
                chunk_ids = self.file_chunks_map[file_id]
                for chunk_id in chunk_ids:
                    await self.chunk_store.delete(chunk_id)
                    self.chunk_to_file_map.pop(chunk_id, None)
                self.file_chunks_map.pop(file_id)

            # Update file content and metadata
            original_file.data = new_content
            original_file.description = f"{original_file.description}\nEdit: {edit_description}"
            if "edit_history" not in original_file.annotations:
                original_file.annotations["edit_history"] = []
            original_file.annotations["edit_history"].append({
                "description": edit_description,
                "time": time.time()
            })
            
            # Clear cached content summary
            original_file._content_summary = None
            
            # Update in store
            await self.file_handler.update(
                id = file_id,
                text = await original_file.get_content(),
                metadata = original_file
            )
            
            # Queue for indexing if lazy indexing is enabled
            if self.lazy_indexing:
                self._pending_index.add(file_id)
            else:
                await self._index_file(original_file)
            
            return original_file


    async def merge_files(
        self,
        file_ids: List[str],
        output_path: str,
        separator: str = "\n\n",
        delete_sources: bool = True
    ) -> File:
        """Merge multiple files into one."""
        # Collect file contents and descriptions
        contents = []
        descriptions = []
        
        for fid in file_ids:
            file = await self.get_file(fid)
            if file:
                contents.append(str(file.data))
                descriptions.append(
                    f"`{file.path}` → {file.type} file"
                    + (f" at {file.path}" if file.path else "")
                    + (f"\n\n{file.description}" if file.description else "")
                )

        # Create merged path
        merged_path = self.temp_dir / output_path
        
        # Create merged file
        merged_file = File(
            data = separator.join(contents),
            path = str(merged_path),
            type = FileType.TEXT,
            description = "\n".join(descriptions),
            annotations = {
                "source_files": file_ids,
                "merge_date": time.time(),
                "source_summaries": descriptions
            }
        )
        
        await self.add_file(merged_file)
        
        if delete_sources:
            file_ids = [fid for fid in file_ids if fid != merged_file.id]
            for fid in file_ids:
                await self.remove_file(fid)
   
        return merged_file


    async def list_files(self) -> List[File]:
        """
        List all files in the store.
        
        :return: List of all files
        :rtype: List[File]
        """
        files = []
       
        for item in self.file_handler.data.values():
            if isinstance(item, (File, str)):  # Handle both objects and serialized strings
                if isinstance(item, str):
                    print(f"Warning: Found serialized string instead of File object: {item[:100]}...")
                else:
                    files.append(item)
        
        return files


    async def cleanup(self) -> None:
        """
        Clean up temporary files and resources.
        
        Removes temporary directory and cleans up vector stores.
        """
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None

    @property
    async def length(self) -> int:
        """
        Get the number of files in the store.
        
        :return: Number of files
        :rtype: int
        """
        files = await self.list_files()
        return len(files)

    async def set_state(self, state: "AgentState") -> None:
        """Set reference to parent state"""
        self._state = state

    @property
    def state(self) -> Optional["AgentState"]:
        """Get reference to parent state"""
        return self._state