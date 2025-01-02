from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel, Field, PrivateAttr
import asyncio
from functools import lru_cache
from pathlib import Path
import base64
import msgpack
import json

from utils.shared.tokenizer import decode_async, encode_async

from agents.storage.models import Chunk
from agents.storage.models import FileType

from agents.storage.models import IDFactory

from utils.serialization import get_compressor, get_decompressor


class File(BaseModel):
    """File representation with content and metadata."""
    
    id: int = Field(default_factory = lambda: IDFactory.next_id())
    data: Union[str, bytes, Any] = Field(description="Raw file data")
    path: Optional[str] = Field(default=None, description="File path")
    type: FileType = Field(default=FileType.TEXT, description="File type")
    description: str = Field(default="", description="File description")
    annotations: Dict[str, Any] = Field(default_factory=dict, description="File annotations")
    score: Optional[float] = Field(default=None, description="Score of the file")
    
    # Cache for content summary
    _content_summary: Optional[str] = PrivateAttr(default=None)


    @classmethod
    async def create(
        cls,
        data: Union[str, bytes, Any],
        path: Optional[Union[str, Path]] = None,
        type: Optional[FileType] = None,
        description: str = "",
        annotations: Dict[str, Any] = None
    ) -> 'File':
        """
        Create a new file instance.
        
        :param data: File content
        :type data: Union[str, bytes, Any]
        :param path: Optional file path
        :type path: Optional[Union[str, Path]]
        :param type: Optional file type
        :type type: Optional[FileType]
        :param description: File description
        :type description: str
        :param annotations: Additional annotations
        :type annotations: Dict[str, Any]
        :return: New File instance
        :rtype: File
        """
        if path:
            path = str(Path(path))
            if type is None:
                type = FileType.from_extension(Path(path).suffix)
        
        return cls(
            data = data,
            path = path,
            type = type or FileType.TEXT,
            description = description,
            annotations = annotations or {}
        )

    async def get_content(self) -> str:
        """
        Get summarized content asynchronously.
        Ensures summary is generated if not already cached.
        
        :return: Summarized content
        :rtype: str
        """
        if self._content_summary is None:
            self._content_summary = await self._generate_summary()
        return self._content_summary

    @property
    def content(self) -> str:
        """
        Get cached summarized content synchronously.
        WARNING: Will return None if summary not yet generated.
        Use get_content() for async access.
        
        :return: Cached summary only
        :rtype: str
        """
        return self._content_summary



    async def _generate_summary(self) -> str:
        """
        Generate content summary using file summarization utility.
        
        :return: Generated summary
        :rtype: str
        """
        from utils.summarization import summarize_file
        
        # Get file path if available
        path = Path(self.path) if self.path else None
        
        # Generate summary
        summary = await summarize_file(
            file_data = self.data,
            file_type = self.type,
            name = path.name if path else "file",
            path = path
        )
        
        # Cache the summary
        self._content_summary = summary
        return summary

    async def to_chunks(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 100
    ) -> List[Chunk]:
        """
        Split file into chunks for processing.
        Uses raw data, not the summary.
        """
     
        # Always use raw data for chunking
        raw_text = str(self.data)
        if not raw_text:
            return []
        
        # Get tokens
        tokens = await encode_async(raw_text)
        if not tokens:
            return []
       
        # Ensure valid chunk parameters
        chunk_size = max(1, min(chunk_size, len(tokens)))
        chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))
        
        # Calculate stride
        stride = chunk_size - chunk_overlap
        if stride <= 0:
            stride = 1
        
        # Create chunks
        chunks = []
        for start in range(0, len(tokens), stride):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            if chunk_tokens:  # Only add non-empty chunks
                chunks.append(chunk_tokens)
            if end >= len(tokens):
                break
            
        # Create chunk objects
        async def create_chunk(chunk_tokens):
            text = await decode_async(chunk_tokens)
            return Chunk(
                content = text,
                path = self.path,
                parent_id = self.id
            )
        
        chunk_objects = await asyncio.gather(*[
            create_chunk(chunk) for chunk in chunks
        ])
        
        return chunk_objects


    async def get_raw_content(self) -> str:
        """Get raw file content"""
        if isinstance(self.data, (str, bytes)):
            return str(self.data)
        elif hasattr(self.data, 'to_string'):  # Handle pandas/polars DataFrames
            return self.data.to_string()
        else:
            return json.dumps(self.data)

    async def serialize(self) -> str:
        """
        Serialize file to a string representation.
        
        Handles both binary and text data efficiently using msgpack and compression.
        
        :return: Serialized file string
        :rtype: str
        """
        # Create base data dictionary with basic types
        data = {
            'id': self.id,
            'path': str(self.path) if self.path else None,
            'type': self.type.value,
            'description': self.description,
            'annotations': self.annotations,
            'score': self.score
        }
        
        # Handle the raw data field
        if isinstance(self.data, (str, bytes)):
            data['data'] = {
                'type': 'raw',
                'encoding': 'bytes' if isinstance(self.data, bytes) else 'str',
                'data': self.data
            }
        else:
            # For other types, convert to string
            data['data'] = {
                'type': 'other',
                'data': str(self.data)
            }
            
        # Include cached content summary if available
        if self._content_summary:
            data['content_summary'] = self._content_summary
            
        # Use msgpack for efficient binary serialization
        packed = msgpack.packb(data, use_bin_type=True)
        
        # Compress the packed data
        compressed = get_compressor().compress(packed)
        
        # Return base64 encoded string
        return base64.b64encode(compressed).decode('ascii')


    @classmethod
    async def deserialize(cls, serialized_data: str) -> 'File':
        """
        Create File instance from serialized string.
        
        :param serialized_data: Serialized file data
        :type serialized_data: str
        :return: New File instance
        :rtype: File
        """
        # Decode base64 and decompress
        compressed = base64.b64decode(serialized_data)
        decompressed = get_decompressor().decompress(compressed)
        
        # Unpack data
        data = msgpack.unpackb(decompressed, raw=False)
        
        # Handle the data field
        if isinstance(data.get('data'), dict):
            if data['data']['type'] == 'raw':
                if data['data']['encoding'] == 'bytes':
                    data['data'] = data['data']['data']
                else:  # str
                    data['data'] = str(data['data']['data'])
            else:  # other
                data['data'] = data['data']['data']
                
        # Convert type back to enum
        data['type'] = FileType(data['type'])
        
        # Create instance
        instance = cls(**data)
        
        # Restore cached content summary if available
        if 'content_summary' in data:
            instance._content_summary = data['content_summary']
            
        return instance


    async def __aenter__(self) -> 'File':
        """Async context manager entry."""
        # Ensure content summary is generated on entry
        await self.get_content()
        return self


    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with cleanup."""
        # Don't clear content summary on exit anymore
        pass


    def __bool__(self) -> bool:
        """Check if file has content."""
        return bool(self.data)


    async def token_length(self) -> int:
        """Get the token length of the file content asynchronously."""
        tokens = await self.tokens()
        return len(tokens)

    def __hash__(self) -> int:
        """Make File hashable for caching"""
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        """Define equality for hashing"""
        if not isinstance(other, File):
            return False
        return self.id == other.id

    @lru_cache(maxsize=1)
    async def tokens(self) -> List[int]:
        """Get tokenized content"""
        return await encode_async(str(self.data))

