import os
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any, List, Optional, Union
from functools import lru_cache
import sys
import asyncio
import pandas as pl
import numpy as np
import torch
import base64
import msgpack
import io
import pyarrow as pa
from enum import Enum, auto

from utils.shared.tokenizer import encode_async
from utils.shared.tokenizer import slice_text, SliceType
from utils.summarization import MetadataType
from utils.serialization import get_compressor, get_decompressor

# Import SerializationType enum from serialization.py
class SerializationType(Enum):
    """Supported serialization types"""
    ARROW = auto()
    PLOTLY = auto()
    MSGPACK = auto()
    PICKLE = auto()
    NUMPY = auto()
    IMAGE = auto()
    PARQUET = auto()
    JSON = auto()
    NONE = auto()


class Metadata(BaseModel):
    """
    Model for metadata attachments in messages.
    
    :ivar id: Unique identifier
    :type id: str
    :ivar date: Creation timestamp
    :type date: datetime
    :ivar description: Optional description
    :type description: str
    :ivar agent_name: Name of agent that created metadata
    :type agent_name: str
    :ivar name: Optional metadata name
    :type name: str
    :ivar type: Metadata type
    :type type: MetadataType
    """
    id: Union[int, str] = Field(default_factory = lambda: int.from_bytes(os.urandom(3), 'big') % 1_000_000)
    date: datetime = Field(default_factory = datetime.now)
    description: str = Field(default = None)
    agent_name: str = Field(default = "base")
    name: str = Field(default = None)
    type: MetadataType = Field(default = MetadataType.UNKNOWN)
    
    # For vector-based retrieval
    score: int = Field(default = 0, description = "Score of the file")

    # Store data reference directly
    stored_data: Any = Field(default=None, alias='data')
    content_cache: Optional[str] = Field(default=None, exclude=True)

    class Config:
        # Allow arbitrary types to be stored by reference
        arbitrary_types_allowed = True
        # Prevent copying of stored data
        copy_on_model_validation = False

    def __init__(self, **data):
        super().__init__(**data)
        # Store reference to data without copying
        if 'stored_data' in data:
            self.stored_data = data['stored_data']
        elif 'data' in data:
            self.stored_data = data['data']

    @property
    def data(self) -> Any:
        """Get metadata content reference"""
        return self.stored_data

    @data.setter 
    def data(self, value: Any) -> None:
        """Set metadata content reference and clear caches"""
        # Special handling for different types
        if isinstance(value, pl.DataFrame):
            # For DataFrames, compare memory addresses
            if id(self.stored_data) != id(value):
                self.clear_caches()
                self.stored_data = value
        elif isinstance(value, np.ndarray):
            # For NumPy arrays, compare memory addresses
            if id(self.stored_data) != id(value):
                self.clear_caches()
                self.stored_data = value
        elif isinstance(value, torch.Tensor):
            # For PyTorch tensors, compare memory addresses
            if id(self.stored_data) != id(value):
                self.clear_caches()
                self.stored_data = value
        else:
            # For regular objects that support equality comparison
            if self.stored_data is not value:
                self.clear_caches()
                self.stored_data = value

    @property
    async def content(self) -> str:
        """
        Get summarized string representation of metadata content.
        Uses summary handler to generate concise description.
        
        :return: Summary string of metadata content
        :rtype: str
        """
        if self.content_cache is not None:
            return self.content_cache
            
        name = self.name or f"metadata_{self.id}"
        handler, inferred_type = await MetadataType.get_handler(self.data)
        
        # Update type if unknown
        if self.type == MetadataType.UNKNOWN:
            self.type = inferred_type
        
        # Run synchronous handler in executor to avoid blocking
        loop = asyncio.get_running_loop()
        summary = await loop.run_in_executor(None, handler, self.data, name)
        
        # Format: name -> type: description (user description)
        content = f"`{name}` -> {summary}"
        if self.description:
            content = f"{content} (Description: {self.description})"
            
        self.content_cache = content
        return content


    @lru_cache(maxsize = 1)
    async def tokens(self) -> List[int]:
        """
        Get tokenized representation of metadata summary.
        
        :return: List of token IDs
        :rtype: List[int]
        """
        summary = str(self.data)
        return await encode_async(summary)


    async def clear_caches(self) -> None:
        """Clear all caches."""
        self.tokens.cache_clear()
        self.content_cache = None


    async def __len__(self) -> int:
        """
        Get token count of metadata summary content.
        
        :return: Number of tokens
        :rtype: int
        """
        tokens = await self.tokens()
        return len(tokens)


    async def truncate(self, 
        max_tokens: int,
        slice_type: SliceType = SliceType.END
    ) -> str:

        truncated = await slice_text(
            text = await self.content,
            max_tokens = max_tokens,
            slice_type = slice_type
        )
        return truncated


    async def serialize(self) -> str:
        """
        Serialize metadata to a string representation.
        
        :return: Serialized metadata string
        :rtype: str
        """
        # Create base data dictionary with basic types
        data = {
            'id': self.id,
            'date': self.date.isoformat(),
            'description': self.description,
            'agent_name': self.agent_name,
            'name': self.name,
            'type': self.type.value,
            'score': self.score
        }

        # Handle the data field separately since it may contain non-JSON serializable types
        if self.data is not None:
            if isinstance(self.data, pl.DataFrame):
                # Most efficient serialization for Polars DataFrames:
                # 1. Convert to Arrow table
                # 2. Use IPC streaming format with compression
                # Source: https://pola-rs.github.io/polars/py-polars/html/reference/io.html
                arrow_table = self.data.to_arrow()
                sink = pa.BufferOutputStream()
                with pa.ipc.new_stream(
                    sink,
                    arrow_table.schema,
                    options=pa.ipc.IpcWriteOptions(compression='zstd')
                ) as writer:
                    writer.write_table(arrow_table)
                data['data'] = {
                    'type': 'polars_ipc',
                    'schema': arrow_table.schema.serialize().to_pybytes(),
                    'data': sink.getvalue().to_pybytes()
                }
                # Use msgpack for the wrapper since it handles bytes efficiently
                packed = msgpack.packb(data, use_bin_type=True)
                compressed = get_compressor().compress(packed)
                return base64.b64encode(compressed).decode('ascii')
            elif isinstance(self.data, np.ndarray):
                # Use numpy's efficient binary format
                buffer = io.BytesIO()
                np.save(buffer, self.data, allow_pickle=False)
                data['data'] = {
                    'type': 'numpy',
                    'data': buffer.getvalue()
                }
                # Use msgpack for binary data
                packed = msgpack.packb(data, use_bin_type=True)
                compressed = get_compressor().compress(packed)
                return base64.b64encode(compressed).decode('ascii')
            elif isinstance(self.data, torch.Tensor):
                # Convert to numpy and use its efficient format
                tensor_data = self.data.detach().cpu().numpy()
                buffer = io.BytesIO()
                np.save(buffer, tensor_data, allow_pickle=False)
                data['data'] = {
                    'type': 'tensor',
                    'data': buffer.getvalue()
                }
                packed = msgpack.packb(data, use_bin_type=True)
                compressed = get_compressor().compress(packed)
                return base64.b64encode(compressed).decode('ascii')
            else:
                # For other types, use msgpack directly
                packed = msgpack.packb(data, use_bin_type=True)
                compressed = get_compressor().compress(packed)
                return base64.b64encode(compressed).decode('ascii')

        # If no data field, use msgpack for consistency
        packed = msgpack.packb(data, use_bin_type=True)
        compressed = get_compressor().compress(packed)
        return base64.b64encode(compressed).decode('ascii')


    @classmethod
    async def deserialize(cls, serialized_data: str) -> 'Metadata':
        """
        Create Metadata instance from serialized string.
        
        :param serialized_data: Serialized metadata data
        :type serialized_data: str
        :return: New Metadata instance
        :rtype: Metadata
        """
        # Decompress and unpack base data
        compressed = base64.b64decode(serialized_data)
        decompressed = get_decompressor().decompress(compressed)
        data = msgpack.unpackb(decompressed, raw=False)
        
        # Handle special data types
        if isinstance(data.get('data'), dict) and 'type' in data['data']:
            if data['data']['type'] == 'polars_ipc':
                # Reconstruct DataFrame from Arrow IPC stream
                schema = pa.ipc.read_schema(pa.py_buffer(data['data']['schema']))
                reader = pa.ipc.open_stream(
                    pa.py_buffer(data['data']['data']),
                    schema=schema
                )
                data['data'] = pl.from_arrow(reader.read_all())
            elif data['data']['type'] == 'numpy':
                buffer = io.BytesIO(data['data']['data'])
                data['data'] = np.load(buffer, allow_pickle=False)
            elif data['data']['type'] == 'tensor':
                buffer = io.BytesIO(data['data']['data'])
                arr = np.load(buffer, allow_pickle=False)
                data['data'] = torch.from_numpy(arr)
        
        # Convert basic types
        data['type'] = MetadataType(data['type'])
        data['date'] = datetime.fromisoformat(data['date'])
        
        return cls(**data)

    
    async def __aenter__(self) -> 'Metadata':
        """Async context manager entry."""
        return self


    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with cleanup."""
        await self.clear_caches()
        if hasattr(self, 'raw_data'):
            del self.raw_data
            

    @property
    async def size(self) -> int:
        """
        Get approximate memory size of metadata.
        
        :return: Size in bytes
        :rtype: int
        """
        return sys.getsizeof(self.data)

    @property
    async def type_name(self) -> str:
        """
        Get human-readable type name.
        
        :return: Type name string
        :rtype: str
        """
        return self.type.value.title()
            
            
    def __bool__(self) -> bool:
        """
        Verifies existance of object
        """
        return bool(self.data)

    async def token_length(self) -> int:
        """Get the token length asynchronously"""
        tokens = await self.tokens()
        return len(tokens)