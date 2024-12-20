"""
Serialization utilities for handling various data types and formats.

This module provides functionality for serializing and deserializing different data types
including pandas DataFrames, numpy arrays, Plotly figures, PIL Images, and more. It supports
both synchronous and asynchronous operations with thread and process pool executors.
"""

from enum import Enum, auto
from functools import lru_cache
from typing import Any, Dict, Set, Type, List, Tuple
import base64
import orjson  
import msgpack
import pickle
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import pandas as pd
import pyarrow as pa
import plotly.graph_objects as go
import zstandard as zstd 
from PIL import Image
import io
import os
import threading
import asyncio
import contextlib


logger = logging.getLogger(__name__)

# Thread-safe compressor instances
_compression_lock = threading.Lock()  # Lock for thread-safe compressor access
_compressor = None  # Global ZStandard compressor instance
_decompressor = None  # Global ZStandard decompressor instance


def get_compressor():
    """
    Get or create a thread-safe ZStandard compressor instance.

    :return: A compressor instance with compression level 3
    :rtype: zstd.ZstdCompressor
    """
    global _compressor
    with _compression_lock:
        if _compressor is None:
            _compressor = zstd.ZstdCompressor(level=3)
        return _compressor


def get_decompressor():
    """
    Get or create a thread-safe ZStandard decompressor instance.

    :return: A decompressor instance
    :rtype: zstd.ZstdDecompressor
    """
    global _decompressor
    with _compression_lock:
        if _decompressor is None:
            _decompressor = zstd.ZstdDecompressor()
        return _decompressor


class SerializationType(Enum):
    """
    Enumeration of supported serialization types.
    
    :cvar ARROW: Apache Arrow serialization for pandas DataFrames
    :cvar PLOTLY: Plotly figure serialization
    :cvar MSGPACK: MessagePack serialization for basic Python types
    :cvar PICKLE: Python pickle serialization (fallback)
    :cvar NUMPY: NumPy array serialization
    :cvar IMAGE: PIL Image serialization
    :cvar PARQUET: Apache Parquet serialization
    :cvar JSON: JSON serialization
    :cvar NONE: No serialization needed
    """
    ARROW = auto()
    PLOTLY = auto()
    MSGPACK = auto()
    PICKLE = auto()
    NUMPY = auto()
    IMAGE = auto()
    PARQUET = auto()
    JSON = auto()
    NONE = auto()


JSON_COMPATIBLE_TYPES = frozenset([type(None), bool, int, float, str])
"""Set of types that are JSON-compatible and don't need serialization"""

TYPE_SERIALIZATION_MAP = {
    pd.DataFrame: SerializationType.ARROW,
    go.Figure: SerializationType.PLOTLY,
    dict: SerializationType.MSGPACK,
    list: SerializationType.MSGPACK,
    tuple: SerializationType.MSGPACK,
    np.ndarray: SerializationType.NUMPY,
    Image.Image: SerializationType.IMAGE,
    bytes: SerializationType.MSGPACK,
    set: SerializationType.JSON,
    frozenset: SerializationType.JSON
}
"""Mapping of Python types to their corresponding serialization methods"""


# Thread-local storage for executors
class ExecutorManager:
    """
    Singleton manager for thread and process pool executors.
    
    This class manages thread-local and process-wide executor pools for
    parallel serialization operations.
    
    :cvar _instance: Singleton instance
    :type _instance: ExecutorManager
    :cvar _lock: Lock for thread-safe singleton creation
    :type _lock: threading.Lock
    :ivar _thread_executor: Thread pool executor instance
    :type _thread_executor: ThreadPoolExecutor
    :ivar _process_executor: Process pool executor instance
    :type _process_executor: ProcessPoolExecutor
    :ivar _thread_local: Thread-local storage
    :type _thread_local: threading.local
    """
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        """
        Initialize a new ExecutorManager instance.
        
        :ivar _thread_executor: Thread pool executor instance
        :type _thread_executor: ThreadPoolExecutor
        :ivar _process_executor: Process pool executor instance
        :type _process_executor: ProcessPoolExecutor
        :ivar _thread_local: Thread-local storage
        :type _thread_local: threading.local
        """
        self._thread_executor = None
        self._process_executor = None
        self._thread_local = threading.local()
    
    @classmethod
    def get_instance(cls):
        """
        Get or create the singleton instance of ExecutorManager.
        
        :return: The singleton ExecutorManager instance
        :rtype: ExecutorManager
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def get_thread_pool(self):
        """
        Get or create a thread pool executor for the current thread.
        
        :return: Thread pool executor instance
        :rtype: ThreadPoolExecutor
        """
        if not hasattr(self._thread_local, 'thread_pool'):
            self._thread_local.thread_pool = ThreadPoolExecutor(
                max_workers=min(32, (os.cpu_count() or 1) + 4),
                thread_name_prefix='serializer'
            )
        return self._thread_local.thread_pool
    
    def get_process_pool(self):
        """
        Get or create a process pool executor.
        
        :return: Process pool executor instance
        :rtype: ProcessPoolExecutor
        """
        if self._process_executor is None:
            with self._lock:
                if self._process_executor is None:
                    self._process_executor = ProcessPoolExecutor(
                        max_workers=max(1, os.cpu_count() - 1)
                    )
        return self._process_executor
    
    def cleanup(self):
        """
        Clean up thread and process pool executors.
        
        Shuts down all executor pools and removes thread-local storage.
        """
        if hasattr(self._thread_local, 'thread_pool'):
            self._thread_local.thread_pool.shutdown(wait=False)
            delattr(self._thread_local, 'thread_pool')
        
        if self._process_executor:
            with self._lock:
                if self._process_executor:
                    self._process_executor.shutdown(wait=False)
                    self._process_executor = None


@lru_cache(maxsize = 128)
def _get_type_category(cls: Type) -> SerializationType:
    """
    Get the serialization type for a given class.

    :param cls: The class type to get serialization type for
    :type cls: Type
    :return: The appropriate serialization type enum value
    :rtype: SerializationType
    """
    return TYPE_SERIALIZATION_MAP.get(cls, SerializationType.PICKLE)


async def _needs_serialization(item: Any, _seen: Set = None) -> bool:
    """
    Check if an item needs serialization.
    
    :param item: Item to check
    :type item: Any
    :param _seen: Set of already seen object IDs for cycle detection
    :type _seen: Set, optional
    :return: True if the item needs serialization, False otherwise
    :rtype: bool
    """
    if _seen is None:
        _seen = set()
        
    item_id = id(item)
    if item_id in _seen:
        return True
    _seen.add(item_id)

    item_type = type(item)
    if item_type in JSON_COMPATIBLE_TYPES:
        return False
        
    if isinstance(item, (list, tuple)):
        results = await asyncio.gather(*[_needs_serialization(x, _seen) for x in item])
        return any(results)
        
    if isinstance(item, dict):
        results = await asyncio.gather(*[_needs_serialization(v, _seen) for v in item.values()])
        return any(results)
        
    return True


def _get_serialization_type(item: Any) -> SerializationType:
    """
    Determine the serialization type for a given item.

    :param item: The item to get serialization type for
    :type item: Any
    :return: The appropriate serialization type enum value
    :rtype: SerializationType
    """
    return _get_type_category(type(item))


async def _serialize_item(item: Any) -> Any:
    """
    Serialize a single item based on its type.

    :param item: The item to serialize
    :type item: Any
    :return: The serialized item or None if serialization fails
    :rtype: Any
    """
    if not await _needs_serialization(item):
        return item
        
    ser_type = _get_serialization_type(item)
    manager = ExecutorManager.get_instance()
    loop = asyncio.get_event_loop()
    
    try:
        if ser_type == SerializationType.ARROW:
            if len(item) < 10000:
                def serialize_arrow():
                    table = pa.Table.from_pandas(item)
                    sink = pa.BufferOutputStream()
                    with pa.ipc.new_stream(sink, table.schema) as writer:
                        writer.write_table(table)
                    return sink.getvalue().to_pybytes()
                
                arrow_bytes = await loop.run_in_executor(
                    manager.get_thread_pool(),
                    serialize_arrow
                )
            else:
                def serialize_parquet():
                    buffer = io.BytesIO()
                    item.to_parquet(buffer, compression='zstd', index=True)
                    return buffer.getvalue()
                
                arrow_bytes = await loop.run_in_executor(
                    manager.get_process_pool(),
                    serialize_parquet
                )
            
            return {
                'type': 'arrow',
                'data': base64.b64encode(arrow_bytes).decode('ascii')
            }
            
        elif ser_type == SerializationType.NUMPY:
            def serialize_numpy():
                buffer = io.BytesIO()
                np.save(buffer, item, allow_pickle=False)
                return buffer.getvalue()
            
            numpy_bytes = await loop.run_in_executor(
                manager.get_thread_pool(),
                serialize_numpy
            )
            
            return {
                'type': 'numpy',
                'data': base64.b64encode(numpy_bytes).decode('ascii')
            }
            
        elif ser_type == SerializationType.IMAGE:
            def serialize_image():
                buffer = io.BytesIO()
                item.save(buffer, format='PNG', optimize=True)
                return buffer.getvalue()
            
            image_bytes = await loop.run_in_executor(
                manager.get_thread_pool(),
                serialize_image
            )
            
            return {
                'type': 'image',
                'data': base64.b64encode(image_bytes).decode('ascii')
            }
            
        elif ser_type == SerializationType.PLOTLY:
            def serialize_plotly():
                fig_dict = item.to_dict()
                return get_compressor().compress(orjson.dumps(fig_dict))
            
            plotly_bytes = await loop.run_in_executor(
                manager.get_thread_pool(),
                serialize_plotly
            )
            
            return {
                'type': 'plotly',
                'data': base64.b64encode(plotly_bytes).decode('ascii')
            }
            
        elif ser_type == SerializationType.MSGPACK:
            def serialize_msgpack():
                packed = msgpack.packb(item, use_bin_type=True)
                return get_compressor().compress(packed)
            
            msgpack_bytes = await loop.run_in_executor(
                manager.get_thread_pool(),
                serialize_msgpack
            )
            
            return {
                'type': 'msgpack',
                'data': base64.b64encode(msgpack_bytes).decode('ascii')
            }
            
        elif ser_type == SerializationType.PICKLE:
            def serialize_pickle():
                pickled = pickle.dumps(item, protocol=5)
                return get_compressor().compress(pickled)
            
            pickle_bytes = await loop.run_in_executor(
                manager.get_process_pool(),
                serialize_pickle
            )
            
            return {
                'type': 'pickle',
                'data': base64.b64encode(pickle_bytes).decode('ascii')
            }
            
    except Exception as e:
        logger.warning(f"Failed to serialize item of type {type(item)}: {str(e)}")
        return None


async def serialize_metadata(
    response_metadata: Dict[str, Any], 
    end_state: bool = False
) -> Dict[str, Any]:
    """
    Serialize metadata dictionary containing various data types.
    
    :param response_metadata: Dictionary containing metadata to serialize
    :type response_metadata: Dict[str, Any]
    :param end_state: Whether this is the final state
    :type end_state: bool, optional
    :return: Dictionary containing serialized data
    :rtype: Dict[str, Any]
    """
    serialized_data = {}
    
    try:
        items_to_process = [
            (key, value) for key, value in response_metadata.items()
            if await _needs_serialization(value)
        ]
        
        if items_to_process:
            results = await asyncio.gather(*[
                _serialize_item(value) for _, value in items_to_process
            ])
            
            for (key, _), result in zip(items_to_process, results):
                if result is not None:
                    serialized_data[key] = result
                    
    except Exception as e:
        logger.error(f"Failed to serialize items: {str(e)}")
        
    return serialized_data


async def _unserialize_item(item: Any) -> Any:
    """
    Unserialize a single item based on its type.

    :param item: The serialized item to unserialize
    :type item: Any
    :return: The unserialized item or None if unserialization fails
    :rtype: Any
    """
    if not isinstance(item, dict) or 'type' not in item or 'data' not in item:
        return item
            
    try:
        binary_data = base64.b64decode(item['data'])
        manager = ExecutorManager.get_instance()
        loop = asyncio.get_event_loop()
        
        if item['type'] == 'arrow':
            def unserialize_arrow():
                if len(binary_data) < 1_000_000:
                    return pa.ipc.open_stream(binary_data).read_all().to_pandas()
                else:
                    return pd.read_parquet(io.BytesIO(binary_data))
            
            return await loop.run_in_executor(
                manager.get_thread_pool(),
                unserialize_arrow
            )
            
        elif item['type'] == 'plotly':
            def unserialize_plotly():
                return go.Figure(
                    orjson.loads(
                        get_decompressor().decompress(binary_data)
                    )
                )
            
            return await loop.run_in_executor(
                manager.get_thread_pool(),
                unserialize_plotly
            )
            
        elif item['type'] == 'msgpack':
            def unserialize_msgpack():
                return msgpack.unpackb(
                    get_decompressor().decompress(binary_data),
                    raw=False
                )
            
            return await loop.run_in_executor(
                manager.get_thread_pool(),
                unserialize_msgpack
            )
            
        elif item['type'] == 'pickle':
            def unserialize_pickle():
                return pickle.loads(
                    get_decompressor().decompress(binary_data)
                )
            
            return await loop.run_in_executor(
                manager.get_process_pool(),
                unserialize_pickle
            )
            
        elif item['type'] == 'numpy':
            def unserialize_numpy():
                return np.load(io.BytesIO(binary_data), allow_pickle=False)
            
            return await loop.run_in_executor(
                manager.get_thread_pool(),
                unserialize_numpy
            )
            
        elif item['type'] == 'image':
            def unserialize_image():
                return Image.open(io.BytesIO(binary_data))
            
            return await loop.run_in_executor(
                manager.get_thread_pool(),
                unserialize_image
            )
            
    except Exception as e:
        logger.error(f"Failed to unserialize item: {str(e)}")
        return None
    
    return item


async def unserialize_metadata(response_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unserialize metadata dictionary containing various data types.
    
    :param response_metadata: Dictionary containing serialized metadata
    :type response_metadata: Dict[str, Any]
    :return: Dictionary containing unserialized data
    :rtype: Dict[str, Any]
    """
    try:
        items_to_process = [
            (key, value) for key, value in response_metadata.items()
            if isinstance(value, dict) and 'type' in value and 'data' in value
        ]
        
        if items_to_process:
            results = await asyncio.gather(*[
                _unserialize_item(value) for _, value in items_to_process
            ])
            
            return {
                key: result if result is not None else value
                for (key, value), result in zip(items_to_process, results)
            }
            
        return response_metadata
                    
    except Exception as e:
        logger.error(f"Failed to unserialize items: {str(e)}")
        return {}


@contextlib.asynccontextmanager
async def serialization_context():
    """
    Context manager for managing serialization resources.
    
    Ensures proper cleanup of executor pools after serialization operations.
    
    :yield: Nothing
    :rtype: None
    """
    manager = ExecutorManager.get_instance()
    try:
        yield
    finally:
        manager.cleanup()
