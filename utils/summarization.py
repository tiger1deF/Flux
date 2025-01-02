"""
Utility functions for generating concise, token-efficient summaries of data types.

This module provides functions for summarizing complex data structures in a way that:
1. Minimizes token usage while preserving key information
2. Helps LLMs understand data types and their potential uses
3. Facilitates tool selection and parameter matching
4. Supports parallel processing of large metadata structures
"""

from typing import Dict, Any, List, Tuple, Callable, Generator, AsyncGenerator, Coroutine, Optional
import polars as pl
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import asyncio
import inspect
import torch
from PIL import Image
import re
from collections import defaultdict, Counter, deque
import uuid
import decimal
from enum import Enum

from agents.storage.models import FileType


class MetadataType(str, Enum):
    """
    Enumeration mapping metadata types to their summary handlers.
    Each type corresponds to a specific data format and summary method.
    """
    DATAFRAME = "dataframe"      # Polars/pandas DataFrames
    TENSOR = "tensor"           # PyTorch tensors
    FIGURE = "figure"           # Plotly figures
    ARRAY = "array"            # NumPy arrays
    SEQUENCE = "sequence"       # Lists/tuples
    MAPPING = "mapping"         # Dictionaries
    COLLECTION = "collection"   # Collections (deque/Counter)
    CALLABLE = "callable"       # Functions
    PRIMITIVE = "primitive"     # Basic types (str/int/float)
    DATETIME = "datetime"       # Datetime objects
    PATH = "path"              # Path objects
    IMAGE = "image"            # PIL Images
    REGEX = "regex"            # Regex patterns
    GENERATOR = "generator"     # Generators
    UNKNOWN = "unknown"        # Fallback type

    @classmethod
    async def get_handler(cls, data: Any) -> Tuple[Callable, 'MetadataType']:
        """
        Get the appropriate summary handler and type for given data.
        
        :param data: Data to get handler for
        :type data: Any
        :return: Tuple of (summary_handler, metadata_type)
        :rtype: Tuple[Callable, MetadataType]
        """
        if isinstance(data, pl.DataFrame):
            return summarize_dataframe, cls.DATAFRAME
        elif isinstance(data, torch.Tensor):
            return summarize_tensor, cls.TENSOR
        elif isinstance(data, go.Figure):
            return summarize_figure, cls.FIGURE
        elif isinstance(data, np.ndarray):
            return summarize_array, cls.ARRAY
        elif isinstance(data, (list, tuple, set)):
            return summarize_sequence, cls.SEQUENCE
        elif isinstance(data, dict):
            return summarize_mapping, cls.MAPPING
        elif isinstance(data, (deque, Counter, defaultdict)):
            return summarize_collection, cls.COLLECTION
        elif callable(data):
            return summarize_callable, cls.CALLABLE
        elif isinstance(data, (str, int, float, bool)):
            return summarize_primitive, cls.PRIMITIVE
        elif isinstance(data, datetime):
            return summarize_datetime, cls.DATETIME
        elif isinstance(data, Path):
            return summarize_path, cls.PATH
        elif isinstance(data, Image.Image):
            return summarize_image, cls.IMAGE
        elif isinstance(data, re.Pattern):
            return summarize_regex, cls.REGEX
        elif isinstance(data, (Generator, AsyncGenerator)):
            return summarize_generator, cls.GENERATOR
        else:
            return lambda x, name: f"`{name}` → {type(x).__name__}()", cls.UNKNOWN


def summarize_dataframe(df: pl.DataFrame, name: str) -> str:
    """
    Generate LLM-friendly DataFrame description.
    
    :param df: DataFrame to describe
    :param name: Variable name
    :return: Description string
    """
    # Show all columns, not just first 3
    cols = ', '.join(df.columns)
    desc = f"dataframe with {df.height} rows containing columns: {cols}"
    return f"`{name}` → {desc}"  # Return just the description, name/type handled by metadata


async def summarize_figure(fig: go.Figure, name: str) -> str:
    """
    Generate plot summary optimized for LLM comprehension.
    
    :param fig: Plotly figure to summarize
    :type fig: go.Figure
    :return: Summary string like "plot<scatter,line>[2](x=date,y=price)"
    :rtype: str
    """
    traces = [t.name or t.type for t in fig.data]
    axes = f"x={getattr(traces[0], 'x', 'data')},y={getattr(traces[0], 'y', 'data')}"
    return f"`{name}` → plot with {len(traces)} traces [{', '.join(traces[:2])}...]"


async def summarize_array(arr: np.ndarray, name: str) -> str:
    """Generate array summary optimized for LLM understanding."""
    if arr.size == 0:
        return f"`{name}` → empty array"
        
    # Enhanced semantic hints based on shape, content and name
    if len(arr.shape) == 1:
        if arr.dtype in [np.int32, np.int64]:
            desc = f"integer vector of {arr.size} values"
            desc += f" [range: {arr.min()} to {arr.max()}]"
        elif arr.dtype in [np.float32, np.float64]:
            desc = f"float vector of {arr.size} values"
            desc += f" [range: {arr.min():.2f} to {arr.max():.2f}]"
        elif arr.dtype == bool:
            true_count = np.sum(arr)
            desc = f"boolean vector of {arr.size} values ({true_count} true)"
        else:
            desc = f"vector of {arr.size} {arr.dtype} values"
    elif len(arr.shape) == 2:
        if 'mask' in name:
            desc = f"binary mask {arr.shape[0]}×{arr.shape[1]} ({arr.dtype})"
            if arr.dtype == bool:
                true_count = np.sum(arr)
                desc += f" with {true_count} true values"
        elif 'image' in name or arr.shape[2] in [1, 3, 4]:
            desc = f"image data {arr.shape[0]}×{arr.shape[1]} ({arr.dtype})"
            if np.issubdtype(arr.dtype, np.number):
                desc += f" [range: {arr.min():.2f} to {arr.max():.2f}]"
        elif arr.shape[0] == arr.shape[1]:
            desc = f"square matrix {arr.shape[0]}×{arr.shape[1]} ({arr.dtype})"
            if np.issubdtype(arr.dtype, np.number):
                desc += f" [range: {arr.min():.2f} to {arr.max():.2f}]"
        else:
            desc = f"matrix {arr.shape[0]}×{arr.shape[1]} ({arr.dtype})"
            if np.issubdtype(arr.dtype, np.number):
                desc += f" [range: {arr.min():.2f} to {arr.max():.2f}]"
    else:
        dims = '×'.join(map(str, arr.shape))
        if 'feature' in name:
            desc = f"{arr.dtype} feature tensor of shape {dims}"
        elif 'conv' in name or 'filter' in name:
            desc = f"{arr.dtype} convolution kernel of shape {dims}"
        elif 'batch' in name:
            desc = f"{arr.dtype} batch tensor of shape {dims}"
        else:
            desc = f"{arr.dtype} tensor of shape {dims}"
        if np.issubdtype(arr.dtype, np.number):
            desc += f" [range: {arr.min():.2f} to {arr.max():.2f}]"
            
    # Add memory info for large arrays
    if arr.size > 1_000_000:  # Only for arrays > 1M elements
        memory_mb = arr.nbytes / (1024 * 1024)
        desc += f" ({memory_mb:.1f}MB)"
        
    return f"`{name}` → {desc}"


async def summarize_sequence(seq: List, name: str) -> str:
    """Generate LLM-friendly sequence description."""
    preview = ', '.join(map(str, seq[:2])) + ("..." if len(seq) > 2 else "")
    desc = f"list of {len(seq)} items: [{preview}]"
    return f"`{name}` → {desc}"


async def summarize_mapping(d: Dict, name: str) -> str:
    """
    Generate mapping summary optimized for LLM understanding.
    
    :param d: Dictionary to summarize
    :param name: Variable name
    :return: Name and description in format "name → description"
    """
    if not d:
        return f"`{name}` → empty dictionary"
        
    # Get key and value types
    key_type = type(next(iter(d.keys()))).__name__
    val_type = type(next(iter(d.values()))).__name__
    
    # Add semantic hints based on content
    if all(isinstance(v, (int, float)) for v in d.values()):
        desc = f"numeric mapping with {len(d)} {key_type} keys to {val_type} values"
    elif all(isinstance(v, str) for v in d.values()):
        desc = f"string mapping with {len(d)} {key_type} keys"
    elif all(isinstance(v, dict) for v in d.values()):
        desc = f"nested mapping with {len(d)} {key_type} keys to sub-dictionaries"
    else:
        desc = f"mapping of {len(d)} {key_type} keys to {val_type} values"
        
    # Add key preview
    preview = ', '.join(map(str, list(d.keys())[:3]))
    if len(d) > 3:
        preview += "..."
    desc += f" [{preview}]"
        
    return f"`{name}` → {desc}"


async def summarize_callable(func: Callable, name: str) -> str:
    """Generate LLM-friendly function description."""
    sig = inspect.signature(func)
    params = [f"{p.name}" for p in sig.parameters.values()]
    
    if 'process' in name or 'transform' in name:
        desc = f"data transformation taking ({', '.join(params)})"
    elif 'predict' in name or 'infer' in name:
        desc = f"model inference function with ({', '.join(params)})"
    else:
        desc = f"function with parameters ({', '.join(params)})"
    return f"`{name}` → {desc}"


async def summarize_primitive(value: Any, name: str) -> str:
    """Generate concise primitive summary."""
    if isinstance(value, str):
        preview = value[:30] + "..." if len(value) > 30 else value
        return f"`{name}` → string of length {len(value)} [{preview}]"
    return f"`{name}` → {type(value).__name__}({value})"


async def summarize_datetime(dt: datetime, name: str) -> str:
    """Generate concise datetime summary."""
    return f"`{name}` → timestamp {dt.isoformat()}"


async def summarize_path(path: Path, name: str) -> str:
    """Generate concise path summary."""
    return f"`{name}` → file path '{path.name}'"


async def summarize_tensor(tensor: torch.Tensor, name: str) -> str:
    """Generate LLM-friendly tensor description."""
    device = "GPU" if tensor.is_cuda else "CPU"
    
    # Enhanced semantic hints based on shape and content
    if len(tensor.shape) == 1:
        if tensor.dtype in [torch.int32, torch.int64]:
            desc = f"integer vector of {tensor.shape[0]} values on {device}"
            if tensor.numel() > 0:
                desc += f" [range: {tensor.min().item()} to {tensor.max().item()}]"
        elif tensor.dtype in [torch.float32, torch.float64]:
            desc = f"float vector of {tensor.shape[0]} values on {device}"
            if tensor.numel() > 0:
                desc += f" [range: {tensor.min().item():.2f} to {tensor.max().item():.2f}]"
        else:
            desc = f"{tensor.dtype} vector of {tensor.shape[0]} values on {device}"
    elif len(tensor.shape) == 2:
        if 'embed' in name or 'embedding' in name:
            desc = f"{tensor.shape[0]} embeddings of dim {tensor.shape[1]} ({tensor.dtype} on {device})"
        elif 'weight' in name:
            desc = f"weight matrix {tensor.shape[0]}×{tensor.shape[1]} ({tensor.dtype} on {device})"
        elif tensor.shape[0] == tensor.shape[1]:
            desc = f"square matrix {tensor.shape[0]}×{tensor.shape[1]} ({tensor.dtype} on {device})"
        else:
            desc = f"matrix {tensor.shape[0]}×{tensor.shape[1]} ({tensor.dtype} on {device})"
    else:
        dims = '×'.join(map(str, tensor.shape))
        if 'conv' in name or 'filter' in name:
            desc = f"{tensor.dtype} convolution kernel of shape {dims} on {device}"
        elif 'feature' in name:
            desc = f"{tensor.dtype} feature tensor of shape {dims} on {device}"
        else:
            desc = f"{tensor.dtype} tensor of shape {dims} on {device}"
            
    # Add gradient info
    if tensor.requires_grad:
        desc += " (requires gradient)"
        
    return f"`{name}` → {desc}"


async def summarize_image(img: Image.Image, name: str) -> str:
    """Generate concise image summary."""
    return f"`{name}` → {img.mode} image {img.width}×{img.height}"


async def summarize_regex(pattern: re.Pattern, name: str) -> str:
    """Generate concise regex summary."""
    return f"`{name}` → pattern '{pattern.pattern}'"


async def summarize_generator(gen: Generator, name: str) -> str:
    """Generate LLM-friendly generator description."""
    desc = "data stream generator"
    return f"`{name}` → {desc}"


async def summarize_async_generator(gen: AsyncGenerator, name: str) -> str:
    """Generate concise async generator summary."""
    return f"`{name}` → async data stream"


async def summarize_coroutine(coro: Coroutine, name: str) -> str:
    """Generate concise coroutine summary."""
    return f"`{name}` → async task {coro.__name__ if hasattr(coro, '__name__') else 'anonymous'}"


async def summarize_collection(col: Any, name: str) -> str:
    """Generate LLM-friendly collection description."""
    if isinstance(col, (defaultdict, Counter)):
        desc = f"collection of {len(col)} counted items"
    elif isinstance(col, deque):
        desc = f"queue with {len(col)} elements"
    else:
        desc = f"collection containing {len(col)} items"
    return f"`{name}` → {desc}"


async def summarize_decimal(dec: decimal.Decimal, name: str) -> str:
    """Generate concise decimal summary."""
    return f"`{name}` → precise number {dec}"


async def summarize_uuid(uid: uuid.UUID, name: str) -> str:
    """Generate concise UUID summary."""
    return f"`{name}` → unique ID {str(uid)[:8]}..."


async def summarize_file(file_data: Any, file_type: 'FileType', name: str, path: Optional[Path] = None) -> str:
    """
    Generate a concise, LLM-friendly summary of a file's contents.
    
    :param file_data: File content to summarize
    :param file_type: Type of the file from FileType enum
    :param name: Name/identifier of the file
    :param path: Optional file path
    :return: Formatted summary string
    """
    # Handle binary file types with size-based summaries
    if file_type in {FileType.BINARY, FileType.IMAGE, FileType.AUDIO, FileType.VIDEO, FileType.ARCHIVE}:
        if isinstance(file_data, bytes):
            size_mb = len(file_data) / (1024 * 1024)
            return f"`{name}` → {file_type} file{f' at {path.name}' if path else ''} ({size_mb:.1f}MB)"
        return f"`{name}` → {file_type} file{f' at {path.name}' if path else ''}"

    # Handle structured data files
    if file_type in {FileType.CSV, FileType.PARQUET}:
        if isinstance(file_data, (pl.DataFrame, pd.DataFrame)):
            cols = ', '.join(file_data.columns)
            rows = len(file_data)
            return f"`{name}` → {file_type} data with {rows} rows and columns: {cols}"
        return f"`{name}` → {file_type} data file{f' at {path.name}' if path else ''}"

    if file_type in {FileType.JSON, FileType.YAML}:
        if isinstance(file_data, dict):
            top_keys = list(file_data.keys())[:3]
            preview = ', '.join(top_keys) + ('...' if len(file_data) > 3 else '')
            return f"`{name}` → {file_type} structure with keys: [{preview}]"
        return f"`{name}` → {file_type} data file{f' at {path.name}' if path else ''}"

    # Handle code and text files
    if file_type in {
        FileType.PYTHON, FileType.JAVASCRIPT, FileType.TYPESCRIPT, FileType.JAVA,
        FileType.CPP, FileType.CSHARP, FileType.GO, FileType.RUST, FileType.PHP,
        FileType.RUBY, FileType.SWIFT, FileType.KOTLIN, FileType.SCALA, FileType.R,
        FileType.SHELL, FileType.SQL
    }:
        content = str(file_data)
        
        # Extract imports/includes
        import_patterns = {
            FileType.PYTHON: r'^(?:from|import)\s+[\w\.*]+',
            FileType.JAVASCRIPT: r'^(?:import|require)\s*\([^)]+\)|^import\s+.*?from\s+[\'"].*?[\'"]',
            FileType.TYPESCRIPT: r'^(?:import|require)\s*\([^)]+\)|^import\s+.*?from\s+[\'"].*?[\'"]',
            FileType.JAVA: r'^import\s+[\w\.]+;',
            FileType.CPP: r'^#include\s+[<"].*?[>"]',
            FileType.RUBY: r'^require\s+[\'"].*?[\'"]',
            FileType.PHP: r'^(?:require|include)(?:_once)?\s+[\'"].*?[\'"];'
        }
        
        # Extract functions/classes
        func_patterns = {
            FileType.PYTHON: r'(?:^|\n)(?:def|class)\s+(\w+)',
            FileType.JAVASCRIPT: r'(?:^|\n)(?:function|class)\s+(\w+)',
            FileType.TYPESCRIPT: r'(?:^|\n)(?:function|class|interface)\s+(\w+)',
            FileType.JAVA: r'(?:^|\n)(?:public|private|protected)?\s*(?:class|interface|enum)\s+(\w+)',
            FileType.CPP: r'(?:^|\n)(?:class|struct|enum)\s+(\w+)',
            FileType.RUBY: r'(?:^|\n)(?:def|class)\s+(\w+)',
            FileType.PHP: r'(?:^|\n)(?:function|class)\s+(\w+)'
        }

        imports = []
        if pattern := import_patterns.get(file_type):
            imports = re.findall(pattern, content, re.MULTILINE)[:3]
            
        functions = []
        if pattern := func_patterns.get(file_type):
            functions = re.findall(pattern, content, re.MULTILINE)[:3]

        summary = f"`{name}` → {file_type} source file\n"
        if path:
            if isinstance(path, str):
                summary += f"(path: '{path}')\n"
            else:
                summary += f"(path: '{path.name}')\n"
                
        if imports:
            summary += f"\nImports: {', '.join(imports)[:100]}..."
        if functions:
            summary += f"\nDefines: {', '.join(functions)}"
        
        loc = len(content.splitlines())
        summary += f"\n{loc} lines of code"
        
        return summary

    # Handle markup and documentation files
    if file_type in {FileType.HTML, FileType.MARKDOWN, FileType.XML}:
        content = str(file_data)
        if file_type == FileType.HTML:
            title_match = re.search(r'<title>(.*?)</title>', content)
            h1_match = re.search(r'<h1>(.*?)</h1>', content)
            title = title_match.group(1) if title_match else h1_match.group(1) if h1_match else None
        elif file_type == FileType.MARKDOWN:
            title_match = re.search(r'^#\s+(.*)$', content, re.MULTILINE)
            title = title_match.group(1) if title_match else None
        else:  # XML
            root_match = re.search(r'<(\w+)[^>]*>', content)
            title = root_match.group(1) if root_match else None

        summary = f"`{name}` → {file_type} document"
        if path:
            summary += f" '{path.name}'"
        if title:
            summary += f" titled '{title}'"
        summary += f" ({len(content.splitlines())} lines)"
        return summary

    # Handle stylesheet files
    if file_type == FileType.CSS:
        content = str(file_data)
        selectors = re.findall(r'([.#]?\w+)\s*{', content)[:5]
        return f"`{name}` → stylesheet with {len(selectors)} rules [{', '.join(selectors[:3])}...]"

    # Handle configuration files
    if file_type == FileType.CONFIG:
        content = str(file_data)
        return f"`{name}` → configuration file{f' at {path.name}' if path else ''} ({len(content.splitlines())} lines)"

    # Default fallback for unknown types
    return f"`{name}` → {file_type} file{f' at {path.name}' if path else ''}"


SUMMARY_HANDLERS = {
    pl.DataFrame: summarize_dataframe,
    go.Figure: summarize_figure,
    np.ndarray: summarize_array,
    list: summarize_sequence,
    tuple: summarize_sequence,
    set: summarize_sequence,
    dict: summarize_mapping,
    datetime: summarize_datetime,
    Path: summarize_path,
    torch.Tensor: summarize_tensor,
    Image.Image: summarize_image,
    re.Pattern: summarize_regex,
    Generator: summarize_generator,
    AsyncGenerator: summarize_async_generator,
    Coroutine: summarize_coroutine,
    defaultdict: summarize_collection,
    Counter: summarize_collection,
    deque: summarize_collection,
    decimal.Decimal: summarize_decimal,
    uuid.UUID: summarize_uuid,
}


async def get_summary_handler(data: Any) -> Callable:
    """
    Get appropriate summary handler for object type.
    
    :param data: Object to get handler for
    :type data: Any
    :return: Summary handler function
    :rtype: Callable
    """
    handler, _ = await MetadataType.get_handler(data)
    return handler


async def summarize_dictionary(metadata: Dict[str, Any]) -> str:
    """
    Generate parallel summaries of dictionary items.
    
    :param metadata: Dictionary of items to summarize
    :type metadata: Dict[str, Any]
    :return: Newline-separated summaries
    :rtype: str
    """
    async def process_item(key: str, value: Any) -> str:
        handler = await get_summary_handler(value)
        return await handler(value, key)

    summaries = await asyncio.gather(*[
        process_item(k, v) for k, v in metadata.items()
    ])
    
    return "\n".join(summaries)

