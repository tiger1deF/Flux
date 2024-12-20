"""
Utility functions for generating summaries of various data types.

This module provides functions for summarizing different types of data,
including files, metadata, and complex data structures, with support for
truncation and type-specific formatting.
"""

from typing import Dict, Any, Union
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import sys
from functools import lru_cache

from agents.messages.models import File

from utils.shared.tokenizer import tokenizer


@lru_cache(maxsize = 128)
def get_size_str(size_bytes: int) -> str:
    """
    Convert byte size to human readable string.
    
    :param size_bytes: Size in bytes
    :type size_bytes: int
    :return: Formatted size string (e.g., "1.5 MB")
    :rtype: str
    """
    for unit in {'B', 'KB', 'MB', 'GB'}:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_type_handler(obj: Any) -> str:
    """
    Determine the appropriate handler for a given object type.
    
    :param obj: Object to determine handler for
    :type obj: Any
    :return: String identifier for the handler type
    :rtype: str
    """
    if isinstance(obj, pd.DataFrame):
        return "dataframe"
    elif isinstance(obj, go.Figure):
        return "figure"
    elif isinstance(obj, np.ndarray):
        return "ndarray"
    elif isinstance(obj, (list, tuple, set)):
        return "sequence"
    elif isinstance(obj, dict):
        return "mapping"
    elif callable(obj):
        return "callable"
    elif isinstance(obj, (str, int, float, bool)):
        return "primitive"
    elif isinstance(obj, datetime):
        return "datetime"
    elif isinstance(obj, Path):
        return "path"
    else:
        return "unknown"


def truncate_text(
    text: str, 
    max_tokens: int
) -> str:
    """
    Truncate text to a maximum number of tokens.
    
    :param text: Text to truncate
    :type text: str
    :param max_tokens: Maximum number of tokens
    :type max_tokens: int
    :return: Truncated text
    :rtype: str
    """
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    truncated_start = tokens[:max_tokens // 2]
    truncated_end = tokens[-max_tokens // 2:]
    
    truncated_text = tokenizer.decode(truncated_start) + "..." + tokenizer.decode(truncated_end)
    
    return truncated_text


def summarize_file(file: File, char_cutoff: int = 100) -> str:
    """
    Generate a summary string for a File object.
    
    :param file: File object to summarize
    :type file: File
    :param char_cutoff: Maximum characters for text previews
    :type char_cutoff: int
    :return: Summary string
    :rtype: str
    """
    summary_parts = []
    
    # Add path if present
    if file.path:
        summary_parts.append(f"path='{file.path}'")
    
    # Add description preview if present
    if file.description:
        desc_preview = file.description[:char_cutoff] + "..." if len(file.description) > char_cutoff else file.description
        summary_parts.append(f"desc='{desc_preview}'")
    
    # Add content type/preview
    content_type = get_type_handler(file.content)
    if content_type == "primitive" and isinstance(file.content, str):
        preview = file.content[:char_cutoff] + "..." if len(file.content) > char_cutoff else file.content
        summary_parts.append(f"content='{preview}'")
    else:
        summary_parts.append(f"content_type={content_type}")
    
    # Add annotation count if present
    if file.annotations:
        summary_parts.append(f"annotations={len(file.annotations)}")
    
    return f"File({', '.join(summary_parts)})"


def summarize_metadata(
    metadata: Dict[str, Any], 
    char_cutoff: int = 100
) -> str:
    """
    Generate a summary string for metadata dictionary.
    
    :param metadata: Dictionary of metadata to summarize
    :type metadata: Dict[str, Any]
    :param char_cutoff: Maximum characters for text previews
    :type char_cutoff: int
    :return: Summary string
    :rtype: str
    """
    summaries = []
    
    for key, value in metadata.items():
        handler_type = get_type_handler(value)
        
        if isinstance(value, File):
            summary = f"`{key}`: {summarize_file(value, char_cutoff)}"
            
        elif handler_type == "dataframe":
            summary = (f"`{key}`: (DataFrame) "
                      f"{value.shape[0]} rows × {value.shape[1]} cols "
                      f"[{', '.join(value.columns[:3])}...]")
            
        elif handler_type == "figure":
            traces = [t.name or t.type for t in value.data]
            summary = f"`{key}`: (Figure) {len(traces)} traces [{', '.join(traces[:3])}...]"
            
        elif handler_type == "ndarray":
            summary = f"`{key}`: (Array) shape={value.shape}, dtype={value.dtype}"
            
        elif handler_type == "sequence":
            items = str(value[:3])[1:-1] + "..." if len(value) > 3 else str(value)[1:-1]
            summary = f"`{key}`: (List) {len(value)} items [{items}]"
            
        elif handler_type == "mapping":
            keys = list(value.keys())[:3]
            summary = f"`{key}`: (Dict) {len(value)} keys [{', '.join(map(str, keys))}...]"
            
        elif handler_type == "callable":
            summary = f"`{key}`: (Function) {value.__name__}()"

        elif handler_type == "primitive":
            if isinstance(value, str) and len(value) > char_cutoff:
                summary = f"`{key}`: (String) \"{value[:char_cutoff]}...\""
            else:
                summary = f"{type(value).__name__} '{key}': {value}"
                
        elif handler_type == "datetime":
            summary = f"`{key}`: (Date) {value.isoformat()}"
            
        elif handler_type == "path":
            summary = f"`{key}`: (Path) {value}"
            
        else:
            size = get_size_str(sys.getsizeof(value))
            summary = f"`{key}`: (Object) type={type(value).__name__}, size={size}"
            
        summaries.append(summary)
    
    
    return "\n".join(summaries)


def summarize_files(
    files: Dict[str, File], 
    char_cutoff: int = 100
) -> str:
    """
    Generate summaries for a dictionary of File objects.
    
    :param files: Dictionary of File objects to summarize
    :type files: Dict[str, File]
    :param char_cutoff: Maximum characters for text previews
    :type char_cutoff: int
    :return: Summary string
    :rtype: str
    """
    summaries = []
    
    for key, file in files.items():
        summary_parts = []
        
        # Add path if present
        if file.path:
            summary_parts.append(f"path='{file.path}'")
        
        # Add description preview if present
        if file.description:
            desc_preview = file.description[:char_cutoff] + "..." if len(file.description) > char_cutoff else file.description
            summary_parts.append(f"desc='{desc_preview}'")
        
        # Add content type/preview
        content_type = get_type_handler(file.content)
        if content_type == "primitive" and isinstance(file.content, str):
            preview = file.content[:char_cutoff] + "..." if len(file.content) > char_cutoff else file.content
            summary_parts.append(f"content='{preview}'")
        else:
            summary_parts.append(f"content_type={content_type}")
        
        # Add annotation count if present
        if file.annotations:
            summary_parts.append(f"annotations={len(file.annotations)}")
        
        summary = f"`{key}`: File({', '.join(summary_parts)})"
        summaries.append(summary)
    
    return "\n".join(summaries)