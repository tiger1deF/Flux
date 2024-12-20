"""
Shared tokenizer instance for text processing.

This module provides a shared tokenizer instance using the cl100k_base encoding
from tiktoken, which is compatible with most modern language models.
"""

import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")