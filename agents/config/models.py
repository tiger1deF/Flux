"""
Core configuration models for agents and vector stores.

This module provides the base configuration models used across the agent system
for context handling and agent configuration.
"""

from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field


class TruncationType(str, Enum):
    """
    Enumeration of truncation strategies.
    """
    TOKEN_LIMIT = "token_limit"
    MESSAGE_COUNT = "message_count"
    TRIM_MAX = "trim_max"
    SLIDING = "sliding"
    PRESERVE_ENDS = "preserve_ends"
    

class RetrievalType(str, Enum):
    """
    Enumeration of context retrieval strategies.
    """
    RELEVANT = "relevant"
    HISTORY = "history"
    ALL = "all"
    NONE = "none"


class AgentStatus(str, Enum):
    """
    Enumeration of possible agent execution states.
    """
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Logging(str, Enum):
    """
    Enumeration of logging modes for agents.
    """
    LANGFUSE = "langfuse"
    ENABLED = "enabled"
    DISABLED = "disabled"


class ContextConfig(BaseModel):
    """
    Configuration for context retrieval and handling.
    """
    strategy: RetrievalType = Field(default=RetrievalType.ALL)
    item_count: int = Field(default=10)
    max_tokens: int = Field(default=2000)
    truncation_type: TruncationType = Field(default=TruncationType.TOKEN_LIMIT)
    message_count: Optional[int] = Field(default=None)
    sliding_window_ratio: float = Field(default=0.5)
    preserve_start_messages: int = Field(default=2)
    preserve_end_messages: int = Field(default=2)


class AgentConfig(BaseModel):
    """
    Configuration for agent behavior and settings.
    """
    task_prompt: Optional[str] = Field(default=None)
    system_prompt: Optional[str] = Field(default=None)
    logging: Logging = Field(default=Logging.DISABLED)
    context: ContextConfig = Field(default_factory=ContextConfig)
    max_tokens: int = Field(default=2000)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.95)
    top_k: int = Field(default=40)
    tools: List[str] = Field(default_factory=list) 