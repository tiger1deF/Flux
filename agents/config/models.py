"""
Core configuration models for agents and vector stores.

This module provides the base configuration models used across the agent system
for context handling and agent configuration.
"""

from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field

from utils.shared.tokenizer import SliceType


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
    SIMILARITY = "similarity"
    CHRONOLOGICAL = "chronological"
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
    retrieval_strategy: RetrievalType = Field(default = RetrievalType.ALL)
    truncation_type: TruncationType = Field(default = TruncationType.TOKEN_LIMIT)
    slice_type: SliceType = Field(default = SliceType.END)

    item_count: int = Field(default = 10)
    max_tokens: int = Field(default = 2_000)
    sliding_window_ratio: float = Field(default = 0.5)
    preserve_start_messages: int = Field(default = 2)
    preserve_end_messages: int = Field(default = 2)

    chunk_size: int = Field(default = 2_000)
    chunk_overlap: int = Field(default = 100)
    
    def with_context(self, **kwargs) -> 'ContextConfig':
        """Create a modified copy of the current context config"""
        return ContextConfig(
            **{**self.model_dump(), **kwargs}
        )
    
    def __truediv__(self, number: float) -> 'ContextConfig':
        """Integer division of the config parameters based on strategy
        
        :param number: The divisor
        :type number: float
        :return: New config with adjusted limits
        :rtype: ContextConfig
        """
        if not isinstance(number, (int, float)):
            raise TypeError("Divisor must be a number")
        
        if number <= 0:
            raise ValueError("Divisor must be positive")

        new_config = ContextConfig(
            retrieval_strategy = self.retrieval_strategy,
            truncation_type = self.truncation_type,
            slice_type = self.slice_type,
            item_count = self.item_count,
            max_tokens = self.max_tokens,
            sliding_window_ratio = self.sliding_window_ratio,
            preserve_start_messages = self.preserve_start_messages,
            preserve_end_messages = self.preserve_end_messages
        )

        if self.truncation_type == TruncationType.TOKEN_LIMIT:
            new_config.max_tokens = self.max_tokens // number
        
        elif self.truncation_type == TruncationType.MESSAGE_COUNT:
            new_config.item_count = max(1, round(self.item_count / number))
        
        elif self.truncation_type == TruncationType.SLIDING:
            new_config.max_tokens = self.max_tokens // number
            new_config.item_count = max(1, round(self.item_count / number))

        elif self.truncation_type == TruncationType.PRESERVE_ENDS:
            new_config.preserve_start_messages = self.preserve_start_messages // number
            new_config.preserve_end_messages = self.preserve_end_messages // number
            new_config.max_tokens = self.max_tokens // number
        
        elif self.truncation_type == TruncationType.TRIM_MAX:
            new_config.max_tokens = self.max_tokens // number
           
        return new_config


    def __mul__(self, number: float) -> 'ContextConfig':
        """Multiplication of the config parameters based on strategy
        
        :param number: The multiplier
        :type number: float
        :return: New config with adjusted limits
        :rtype: ContextConfig
        """
        if not isinstance(number, (int, float)):
            raise TypeError("Multiplier must be a number")
        
        if number <= 0:
            raise ValueError("Multiplier must be positive")

        new_config = ContextConfig(
            retrieval_strategy = self.retrieval_strategy,
            truncation_type = self.truncation_type,
            slice_type = self.slice_type,
            item_count = self.item_count,
            max_tokens = self.max_tokens,
            sliding_window_ratio = self.sliding_window_ratio,
            preserve_start_messages = self.preserve_start_messages,
            preserve_end_messages = self.preserve_end_messages
        )

        if self.truncation_type == TruncationType.TOKEN_LIMIT:
            new_config.max_tokens = round(self.max_tokens * number)
        
        elif self.truncation_type == TruncationType.MESSAGE_COUNT:
            new_config.item_count = max(1, round(self.item_count * number))
        
        elif self.truncation_type == TruncationType.SLIDING:
            new_config.max_tokens = round(self.max_tokens * number)
            new_config.item_count = max(1, round(self.item_count * number))

        elif self.truncation_type == TruncationType.PRESERVE_ENDS:
            new_config.preserve_start_messages = round(self.preserve_start_messages * number)
            new_config.preserve_end_messages = round(self.preserve_end_messages * number)
            new_config.max_tokens = round(self.max_tokens * number)
        
        elif self.truncation_type == TruncationType.TRIM_MAX:
            new_config.max_tokens = round(self.max_tokens * number)
           
        return new_config
    
    
class AgentConfig(BaseModel):
    """Agent configuration settings"""
    task_prompt: str = Field(default="")
    logging: Logging = Field(default=Logging.ENABLED)
    context_config: ContextConfig = Field(default_factory=ContextConfig)