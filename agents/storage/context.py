from pydantic import BaseModel, Field, PrivateAttr
from enum import Enum
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
from functools import lru_cache

from utils.shared.tokenizer import encode_async
from agents.storage.models import IDFactory
from agents.vectorstore.models import BaseEmbeddingFunction


class ContextType(Enum):
    """
    Types of context entries.
    
    :cvar PROCESS: Process-related context
    :cvar CONTEXT: General context information
    :cvar METRIC: Metric/measurement data
    :cvar PLAN: Planning/strategy information
    :cvar ERROR: Error context
    """
    PROCESS = "process"
    CONTEXT = "context"
    METRIC = "metric"
    PLAN = "plan"
    ERROR = "error"


class Context(BaseModel):
    """
    Context entry for storing agent execution context.
    
    :ivar id: Unique identifier
    :type id: str
    :ivar type: Type of context entry
    :type type: ContextType
    :ivar content: Context content
    :type content: str
    :ivar date: Creation timestamp
    :type date: datetime
    :ivar sender: Entity that sent the message
    :type sender: str
    :ivar agent: Name of the agent
    :type agent: str
    :ivar context_type: Type of context entry (input/output/error)
    :type context_type: str
    :ivar duration: Processing duration in seconds
    :type duration: Optional[float]
    """
    id: str = Field(default_factory=lambda: IDFactory.next_id())
    type: ContextType
    content: str
    date: datetime = Field(default_factory=datetime.now)
    sender: Optional[str] = None
    agent: str
    context_type: str  # "input", "output", "error", etc.
    duration: Optional[float] = None
    error_type: Optional[str] = None
    score: Optional[float] = None

    _tokens: Optional[List[int]] = PrivateAttr(default=None)

    async def to_dict(self) -> Dict[str, Any]:
        """
        Convert context to dictionary representation.
        
        :return: Dictionary containing context data
        :rtype: Dict[str, Any]
        """
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "date": self.date.isoformat(),
            "metadata": self.metadata,
            "score": self.score
        }

    @classmethod
    async def from_dict(cls, data: Dict[str, Any]) -> 'Context':
        """
        Create Context instance from dictionary.
        
        :param data: Dictionary containing context data
        :type data: Dict[str, Any]
        :return: New Context instance
        :rtype: Context
        """
        data["type"] = ContextType(data["type"])
        data["date"] = datetime.fromisoformat(data["date"])
        return cls(**data)

    async def serialize(self) -> str:
        """
        Serialize context to JSON string.
        
        :return: JSON string representation
        :rtype: str
        """
        data = await self.to_dict()
        if hasattr(self, 'embedding_function') and self.embedding_function:
            data['embedding_function'] = await self.embedding_function.serialize()
        return json.dumps(data)

    @classmethod
    async def deserialize(cls, serialized_data: str) -> 'Context':
        """
        Create Context instance from serialized string.
        
        :param serialized_data: Serialized context data
        :type serialized_data: str
        :return: New Context instance
        :rtype: Context
        """
        data = json.loads(serialized_data)
        if 'embedding_function' in data:
            data['embedding_function'] = await BaseEmbeddingFunction.deserialize(
                data['embedding_function']
            )
        return await cls.from_dict(data)

    async def tokens(self) -> List[int]:
        """Get tokenized content with caching"""
        if self._tokens is None:
            self._tokens = await encode_async(self.content)
        return self._tokens
    
    async def token_length(self) -> int:
        """Get the token length asynchronously"""
        if self._tokens is None:
            await self.tokens()
        return len(self._tokens)

    def __hash__(self) -> int:
        """Make Context hashable for caching"""
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        """Define equality for hashing"""
        if not isinstance(other, Context):
            return False
        return self.id == other.id

