"""
Message models and types for agent communication.

This module defines the core message types and models used for agent-to-agent 
and agent-to-user communication, including support for attachments and metadata.
"""
import os
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import Dict, Any, List, Union, Optional
from functools import lru_cache
import asyncio    
import orjson
import base64

from utils.shared.tokenizer import encode_async
from utils.shared.tokenizer import slice_text, SliceType
from utils.serialization import get_compressor, get_decompressor


class Sender(str, Enum):
    """
    Enumeration of possible message senders.
    
    :cvar USER: Message from a user
    :cvar AI: Message from an AI agent
    """
    USER = 'user'
    AI = 'ai'
    

class MessageType(str, Enum):
    """
    Enumeration of possible message types.
    """
    INPUT = 'input'
    OUTPUT = 'output'
    ERROR = 'error'
    INTERMEDIATE = 'intermediate'
    CONTEXT = 'context'

class Message(BaseModel):
    """
    Base message class for agent communication.
    
    Represents a single message in an agent conversation with support
    for content, metadata, and file attachments.
    
    :ivar sender: Entity that sent the message
    :type sender: Sender
    :ivar agent_name: Name of the sending agent 
    :type agent_name: str
    :ivar date: Timestamp of the message
    :type date: datetime
    :ivar content: Main message content
    :type content: str
    :ivar metadata_ids: Optional message metadata identifiers
    :type metadata_ids: List[Union[int, str]]
    :ivar file_ids: Optional message file identifiers
    :type file_ids: List[Union[int, str]]
    :ivar annotations: Configuration and state annotations
    :type annotations: Dict[str, Any]
    """
    id: Union[int, str] = Field(default_factory = lambda: int.from_bytes(os.urandom(3), 'big') % 1_000_000)
    sender: Sender = Field(default = Sender.AI, description = "The sender of the message")
    agent_name: str = Field(default = "default_agent", description = "The name of the agent that sent the message")
    date: datetime = Field(default_factory = datetime.now, description = "The date of the message")
    content: str = Field(default = "", description = "Message string content")
    type: MessageType = Field(default = MessageType.INTERMEDIATE, description = "The type of the message")
    
    metadata_ids: List[Union[int, str]] = Field(default = [], description = "Message metadata")
    file_ids: List[Union[int, str]] = Field(default = [], description = "Message files")
    annotations: Dict[str, Any] = Field(default = {}, description = "Message annotations for config/state items")

    # For vector-based retrieval
    score: int = Field(default = 0, description = "Score of the file")


    async def to_json(self) -> str:
        """
        Convert message to JSON string.
        
        :return: JSON string representation
        :rtype: str
        """
        loop = asyncio.get_running_loop()
        json_data = {
            'sender': self.sender.value,
            'date': self.date.isoformat(),             
        }
        return await loop.run_in_executor(None, json.dumps, json_data)
        
        
    async def read_json(self, json_str: str) -> None:
        """
        Update message from JSON string.
        
        :param json_str: JSON string containing message data
        :type json_str: str
        """
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, json.loads, json_str)
        self.sender = Sender(data['sender'])
        self.date = datetime.fromisoformat(data['date'])
        

    @lru_cache(maxsize = 1)
    async def tokens(self) -> List[int]:
        """
        Get the tokenized representation of the message content.
        
        :return: List of token IDs for the message content
        :rtype: List[int]
        """
        return await encode_async(self.content)
    
    
    @lru_cache(maxsize = 1)
    async def __len__(self) -> int:
        """
        Get the length of the message content
        """
        return len(self.content)


    @property
    async def content(self) -> str:
        """Get the message content
        
        :return: Message content
        :rtype: str
        """
        return self._content


    @content.setter
    def content(self, value: str) -> None:
        """
        Set message content and clear length cache if content changes
        
        :param value: New message content
        :type value: str
        """
        if not hasattr(self, '_content') or self._content != value:
            self.__len__.cache_clear()
        self._content = value


    async def clear_caches(self):
        """Clear all caches asynchronously"""
        self.__len__.cache_clear()
      
    
    @lru_cache(maxsize = 1)
    async def truncate(
        self,
        max_tokens: int,
        slice_type: SliceType = SliceType.START
    ) -> str:
        """
        Truncate the message content to a maximum number of tokens.
        
        :param max_tokens: Maximum number of tokens
        :type max_tokens: int
        """
        return await slice_text(
            text = self.content,
            slice_type = slice_type,
            max_tokens = max_tokens
        )


    def __bool__(self) -> bool:
        """
        Verifies existance of object
        """
        return bool(self.content)


    async def serialize(self) -> str:
        """
        Serialize message to a string representation.
        
        :return: Serialized message string
        :rtype: str
        """
        data = {
            'id': self.id,
            'sender': self.sender.value,
            'agent_name': self.agent_name,
            'date': self.date.isoformat(),
            'content': self.content,
            'type': self.type.value,
            'metadata_ids': list(self.metadata_ids),
            'file_ids': list(self.file_ids),
            'annotations': self.annotations,
            'score': self.score
        }

        # Convert any bytes objects in annotations to base64
        if self.annotations:
            data['annotations'] = self._encode_bytes_in_dict(self.annotations)
        
        # Use orjson for serialization
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: orjson.dumps(data).decode('utf-8'))

    def _encode_bytes_in_dict(self, d: Dict) -> Dict:
        """Helper to encode bytes in dictionary values"""
        encoded = {}
        for k, v in d.items():
            if isinstance(v, bytes):
                encoded[k] = {
                    'type': 'bytes',
                    'data': base64.b64encode(v).decode('ascii')
                }
            elif isinstance(v, dict):
                encoded[k] = self._encode_bytes_in_dict(v)
            else:
                encoded[k] = v
        return encoded

    def _decode_bytes_in_dict(self, d: Dict) -> Dict:
        """Helper to decode bytes in dictionary values"""
        decoded = {}
        for k, v in d.items():
            if isinstance(v, dict) and v.get('type') == 'bytes':
                decoded[k] = base64.b64decode(v['data'])
            elif isinstance(v, dict):
                decoded[k] = self._decode_bytes_in_dict(v)
            else:
                decoded[k] = v
        return decoded

    @classmethod
    async def deserialize(cls, serialized_data: str) -> 'Message':
        """
        Create Message instance from serialized string.
        
        :param serialized_data: Serialized message data
        :type serialized_data: str
        :return: New Message instance
        :rtype: Message
        """
        # Use orjson for deserialization
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, orjson.loads, serialized_data)
        
        # Convert strings back to proper types
        data['sender'] = Sender(data['sender'])
        data['type'] = MessageType(data['type'])
        data['date'] = datetime.fromisoformat(data['date'])
        data['metadata_ids'] = set(data['metadata_ids'])
        data['file_ids'] = set(data['file_ids'])

        # Decode any bytes in annotations
        if data.get('annotations'):
            data['annotations'] = cls._decode_bytes_in_dict(data['annotations'])
        
        return cls(**data)

    async def token_length(self) -> int:
        """Get the token length asynchronously"""
        tokens = await self.tokens()
        return len(tokens)