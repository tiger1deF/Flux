"""
Message models and utilities for agent communication.

This module provides the core message models used for communication between agents,
including message content, metadata, file attachments, and sender information.
"""

from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import Dict, Any, List, Union


class Sender(str, Enum):
    """
    Enumeration of possible message senders.
    
    :cvar USER: Message from a user
    :cvar AI: Message from an AI agent
    """
    USER = 'user'
    AI = 'ai'
    

class File(BaseModel):
    """
    Model for file attachments in messages.
    
    :ivar content: Content of the file
    :type content: Union[str, Any]
    :ivar description: Description of the file
    :type description: Union[str, None]
    :ivar path: Path to the file
    :type path: Union[str, None]
    :ivar annotations: Additional annotations for the file
    :type annotations: Dict[str, Any]
    """
    content: Union[str, Any] = Field(description = "File content")
    description: Union[str, None] = Field(description = "File description")
    path: Union[str, None] = Field(description = "File path")
    annotations: Dict[str, Any] = Field(default = {}, description = "File annotations")


    def to_dict(self) -> dict:
        """
        Convert file to dictionary representation.
        
        :return: Dictionary containing file data
        :rtype: dict
        """
        return {
            'content': self.content,
            'description': self.description,
            'path': self.path,
            'annotations': self.annotations
        }

    
    def from_dict(self, data: dict) -> 'File':
        """
        Create file instance from dictionary data.
        
        :param data: Dictionary containing file data
        :type data: dict
        :return: New file instance
        :rtype: File
        """
        return File(
            content = data['content'],
            description = data['description'],
            path = data['path'],
            annotations = data['annotations']
        )


    def summary(self) -> str:
        """
        Generate a text summary of the file.
        
        :return: Formatted summary string
        :rtype: str
        """
        return f"File: {self.path}\nDescription: {self.description}\nAnnotations: {self.annotations}"


class Message(BaseModel):
    """
    Model for messages exchanged between agents.
    
    :ivar sender: Entity that sent the message
    :type sender: Sender
    :ivar agent_name: Name of the sending agent
    :type agent_name: str
    :ivar date: Timestamp of the message
    :type date: datetime
    :ivar content: Text content of the message
    :type content: str
    :ivar metadata: Additional metadata for the message
    :type metadata: Dict[str, Any]
    :ivar files: File attachments
    :type files: List[File]
    :ivar annotations: Configuration and state annotations
    :type annotations: Dict[str, Any]
    """
    sender: Sender = Field(default = Sender.AI, description = "The sender of the message")
    agent_name: str = Field(default = None, description = "The name of the agent that sent the message")
    date: datetime = Field(default_factory = datetime.now, description = "The date of the message")
    
    content: str = Field(default = "", description = "Message string content")
    
    metadata: Dict[str, Any] = Field(default = {}, description = "Message metadata")
    files: List[File] = Field(default = [], description = "Message files")
    
    annotations: Dict[str, Any] = Field(default = [], description = "Message annotations for config/state items")
    
    
    async def to_json(self) -> str:
        """
        Convert message to JSON string.
        
        :return: JSON string representation
        :rtype: str
        """
        import json
        json_data = {
            'sender': self.sender.value(),
            'date': self.date.isoformat(),             
        }
        return json.dumps(json_data)
        
        
    async def read_json(self, json_str) -> None:
        """
        Update message from JSON string.
        
        :param json_str: JSON string containing message data
        :type json_str: str
        """
        import json
        data = json.loads(json_str)
        self.sender = Sender(data['sender'])
        self.date = datetime.fromisoformat(data['date'])
        
