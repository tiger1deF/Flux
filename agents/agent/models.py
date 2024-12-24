"""
Core agent models and functionality.

This module provides the base Agent class and related utilities for building
and managing agents in the system. It includes functionality for:

- Agent state and configuration management
- Message handling and communication
- LLM integration
- Vector store management
- Logging and monitoring
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import logging
import asyncio
from datetime import datetime
from functools import wraps
from uuid import uuid4
import os
import json
from pathlib import Path
import aiofiles

from agents.messages.message import (
    Message, Sender
)

from agents.config.models import AgentConfig, AgentStatus, Logging

from agents.state.models import AgentState


from llm import (
    LLM,
    gemini_llm_async_inference,
)

from agents.monitor.logger import AgentLogger, AgentLogHandler
from agents.monitor.agent_logs import AgentLog

from agents.monitor.wrappers.logging import logging_agent_wrapper
from agents.monitor.wrappers.langfuse import langfuse_agent_wrapper
from agents.monitor.wrappers.default import default_agent_wrapper


def conditional_logging(capture_input = False):
    """
    Decorator for conditional logging of agent operations.
    
    :param capture_input: Whether to capture input parameters in logs
    :type capture_input: bool
    :return: Decorated function
    :rtype: Callable
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            if self.config.logging == Logging.DISABLED:
                return await default_agent_wrapper(func)(self, *args, **kwargs)
            
            elif self.config.logging == Logging.LANGFUSE:
                return await langfuse_agent_wrapper(func)(self, *args, **kwargs)

            else:
                return await logging_agent_wrapper(func)(self, *args, **kwargs)
                
        return wrapper
    return decorator


class Agent(BaseModel):
    """
    Base agent class providing core functionality for AI agents.
    
    :ivar name: Name of the agent
    :type name: str
    :ivar type: Type identifier for the agent
    :type type: str
    :ivar description: Optional description of the agent's purpose
    :type description: Optional[str]
    :ivar state: Current state of the agent
    :type state: AgentState
    :ivar config: Agent configuration
    :type config: AgentConfig
    :ivar source_agents: List of upstream agents
    :type source_agents: List[Agent]
    :ivar target_agents: List of downstream agents
    :type target_agents: List[Agent]
    :ivar llm: Language model interface
    :type llm: LLM
    :ivar logging: Whether logging is enabled
    :type logging: bool
    :ivar logger: Agent logger instance
    :type logger: Optional[AgentLogger]
    :ivar agent_log: Log of agent operations
    :type agent_log: AgentLog
    :ivar runtime_logger: Runtime logger instance
    :type runtime_logger: AgentLogger
    :ivar agent_status: Current status of the agent
    :type agent_status: AgentStatus
    :ivar session_id: Unique identifier for the session
    :type session_id: str
    """
    name: str = Field(default = "Agent")
    type: str = Field(default = "base")
    description: Optional[str] = None
    
    state: AgentState = Field(default_factory = AgentState)
    config: AgentConfig = Field(default_factory = AgentConfig)
   
    source_agents: List['Agent'] = Field(default_factory = list)
    target_agents: List['Agent'] = Field(default_factory = list)

    llm: LLM = Field(default_factory = lambda: LLM(gemini_llm_async_inference))
        
    logging: bool = Field(default = True)
    logger: Optional[AgentLogger] = None
    
    agent_log: AgentLog = Field(default = None)
    runtime_logger: AgentLogger = Field(default = None)
    agent_status: AgentStatus = Field(default = AgentStatus.IDLE)
    
    session_id: str = Field(default = str(uuid4()), exclude = True)
    
    
    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"
    
    def __init__(self, *args, **kwargs):
        """
        Initialize an agent instance.
        
        :param args: Positional arguments
        :param kwargs: Keyword arguments
        """
        super().__init__(*args, **kwargs)
        if self.logging:
            self.agent_log = AgentLog(
                session_id = self.session_id,
                agent_name = self.name,
                agent_type = self.type,
                agent_description = self.description,
                llm = self.llm,
                source_agents = [a.name for a in self.source_agents],
                target_agents = [a.name for a in self.target_agents]
            )
            
            self.runtime_logger = AgentLogger(self.name)
            self.runtime_logger.agent_handler = AgentLogHandler(self.agent_log)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            self.runtime_logger.agent_handler.setFormatter(formatter)
            self.runtime_logger.addHandler(self.runtime_logger.agent_handler)
    
    
    @property
    def text(self) -> str:
        """
        Get text representation of agent logs.
        
        :return: Formatted log text
        :rtype: str
        """
        return self.agent_log.text


    async def add_message(
        self, 
        message: Message, 
        message_id: Union[str, Any] = None,
    ) -> Message:
        """
        Add a message to the agent's message store.
        
        :param message: Message to add
        :type message: Message
        :param message_id: Optional identifier for the message
        :type message_id: Union[str, Any]
        :return: Added message
        :rtype: Message
        """
        # Use state's message store instead of local dict
        await self.state.add_message(message)
        return message


    async def get_messages(
        self, 
        limit: Optional[int] = None,
        ids: Optional[List[Union[str, Any]]] = None
    ) -> List[Message]:
        """
        Retrieve messages from the agent's history.
        
        :param limit: Maximum number of messages to return
        :type limit: Optional[int]
        :param ids: List of specific message IDs to retrieve
        :type ids: Optional[List[Union[str, Any]]]
        :return: List of messages
        :rtype: List[Message]
        """
        # Use state's message store for retrieval
        filter_dict = {}
        if ids:
            filter_dict["id"] = {"$in": ids}
            
        messages = await self.state.obtain_message_context(
            query = None,  # No query needed for direct retrieval
            context_config = self.config.context,
            filter = filter_dict,
            truncate = False
        )
        
        if limit:
            messages = messages[:limit]
            
        return messages


    @conditional_logging()
    async def send_message(
        self, 
        input_message: Message,
        state: Optional[AgentState] = None,
    ) -> Message:
        """
        Send a message to the agent and get a response.
        
        :param input_message: Message to send
        :type input_message: Message
        :param state: Optional state override
        :type state: Optional[AgentState]
        :return: Response message
        :rtype: Message
        """
        prompt = ""
        if self.llm.system_prompt:
            prompt += self.llm.system_prompt
        
        if self.config.task_prompt:
            prompt += self.config.task_prompt
        
        if self.state.context:
            prompt += self.state.context
        
        if input_message:
            prompt += input_message.content
        
        response = await self.llm(prompt)
        response_message = Message(
            content = response,
            sender = Sender.AI,
            agent_name = self.name,
        )
        
        return response_message

    
    def sync_send_message(
        self, 
        input_message: Message,
        state: Optional[AgentState] = None,
    ) -> Message:
        """
        Synchronous version of send_message.
        
        :param input_message: Message to send
        :type input_message: Message
        :param state: Optional state override
        :type state: Optional[AgentState]
        :return: Response message
        :rtype: Message
        """
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                self.send_message(
                    input_message = input_message,
                    state = state
                )
            )
        finally:
            loop.close()


    def sync_call(self, *args, **kwargs):
        """
        Synchronous call interface for the agent.
        
        :param args: Positional arguments
        :param kwargs: Keyword arguments
        :return: Response message
        :rtype: Message
        """
        if isinstance(args[0], str):
            message = Message(
                content=args[0],
                sender=Sender.USER
            )
            return self.sync_send_message(message, **kwargs)
        return self.sync_send_message(*args, **kwargs)


    async def async_call(self, *args, **kwargs):
        """
        Asynchronous call interface for the agent.
        
        :param args: Positional arguments
        :param kwargs: Keyword arguments
        :return: Response message
        :rtype: Message
        """
        if isinstance(args[0], str):
            message = Message(
                content=args[0],
                sender=Sender.USER
            )
            return await self.send_message(message, **kwargs)
        return await self.send_message(*args, **kwargs)


    def __call__(self, *args, **kwargs):
        """
        Call operator implementation for the agent.
        
        :param args: Positional arguments
        :param kwargs: Keyword arguments
        :return: Response message
        :rtype: Message
        """
        try:
            asyncio.get_running_loop()
            return self.async_call(*args, **kwargs)
        except RuntimeError:
            return self.sync_call(*args, **kwargs)
    
    
    async def clear_history(self) -> None:
        """
        Clear the agent's message history.
        """
        self.messages = []


    async def update_context(self, **kwargs) -> None:
        """
        Update the agent's context with new values.
        
        :param kwargs: Key-value pairs to update in context
        """
        self.state.context.update(kwargs)

    
    def __repr__(self) -> str:
        """
        Get string representation of the agent.
        
        :return: String representation
        :rtype: str
        """
        return f'Agent(state={self.state}, config={self.config}, num_messages={len(self.messages)})'
    
    
    def __str__(self) -> str:
        """
        Get string representation of the agent.
        
        :return: String representation
        :rtype: str
        """
        return f'Agent(state={self.state}, config={self.config}, num_messages={len(self.messages)})'


    async def serialize(self, path: Union[str, Path]) -> None:
        """
        Serialize agent to disk.
        
        :param path: Directory path to save serialized data
        :type path: Union[str, Path]
        """
        path = Path(path)
        path.mkdir(exist_ok=True)
        
        # Save core agent data
        agent_data = {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "session_id": self.session_id,
            "agent_status": self.agent_status.value,
            "config": self.config.dict()
        }
        
        async with aiofiles.open(path / "agent.json", "w") as f:
            await f.write(json.dumps(agent_data))
            
        # Serialize state (which includes message store, file store, etc)
        await self.state.serialize(path / "state")


    @classmethod
    async def deserialize(cls, path: Union[str, Path]) -> 'Agent':
        """
        Create Agent instance from serialized data.
        
        :param path: Directory path containing serialized data
        :type path: Union[str, Path]
        :return: New Agent instance
        :rtype: Agent
        """
        path = Path(path)
        
        # Load core agent data
        async with aiofiles.open(path / "agent.json", "r") as f:
            agent_data = json.loads(await f.read())
        
        # Create base agent instance
        agent = cls(
            name=agent_data["name"],
            type=agent_data["type"],
            description=agent_data["description"]
        )
        
        agent.session_id = agent_data["session_id"]
        agent.agent_status = AgentStatus(agent_data["agent_status"])
        agent.config = AgentConfig(**agent_data["config"])
        
        # Load state (which includes message store, file store, etc)
        await agent.state.deserialize(path / "state")
        
        return agent