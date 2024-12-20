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

from agents.messages.models import (
    Message, Sender
)

from agents.models import AgentState, AgentConfig, AgentStatus, Logging

from llm import (
    LLM,
    gemini_llm_async_inference,
    gemini_generate_embedding,
    BaseEmbeddingFunction
)

from agents.vectorstore.models import BaseVectorStore  
from agents.vectorstore.default.store import HNSWStore

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
    :ivar chat_agents: List of agents for chat interactions
    :type chat_agents: List[Agent]
    :ivar llm: Language model interface
    :type llm: LLM
    :ivar embed: Embedding function
    :type embed: BaseEmbeddingFunction
    :ivar vector_store: Vector store for embeddings
    :type vector_store: BaseVectorStore
    :ivar logging: Whether logging is enabled
    :type logging: bool
    :ivar logger: Agent logger instance
    :type logger: Optional[AgentLogger]
    :ivar messages: Dictionary of messages
    :type messages: Dict[str, Message]
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
    chat_agents: List['Agent'] = Field(default_factory = list)
    
    llm: LLM = Field(default_factory = lambda: LLM(gemini_llm_async_inference))
    embed: BaseEmbeddingFunction = Field(default_factory = lambda: BaseEmbeddingFunction(gemini_generate_embedding))
    vector_store: BaseVectorStore = Field(default = None)
        
    logging: bool = Field(default = True)
    logger: Optional[AgentLogger] = None
    
    messages: Dict[str, Message] = Field(default_factory = dict)
    
    agent_log: AgentLog = Field(default = None)
    runtime_logger: AgentLogger = Field(default = None)
    agent_status: AgentStatus = Field(default = AgentStatus.IDLE)
    
    session_id: str = Field(default = str(uuid4()), exclude = True)
    
    
    class Config:
        arbitrary_types_allowed = True

    
    def __init__(self, *args, **kwargs):
        """
        Initialize an agent instance.
        
        :param args: Positional arguments
        :param kwargs: Keyword arguments
        """
        super().__init__(*args, **kwargs)

        self.vector_store = HNSWStore(embedding_function = self.embed)
        
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
        Add a message to the agent's message history.
        
        :param message: Message to add
        :type message: Message
        :param message_id: Optional identifier for the message
        :type message_id: Union[str, Any]
        :return: Added message
        :rtype: Message
        """
        if message_id is None:
            message_id = datetime.now()
        
        self.state.messages[message_id] = message
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
        if limit:
            return self.state.messages.items()[:limit]
        elif ids:
            return [self.state.messages[id] for id in ids]
        else:
            return self.state.messages.items()


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
            prompt += str(self.state.context)
        
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
                    input_message=input_message,
                    state=state
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
        self.state.messages = []


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
        return f'Agent(state={self.state}, config={self.config}, num_messages={len(self.state.messages)})'
    
    
    def __str__(self) -> str:
        """
        Get string representation of the agent.
        
        :return: String representation
        :rtype: str
        """
        return f'Agent(state={self.state}, config={self.config}, num_messages={len(self.state.messages)})'