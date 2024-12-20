"""
Vector chat agent implementation with embedding-based context management.

This module provides a chat agent implementation that uses vector embeddings
for context management and message history, supporting both synchronous and
asynchronous operations.
"""

from typing import List, Optional, Union, Any, Dict
from uuid import uuid4
from datetime import datetime
import asyncio
import logging
from pydantic import Field

from agents.models import (
    AgentConfig, AgentState, AgentStatus
)

from agents.messages.models import (
    Sender, Message
)

from agents.monitor.logger import (
    AgentLogger, AgentLogHandler
)

from agents.monitor.agent_logs import (
    AgentLog
)

from agents.agent.models import (
    Agent
)

from agents.vectorstore.models import (
    BaseVectorStore
)

from agents.vectorstore.models import BaseVectorStore  
from agents.vectorstore.default.store import HNSWStore

from llm import (
    LLM, BaseEmbeddingFunction, gemini_generate_embedding, gemini_llm_async_inference
)


class VectorChatAgent(Agent):
    """
    Chat agent with vector-based context management.
    
    This agent uses vector embeddings to store and retrieve context,
    supporting both sync and async operations with configurable logging.
    
    :ivar type: Type identifier for the agent
    :type type: str
    :ivar description: Optional description of the agent
    :type description: Optional[str]
    :ivar state: Agent state management
    :type state: AgentState
    :ivar config: Agent configuration
    :type config: AgentConfig
    :ivar chat_agents: List of connected chat agents
    :type chat_agents: List[Agent]
    :ivar llm: Language model for inference
    :type llm: LLM
    :ivar embed: Embedding function
    :type embed: BaseEmbeddingFunction
    :ivar vector_store: Vector storage for embeddings
    :type vector_store: BaseVectorStore
    :ivar logging: Whether logging is enabled
    :type logging: bool
    :ivar logger: Optional logger instance
    :type logger: Optional[AgentLogger]
    :ivar messages: Message history
    :type messages: Dict[str, Message]
    :ivar agent_log: Agent logging instance
    :type agent_log: Optional[AgentLog]
    :ivar runtime_logger: Runtime logger instance
    :type runtime_logger: Optional[AgentLogger]
    :ivar agent_status: Current agent status
    :type agent_status: AgentStatus
    :ivar session_id: Unique session identifier
    :type session_id: str
    """
    
    type: str = Field(default = "chat")
    description: Optional[str] = None
    
    state: AgentState = Field(default_factory = AgentState)
    config: AgentConfig = Field(default_factory = AgentConfig)
   
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
        
        self.messages[message_id] = message
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
            return self.messages.items()[:limit]
        elif ids:
            return [self.messages[id] for id in ids]
        else:
            return self.messages.items()


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