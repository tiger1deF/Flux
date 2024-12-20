"""
Agent logging implementation with async support.

This module provides logging handlers and loggers specifically designed for agents,
supporting both synchronous and asynchronous logging operations.
"""

import logging
from typing import Optional
from dataclasses import dataclass
import asyncio
from functools import partial

from agents.monitor.agent_logs import AgentLog


@dataclass
class AgentLogHandler(logging.Handler):
    """
    Custom logging handler for agent logs.
    
    Extends the standard logging.Handler to support async logging operations
    and integration with AgentLog storage.
    
    :ivar agent_log: Log storage for the agent
    :type agent_log: AgentLog
    """
    agent_log: AgentLog
    
    async def async_emit(self, record: logging.LogRecord):
        """
        Asynchronously emit a log record.
        
        :param record: Log record to emit
        :type record: logging.LogRecord
        """
        try:
            msg = self.format(record)
            if self.agent_log.agent_logs is None:
                self.agent_log.agent_logs = msg + "\n"
            else:
                self.agent_log.agent_logs += msg + "\n"
                
        except Exception:
            self.handleError(record)
    
    def emit(self, record: logging.LogRecord):
        """
        Emit a log record, handling both async and sync contexts.
        
        Creates or uses existing event loop to handle async emission.
        
        :param record: Log record to emit
        :type record: logging.LogRecord
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_running():
            loop.create_task(self.async_emit(record))
        else:
            loop.run_until_complete(self.async_emit(record))


class AgentLogger(logging.Logger):
    """
    Custom logger for agents with async support.
    
    Extends the standard Logger to provide async logging methods and
    agent-specific handler management.
    
    :ivar agent_handler: Handler for agent-specific logging
    :type agent_handler: Optional[AgentLogHandler]
    """
    
    def __init__(self, agent_name: str):
        """
        Initialize agent logger with console handler.
        
        :param agent_name: Name of the agent for logging
        :type agent_name: str
        """
        super().__init__(agent_name)
        self.agent_handler: Optional[AgentLogHandler] = None

        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        console_handler.setFormatter(formatter)
        self.addHandler(console_handler)

    async def async_log(
        self, 
        level: int, 
        msg: str, 
        *args,      
        **kwargs
    ):
        """
        Asynchronously log a message at specified level.
        
        :param level: Logging level
        :type level: int
        :param msg: Message to log
        :type msg: str
        :param args: Additional positional arguments
        :param kwargs: Additional keyword arguments
        """
        if self.isEnabledFor(level):
            record = self.makeRecord(
                self.name, level, "(unknown file)", 0, msg, args, None
            )
            if self.agent_handler:
                await self.agent_handler.async_emit(record)
            
            # Handle other handlers synchronously
            for handler in self.handlers:
                if handler != self.agent_handler:
                    handler.emit(record)

    async def async_debug(self, msg: str, *args, **kwargs):
        """
        Asynchronously log a debug message.
        
        :param msg: Debug message
        :type msg: str
        :param args: Additional positional arguments
        :param kwargs: Additional keyword arguments
        """
        await self.async_log(logging.DEBUG, msg, *args, **kwargs)

    async def async_info(self, msg: str, *args, **kwargs):
        """
        Asynchronously log an info message.
        
        :param msg: Info message
        :type msg: str
        :param args: Additional positional arguments
        :param kwargs: Additional keyword arguments
        """
        await self.async_log(logging.INFO, msg, *args, **kwargs)

    async def async_warning(self, msg: str, *args, **kwargs):
        """
        Asynchronously log a warning message.
        
        :param msg: Warning message
        :type msg: str
        :param args: Additional positional arguments
        :param kwargs: Additional keyword arguments
        """
        await self.async_log(logging.WARNING, msg, *args, **kwargs)

    async def async_error(self, msg: str, *args, **kwargs):
        """
        Asynchronously log an error message.
        
        :param msg: Error message
        :type msg: str
        :param args: Additional positional arguments
        :param kwargs: Additional keyword arguments
        """
        await self.async_log(logging.ERROR, msg, *args, **kwargs)

    async def attach_agent(self, agent_log: AgentLog):
        """
        Attach an agent log handler to the logger.
        
        :param agent_log: Agent log storage to attach
        :type agent_log: AgentLog
        """
        if self.agent_handler is not None:
            await self.async_warning("Agent handler already attached. Detaching previous handler.")
            await self.detach_agent()
            
        # Create and attach the agent handler
        self.agent_handler = AgentLogHandler(agent_log)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        self.agent_handler.setFormatter(formatter)
        self.addHandler(self.agent_handler)

    async def detach_agent(self) -> Optional[AgentLog]:
        """
        Detach the current agent log handler.
        
        :return: The detached agent log, if any
        :rtype: Optional[AgentLog]
        """
        if self.agent_handler is None:
            return None
            
        agent_log = self.agent_handler.agent_log
        self.removeHandler(self.agent_handler)
        self.agent_handler = None
        
        return agent_log


