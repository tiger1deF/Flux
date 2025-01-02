"""
Logging wrapper implementation for agent functions.

This module provides a wrapper that adds comprehensive logging functionality
to agent functions, including input/output tracking, timing, and error logging.
"""

from datetime import datetime   
import asyncio
from functools import wraps
import traceback

from agents.config.models import AgentStatus
from agents.storage.context import Context, ContextType
from agents.storage.message import Message, Sender, MessageType


def logging_agent_wrapper(func):
    """Wrapper for logging agent execution"""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            start_time = datetime.now()
            
            input_message = kwargs.get('input_message') or args[0]
            if not isinstance(input_message, Message):
                raise ValueError("Input Message class is required for agent!")
                
            self.agent_status = AgentStatus.RUNNING
            
            # Ingests data from message
            await self.state.ingest_message_data(input_message)
            await self.agent_log.log_input(input_message)
            
            # Execute main function
            result = await func(self, *args, **kwargs)
            end_time = datetime.now()
            
            if not isinstance(result, Message):
                raise ValueError("Output Message class is required for agent!")
            
            # Log output as context with proper format
            output_context = Context(
                type = ContextType.PROCESS,
                content = f"Input: {input_message.content}\nOutput: {result.content}",
                sender = result.sender,
                agent = self.name,
                context_type = "output",
                date = end_time,
                duration = (end_time - start_time).total_seconds()
            )
            await self.state.add_context(output_context)
            await self.agent_log.log_output(result)
            
            self.agent_status = AgentStatus.COMPLETED
            return result
            
        except Exception as e:
            self.agent_status = AgentStatus.FAILED
            error = traceback.format_exc()
            print(error)
            import sys
            sys.exit()
            
            # Log error as context
            error_context = Context(
                type = ContextType.ERROR,
                content = error,
                agent = self.name,
                context_type = "error",
                error_type = type(e).__name__,
                date = datetime.now()
            )
            await self.state.add_context(error_context)
            await self.agent_log.log_error(error)
            
    return wrapper