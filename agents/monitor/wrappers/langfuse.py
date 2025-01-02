"""
Langfuse integration wrapper for agent functions.

This module provides a wrapper that integrates Langfuse monitoring and tracing
functionality into agent functions, enabling detailed tracking and analysis of
agent operations through the Langfuse platform.
"""

from datetime import datetime   
import asyncio
from functools import wraps
import traceback

import langfuse

from agents.config.models import AgentStatus

from agents.storage.message import Message
from agents.storage.context import Context, ContextType

from langfuse import Langfuse
langfuse = Langfuse()


def langfuse_agent_wrapper(func):
    """
    Wrapper that adds Langfuse monitoring to agent functions.
    
    Provides comprehensive monitoring including:
    - Input/output message tracking
    - Execution timing
    - Error logging
    - Session tracking
    - Agent metadata collection
    - Langfuse trace generation
    
    :param func: Agent function to wrap
    :type func: Callable
    :return: Wrapped function with Langfuse monitoring
    :rtype: Callable
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        """
        Wrapper implementation that handles Langfuse monitoring and message flow.
        
        :param self: Agent instance
        :param args: Positional arguments for the agent function
        :param kwargs: Keyword arguments for the agent function
        :return: Response message or error message
        :rtype: Message
        """
        try:
            start_time = datetime.now()
            input_message = kwargs.get('input_message') or args[0]
            if not isinstance(input_message, Message):
                raise ValueError("Input Message class is required for agent!")
                
            self.agent_status = AgentStatus.RUNNING
            
            # Ingests data from messages
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
            error = traceback.format_exc()
            
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
            
            self.agent_status = AgentStatus.FAILED
            raise
            
        finally:
            # Add Langfuse trace
            input_text = await self.agent_log.input_text()
            output_text = await self.agent_log.output_text()
            
            langfuse.trace(
                name = f"{self.name}:{func.__name__}",
                input = input_text,
                output = output_text,
                session_id = self.session_id,
                metadata = {
                    "agent_type": self.type
                }
            )
        
    return wrapper