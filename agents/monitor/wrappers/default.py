"""
Default agent wrapper implementation.

This module provides a basic wrapper for agent functions that handles message
tracking, state management, and error handling without additional logging.
"""

from datetime import datetime   


from agents.config.models import AgentStatus
from agents.storage.context import Context, ContextType
from agents.storage.message import Message, Sender, MessageType


def default_agent_wrapper(func):
    """
    Default wrapper for agent functions.
    
    Provides basic message tracking, state management, and error handling.
    Does not include additional logging or monitoring.
    
    :param func: Agent function to wrap
    :type func: Callable
    :return: Wrapped function
    :rtype: Callable
    """
    async def wrapper(self, *args, **kwargs):
        """
        Wrapper implementation that handles message flow and state.
        
        :param self: Agent instance
        :param args: Positional arguments for the agent function
        :param kwargs: Keyword arguments for the agent function
        :return: Response message or error message
        :rtype: Message
        """
        start_time = datetime.now()
        
        input_message = kwargs.get('input_message') or args[0]
        if not isinstance(input_message, Message):
            raise ValueError("Input Message class is required for agent!")
                
        try:
            self.agent_status = AgentStatus.RUNNING
            
            # Ingests data from message
            await self.state.ingest_message_data(input_message)
            
            # Execute main function
            result = await func(self, *args, **kwargs)
            end_time = datetime.now()
            
            if not isinstance(result, Message):
                raise ValueError("Output Message class is required for agent!")
            
            # Store result as context with proper format
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
            
            self.agent_status = AgentStatus.COMPLETED
            return result
            
        except Exception as e:
            import traceback
            error = traceback.format_exc()
            
            # Store error as context
            error_context = Context(
                type = ContextType.ERROR,
                content = error,
                agent = self.name,
                context_type = "error",
                error_type = type(e).__name__,
                date = datetime.now()
            )
            await self.state.add_context(error_context)
            
            self.agent_status = AgentStatus.FAILED
            raise
        
    return wrapper