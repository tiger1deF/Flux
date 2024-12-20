"""
Default agent wrapper implementation.

This module provides a basic wrapper for agent functions that handles message
tracking, state management, and error handling without additional logging.
"""

from datetime import datetime   
import asyncio

from agents.models import AgentStatus
from agents.messages.models import Message, Sender


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
        
        input_message = kwargs.get('input_message')
        if not input_message:
            input_message = args[0]
            
        if input_message:
            self.state.input_messages[start_time] = input_message
        else:
            raise ValueError("Input Message class is required for agent!")
                
        state = kwargs.get('state')
        if state:
            self.state = state
                
        try:
            self.agent_status = AgentStatus.RUNNING
            
            # Execute main function
            result = await func(self, *args, **kwargs)
            end_time = datetime.now()
            
            if not isinstance(result, Message):
               raise ValueError("Output Message class is required for agent!")
           
            self.state.output_messages[end_time] = result
            self.agent_status = AgentStatus.COMPLETED
                
            return result
            
        except:
            end_time = datetime.now()
            import traceback
            error = traceback.format_exc()
                
            self.agent_status = AgentStatus.FAILED
            error_message = Message(
                content = error,
                sender = Sender.AI,
                agent_name = self.name,
                date = end_time
            )
            self.state.error_messages[end_time] = error_message
            
            return error_message
        
    return wrapper