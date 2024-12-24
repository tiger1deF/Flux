"""
Default agent wrapper implementation.

This module provides a basic wrapper for agent functions that handles message
tracking, state management, and error handling without additional logging.
"""

from datetime import datetime   


from agents.config.models import AgentStatus
from agents.messages.message import Message, Sender, MessageType


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
            input_message.type = MessageType.INPUT
            # Use state message store instead of messages dict
            await self.state.add_message(input_message)
        else:
            raise ValueError("Input Message class is required for agent!")
                
        if state := kwargs.get('state'):
            self.state = state
                
        try:
            self.agent_status = AgentStatus.RUNNING
            
            # Execute main function
            result = await func(self, *args, **kwargs)
            end_time = datetime.now()
            
            if not isinstance(result, Message):
               raise ValueError("Output Message class is required for agent!")
           
            result.type = MessageType.INTERMEDIATE
            # Use state message store
            await self.state.add_message(result)
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
                date = end_time,
                type = MessageType.ERROR
            )
            # Use state message store
            await self.state.add_message(error_message)
            
            return error_message
        
    return wrapper