"""
Logging wrapper implementation for agent functions.

This module provides a wrapper that adds comprehensive logging functionality
to agent functions, including input/output tracking, timing, and error logging.
"""

from datetime import datetime   
import asyncio

from agents.config.models import AgentStatus

from agents.messages.message import Message, Sender, MessageType


def logging_agent_wrapper(func):
    """
    Wrapper that adds logging functionality to agent functions.
    
    Provides comprehensive logging including:
    - Input/output message tracking
    - Execution timing
    - Error logging
    - Intermediate message logging
    
    :param func: Agent function to wrap
    :type func: Callable
    :return: Wrapped function with logging
    :rtype: Callable
    """
    async def wrapper(self, *args, **kwargs):
        """
        Wrapper implementation that handles logging and message flow.
        
        :param self: Agent instance
        :param args: Positional arguments for the agent function
        :param kwargs: Keyword arguments for the agent function
        :return: Response message or error message
        :rtype: Message
        """
        logging_tasks = []
        start_time = datetime.now()
        
        if args and isinstance(args[0], Message):
            input_message = args[0]
        elif kwargs.get('input_message'):
            input_message = kwargs.get('input_message')
        else:
            raise ValueError("Input Message class is required for agent!")
            
        if input_message:
            input_message.type = MessageType.INPUT
            # Use state message store
            await self.state.add_message(input_message)
        else:
            raise ValueError("Input Message class is required for agent!")
                
        if state := kwargs.get('state'):
            self.state = state
                
        try:
            self.agent_status = AgentStatus.RUNNING
            
            logging_tasks.append(self.agent_log.log_input(input_message))
            
            # Execute main function
            result = await func(self, *args, **kwargs)
            end_time = datetime.now()
                
            if not isinstance(result, Message):
                raise ValueError("Output Message class is required for agent!")
            
            # Get messages from state store for logging
            messages = await self.state.obtain_message_context(
                query=None,
                context_config=self.config.context,
                truncate=False
            )
            for message in messages:
                logging_tasks.append(self.agent_log.log_message(message))
                    
            logging_tasks.append(self.agent_log.log_output(result))
            logging_tasks.append(
                self.runtime_logger.async_info(
                    f"{self.name} completed {func.__name__} in {end_time - start_time}"
                )
            )

            result.type = MessageType.OUTPUT
            # Use state message store
            await self.state.add_message(result)
            
            await asyncio.gather(*logging_tasks)
            
            self.agent_status = AgentStatus.COMPLETED
                
            return result
            
        except:
            end_time = datetime.now()
            import traceback
            error = traceback.format_exc()
            
            logging_tasks.append(
                self.agent_log.log_error(
                    f"{self.name} failed {func.__name__}: {error}"
                )
            )
            
            await asyncio.gather(*logging_tasks)
                
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