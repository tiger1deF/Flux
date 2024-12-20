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

from langfuse import Langfuse

from agents.models import AgentStatus
from agents.messages.models import Message, Sender


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
        # Initialize Langfuse client
        langfuse = Langfuse()
        trace = None
        logging_tasks = []
        start_time = datetime.now()
    
        if args and isinstance(args[0], Message):
            input_message = args[0]
        elif kwargs.get('input_message'):
            input_message = kwargs.get('input_message')
        else:
            raise ValueError("Input Message class is required for agent!")
                
        state = kwargs.get('state')
        if state:
            self.state = state
                
        self.agent_status = AgentStatus.RUNNING
        
        await self.agent_log.log_input(input_message)
        
        try:
            # Execute main function
            result = await func(self, *args, **kwargs)
            end_time = datetime.now()
            
            if not isinstance(result, Message):
                raise ValueError("Output Message class is required for agent!")
                            
            for message in self.messages:
                logging_tasks.append(self.agent_log.log_message(message))
            
            await self.agent_log.log_output(result)
            
            logging_tasks.append(
                self.runtime_logger.async_info(
                    f"{self.name} completed {func.__name__} in {end_time - start_time}"
                )
            )

            self.output_messages[end_time] = result
            
            await asyncio.gather(*logging_tasks)
                        
            self.agent_status = AgentStatus.COMPLETED
            return result
            
        except Exception as e:
            end_time = datetime.now()
            error = traceback.format_exc()
            
            # Log error before updating trace
            await self.agent_log.log_error(
                f"{self.name} failed {func.__name__}: {error}"
            )
            
            await asyncio.gather(*logging_tasks)
                
            self.agent_status = AgentStatus.FAILED
            error_message = Message(
                content = error,
                sender = Sender.AI,
                agent_name = self.name,
                date = end_time
            )
            
            self.error_messages[end_time] = error_message
                        
            return error_message
        
        finally:
            langfuse.trace(
                name = f"{self.name}:{func.__name__}",
                input = self.agent_log.input_text(),
                output = self.agent_log.output_text(),
                session_id = self.session_id,
                metadata = {
                    "agent_type": self.type
                }
            )
        
    return wrapper