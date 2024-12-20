"""
ReAct agent implementation for tool-based reasoning and action.

This module provides a ReAct (Reasoning + Acting) agent implementation that can
use tools to solve tasks through a cycle of thought, action, and observation.
"""

from typing import List, Dict, Any, Optional, Callable, get_type_hints, Union
from pydantic import BaseModel, Field
import inspect
import logging
import asyncio
from datetime import datetime
from agents.messages.models import Message, Sender, File
from agents.models import AgentState, AgentConfig

from llm import (
    LLM,
    gemini_llm_inference,
    BaseEmbeddingFunction,
    base_local_embedder,
)

from agents.vectorstore.models import BaseEmbeddingStore  
from agents.vectorstore.default.store import HNSWStore

from agents.tools.models import Tool, ToolParameter
from agents.monitor.logger import AgentLogger, AgentLogHandler
from agents.monitor.agent_logs import AgentLog

from agents.agent.models import Agent


class ReactAgent(Agent):
    """
    ReAct agent that combines reasoning and acting through tool use.
    
    :ivar type: Type identifier for the agent
    :type type: str
    :ivar tools: List of available tools
    :type tools: List[Tool]
    """
    type: str = Field(default = "react")
    tools: List[Tool] = Field(default_factory = list)
    
    class Config:
        arbitrary_types_allowed = True
        
       
    async def add_message(
        self, 
        message: Message, 
        message_id: Union[str, Any] = datetime.now,
        message_type: Union[str, Any] = None,
    ) -> Message:
        """
        Add a message to the agent's history with appropriate logging.
        
        :param message: Message to add
        :type message: Message
        :param message_id: Identifier for the message
        :type message_id: Union[str, Any]
        :param message_type: Type of message (input/output/other)
        :type message_type: Union[str, Any]
        :return: Added message
        :rtype: Message
        """
        if message_type == 'input':
            await self.logger.log_input(message)
            
        elif message_type == 'output':
            await self.logger.log_output(message)
            
        else:
            await self.logger.log_message(message)

        self.state.messages[message_id] = message


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


    async def send_message(
        self, 
        input_message: Message,
        external_context: Optional[str] = None
    ) -> Message:
        """
        Process an input message and generate a response.
        
        :param input_message: Message to process
        :type input_message: Message
        :param external_context: Additional context to include
        :type external_context: Optional[str]
        :return: Response message
        :rtype: Message
        """
        prompt = ""
        if self.system_prompt:
            prompt += self.system_prompt
        
        if external_context:
            prompt += external_context
        
        if input_message:
            prompt += input_message.content
        
        self.add_message(
            data = input_message,
            message_type = 'input'
        )
        
        response = await self.llm(prompt)
        response_message = Message(
            content = response,
            sender = Sender.AI,
            agent_name = self.metadata.name,
        )
        await self.runtime_logger.log_message(f"Agent {self.name} Finished Processing!") 
        
        self.add_message(
            data = response_message,
            message_type = 'output' 
        )
        
        return response_message


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


    async def get_context(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the agent's context.
        
        :param key: Key to retrieve
        :type key: str
        :param default: Default value if key not found
        :type default: Any
        :return: Value from context
        :rtype: Any
        """
        return self.state.context.get(key, default)
    
    
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


    async def add_tool(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        static_params: Optional[Dict[str, Any]] = None
    ) -> Tool:
        """
        Add a tool to the agent's available tools.
        
        :param func: Function to wrap as a tool
        :type func: Callable
        :param name: Name for the tool
        :type name: Optional[str]
        :param description: Description of the tool
        :type description: Optional[str]
        :param static_params: Parameters to always pass to the tool
        :type static_params: Optional[Dict[str, Any]]
        :return: Created tool
        :rtype: Tool
        """
        # Get function signature and type hints
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        # Get docstring and parse it
        doc = inspect.getdoc(func) or ""
        description = description or doc.split('\n\n')[0]
        
        # Parse docstring for parameter descriptions
        param_docs = {}
        if doc:
            lines = doc.split('\n')
            current_param = None
            for line in lines:
                line = line.strip()
                if line.startswith(':param '):
                    current_param = line[7:].split(':')[0].strip()
                    param_docs[current_param] = line.split(':', 2)[2].strip()
                elif line.startswith('Args:'):
                    continue
                elif current_param and line and line[0] == ' ':
                    param_docs[current_param] += ' ' + line.strip()
        
        # Create tool parameters, excluding static params
        parameters = []
        static_params = static_params or {}
        
        for param_name, param in signature.parameters.items():
            # Skip parameters that are provided as static
            if param_name in static_params:
                continue
                
            param_type = type_hints.get(param_name, Any)
            required = param.default == param.empty
            default = None if param.default == param.empty else param.default
            
            parameters.append(
                ToolParameter(
                    name = param_name,
                    type = param_type,
                    required = required,
                    default = default,
                    description = param_docs.get(param_name)
                )
            )
   
        tool = Tool(
            name = name or func.__name__,
            description = description,
            parameters = parameters,
            static_params = static_params,
            function = func,
            return_type = type_hints.get('return')
        )
       
        self.tools.append(tool)
        return tool


    async def get_tools(self) -> List[Tool]:
        """
        Get list of available tools.
        
        :return: List of tools
        :rtype: List[Tool]
        """
        return self.state.tools


    async def get_tool(self, name: str) -> Optional[Tool]:
        """
        Get a specific tool by name.
        
        :param name: Name of the tool to retrieve
        :type name: str
        :return: Tool if found, None otherwise
        :rtype: Optional[Tool]
        """
        for tool in self.state.tools:
            if tool.name == name:
                return tool
        return None
    

    async def send_react_message(
        self,
        content: Union[str, Message],
        sender: Sender = Sender.USER,
        max_iterations: int = 5,
        **metadata
    ) -> Message:
        """
        Process a message using the ReAct cycle of thought, action, and observation.
        
        :param content: Message content to process
        :type content: Union[str, Message]
        :param sender: Entity sending the message
        :type sender: Sender
        :param max_iterations: Maximum number of ReAct cycles
        :type max_iterations: int
        :param metadata: Additional metadata for the message
        :return: Final response message
        :rtype: Message
        """
        logger.debug(f"Full message: {content}")
        logger.debug(f"Available tools: {[t.name for t in self.tools]}")
        
        # Add user message to history
        user_message = await self.add_message(content, sender = sender, **metadata)
        logger.debug(f"Added user message: {user_message}")
        
        # Build context with available tools
        tools_context = "\n".join([
            "\n".join([  
                f"Tool: {tool.name}",
                f"Description: {tool.description}",
                "Parameters:",
                "\n".join([
                    f"  - {p.name}: {p.type.__name__}"
                    f"{' (optional)' if not p.required else ''}"
                    f"{f' = {p.default}' if p.default is not None else ''}"
                    f"{f' - {p.description}' if p.description else ''}"
                    for p in tool.parameters
                ]),
                f"Returns: {tool.return_type.__name__ if tool.return_type else 'None'}"
            ])
            for tool in self.tools
        ])
        logger.debug(f"Built tools context:\n{tools_context}")

        # Build system prompt
        system_prompt = f"""You are a helpful AI assistant with access to tools. For each user message, you can either:
1. Respond directly if no tools are needed
2. Use tools in this format:
   Thought: Explain your reasoning
   Action: tool_name(param1=value1, param2=value2)
   Observation: <wait for tool result>
   ... (repeat if needed)
   Response: Final response incorporating tool results

Available Tools:
{tools_context}

Rules:
- Only use listed tools with exact parameter names
- Always format tool calls as shown above
- You can use multiple tools before responding
- Stop when you have enough information to respond
- Keep thoughts concise and focused
"""
        logger.debug(f"System prompt built with length: {len(system_prompt)}")

        # Initialize response building
        iterations = 0
        final_response = ""
        final_metadata = {}
        
        logger.info(f"Starting tool interaction loop (max iterations: {max_iterations})")
        while iterations < max_iterations:
            logger.debug(f"Starting iteration {iterations + 1}/{max_iterations}")
            
            # Get recent context
            current_context = "\n".join(
                [str(m) for m in self.state.messages[-5:]]
            )
            logger.debug(f"Current context length: {len(current_context)}")
            
            # Get AI response
            full_prompt = f"{system_prompt}\n\nContext:\n{current_context}\n\nCurrent iteration: {iterations + 1}/{max_iterations}"
            logger.debug(f"Sending prompt to LLM (length: {len(full_prompt)})")
            
            response = await self.llm(full_prompt)
            logger.debug(f"Received LLM response: {response[:200]}...")
            
            # Check response type
            if "Thought:" not in response:
                logger.info("Direct response received (no tool use)")
                final_response = response
                break
                
            # Parse ReAct components
            logger.debug("Parsing ReAct components")
            components = response.split("\n")
            for i, line in enumerate(components):
                if line.startswith("Action: "):
                    # Parse tool call
                    tool_call = line[8:].strip()
                    tool_name = tool_call[:tool_call.index("(")]
                    logger.info(f"Tool call detected: {tool_name}")
                    
                    tool = await self.get_tool(tool_name)
                    if not tool:
                        logger.error(f"Tool not found: {tool_name}")
                        final_response = f"Error: Tool {tool_name} not found"
                        break
                    
                    # Parse parameters
                    try:
                        params_str = tool_call[tool_call.index("(")+1:tool_call.rindex(")")]
                        params = {}
                        if params_str:
                            for param in params_str.split(","):
                                key, value = param.split("=")
                                key = key.strip()
                                value = eval(value.strip())
                                params[key] = value
                        logger.debug(f"Parsed parameters: {params}")
                        
                    except Exception as e:
                        logger.error(f"Error parsing tool parameters: {e}")
                        components.insert(i + 1, f"Observation: Error parsing parameters: {str(e)}")
                        continue
                    
                    # Execute tool
                    try:
                        logger.info(f"Executing tool {tool_name} with params: {params}")
                        result = await tool(**params)
                        logger.debug(f"Tool execution result: {result}")
                        
                        # Handle different result types
                        if isinstance(result, tuple) and len(result) == 2:
                            output_str, meta = result
                            logger.debug(f"Tool returned string and metadata: {output_str[:100]}..., {meta}")
                            final_metadata.update(meta)
                        elif isinstance(result, dict):
                            output_str = "\n".join(f"{k}: {v}" for k, v in result.items())
                            logger.debug(f"Tool returned dictionary: {result}")
                            final_metadata.update(result)
                        else:
                            output_str = str(result)
                            logger.debug(f"Tool returned simple value: {output_str}")
                            
                        components.insert(i + 1, f"Observation: {output_str}")
                        
                    except Exception as e:
                        logger.error(f"Error executing tool: {e}", exc_info=True)
                        components.insert(i + 1, f"Observation: Error executing tool: {str(e)}")
                
                elif line.startswith("Response: "):
                    logger.info("Final response found in components")
                    final_response = line[10:].strip()
                    for remaining in components[i+1:]:
                        if remaining.strip():
                            final_response += "\n" + remaining.strip()
                    break
            
            if final_response:
                logger.info("Response complete, breaking loop")
                break
                
            iterations += 1
            logger.debug(f"Completed iteration {iterations}")
        
        # Handle final response
        if not final_response and final_metadata:
            logger.info("No text response but metadata present, creating metadata summary")
            final_response = "Processed with the following results:\n" + \
                           "\n".join(f"{k}: {v}" for k, v in final_metadata.items())
        
        # Create final message
        logger.info("Creating final message")
        logger.debug(f"Final response: {final_response}")
        logger.debug(f"Final metadata: {final_metadata}")
        
        return await self.send_message(
            final_response or "I apologize, but I was unable to complete the task successfully.",
            Sender.AI,
            **final_metadata
        )
    

    async def __call__(
        self, 
        message: str,
        max_iterations: int = 5,
        **metadata
    ) -> Message:
        
        
        return await self.send_message_with_tools(
            data = message,
            sender = Sender.USER,
            max_iterations = max_iterations,
            **metadata
        )
    