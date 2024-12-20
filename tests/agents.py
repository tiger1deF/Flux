"""
Test suite for agent functionality.

This module provides tests for various agent configurations and logging modes,
including Langfuse integration, basic logging, and default operation without logging.
"""


def test_base_agent():
    """
    Test basic agent functionality with different logging configurations.
    
    Tests three configurations:
    1. Agent with Langfuse logging
    2. Agent with basic logging enabled
    3. Agent with logging disabled
    
    Each configuration tests message sending and response handling.
    """
    import asyncio
    from agents import (
        Agent,
        HNSWStore,
        Message,
        AgentConfig,
        AgentState,
        Logging
    )

    from llm import (
        BaseEmbeddingFunction, 
        gemini_generate_embedding,
        LLM,
        pulse_llm_async_inference
    )
    
    from agents import HNSWStore
    
    from langfuse import Langfuse
    langfuse = Langfuse()
    
    llm = LLM(
        pulse_llm_async_inference, 
        input_tokens = 2000, 
        max_tokens = 2000,
        system_prompt = "You must respond like a pirate"
    )
    
    embedder = BaseEmbeddingFunction(gemini_generate_embedding)
    store = HNSWStore(embedding_function = embedder)
        
    print(f'Testing langfuse langchain agent')
    config = AgentConfig(
        task_prompt = "You must respond like a pirate",
        logging = Logging.LANGFUSE
    )
    
    agent = Agent(
        name = "Test Agent",
        llm = llm,
        vector_store = store,
        config = config
    )
    
    message = Message(content = "Hello, world!")
    async def test_agent():
        """
        Test helper function for running agent tests.
        
        Sends a test message to the agent and handles the response.
        """
        response = await agent(message)
        
    asyncio.run(test_agent())
  
    
    print(f'Testing logging agent')
    config = AgentConfig(
        task_prompt = "You must respond like a pirate",
        logging = Logging.ENABLED
    )
    
    agent = Agent(
        name = "Test Agent",
        llm = llm,
        vector_store = store,
        config = config
    )
    
    message = Message(content = "Hello, world!")
    
    async def test_agent():
        """
        Test helper function for running agent tests.
        
        Sends a test message to the agent and prints the log output.
        """
        response = await agent(message)
        print(agent.text())
     
    asyncio.run(test_agent())
    
    
    print(f'Testing default agent')
    config = AgentConfig(
        task_prompt = "You must respond like a pirate",
        logging = Logging.DISABLED
    )
    
    agent = Agent(
        name = "Test Agent",
        llm = llm,
        vector_store = store,
        config = config
    )
    
    message = Message(content = "Hello, world!")
    
    async def test_agent():
        """
        Test helper function for running agent tests.
        
        Sends a test message to the agent and prints the response.
        """
        response = await agent(message)
        print(response)
     
    asyncio.run(test_agent())
    