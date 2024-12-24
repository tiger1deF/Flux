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
        Message,
        AgentConfig,
        Logging
    )

    from llm import (
        LLM,
        pulse_llm_async_inference
    )
    
    llm = LLM(
        pulse_llm_async_inference, 
        input_tokens = 2000, 
        max_tokens = 2000,
        system_prompt = "You must respond like a pirate"
    )
        
    print(f'Testing langfuse langchain agent')
    config = AgentConfig(
        task_prompt = "You must respond like a pirate",
        logging = Logging.LANGFUSE
    )
    
    agent = Agent(
        name = "Test Agent",
        llm = llm,
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
    
    
def test_chat_agents():
    """
    Test chat agents with different logging configurations.
    """
    import asyncio
    from agents import (
        Message,
        AgentConfig,
        Logging
    )

    from agents.agent.chat.vector import VectorChatAgent

    from llm import (
        LLM,
        pulse_llm_async_inference,
        BaseEmbeddingFunction,
        gemini_generate_embedding
    )
    
    from agents import HNSWStore
   
    llm = LLM(
        pulse_llm_async_inference, 
        input_tokens = 2000, 
        max_tokens = 2000,
        system_prompt = "You must respond like a pirate"
    )
        
    print(f'Testing langfuse langchain agent')
    config = AgentConfig(
        task_prompt = "You must respond like a pirate",
        logging = Logging.LANGFUSE
    )
    
    embedding_function = BaseEmbeddingFunction(
        gemini_generate_embedding
    )
    
    agent = VectorChatAgent(
        name = "Test Agent",
        llm = llm,
        config = config,
        vector_store = HNSWStore(embedding_function = embedding_function)
    )
    
    message = Message(content = "Hello, world!")
    async def test_agent():
        """
        Test helper function for running agent tests.
        
        Sends a test message to the agent and handles the response.
        """
        response = await agent(message)
        
    asyncio.run(test_agent())

def test_agent_serialization():
    """
    Test agent serialization and deserialization.
    
    Tests:
    1. Message store serialization
    2. File store serialization
    3. Metadata store serialization
    4. Core agent data serialization
    5. Complete state restoration
    """
    import asyncio
    from pathlib import Path
    import shutil
    import polars as pl
    
    from agents import Agent
    from agents.messages.message import Message, Sender
    from agents.messages.metadata import Metadata

    async def _run_test():
        # Create test agent
        agent = Agent(name="TestAgent", type="test")
        
        # Add some test data
        test_message = Message(
            content="Hello, this is a test message",
            sender=Sender.USER
        )
        await agent.add_message(test_message)
        
        # Create a test file
        test_file = await agent.state.create_file(
            "test.txt",
            content="Test file content"
        )
        
        # Add some test metadata
        test_metadata = Metadata(
            name="test_meta",
            description="Test metadata",
            data=pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        )
        await agent.state.add_metadata(test_metadata)
        
        # Get initial store lengths
        orig_message_len = len(agent.state.message_store.data)
        orig_file_len = len(agent.state.file_store.data)
        orig_metadata_len = len(agent.state.metadata_store.data)
        
        # Serialize agent
        test_path = Path("test_serialization")
        await agent.serialize(test_path)
        
        # Deserialize to new agent
        loaded_agent = await Agent.deserialize(test_path)
        
        # Verify data
        assert loaded_agent.name == agent.name
        assert loaded_agent.type == agent.type
        
        # Compare store lengths
        loaded_message_len = len(loaded_agent.state.message_store.data)
        loaded_file_len = len(loaded_agent.state.file_store.data)
        loaded_metadata_len = len(loaded_agent.state.metadata_store.data)
       
        assert loaded_message_len == orig_message_len, f"Message store length mismatch: {loaded_message_len} != {orig_message_len}"
        assert loaded_file_len == orig_file_len, f"File store length mismatch: {loaded_file_len} != {orig_file_len}"
        assert loaded_metadata_len == orig_metadata_len, f"Metadata store length mismatch: {loaded_metadata_len} != {orig_metadata_len}"
        
        # Clean up
        shutil.rmtree(test_path)
        
        return True

    # Run async test in sync context
    success = asyncio.run(_run_test())
    if success:
        print("Agent serialization test passed!")
    
    return success