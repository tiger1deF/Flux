"""
Test suite for agent functionality.

This module provides tests for various agent configurations and logging modes,
including Langfuse integration, basic logging, and default operation without logging.
"""
import asyncio
import polars as pl

from agents import (
    Agent,
    AgentConfig,
    File,
    HNSWStore,
    Logging,
    Message,
    Metadata
)

from agents.agent.react.models import ReactAgent
from llm import (
    BaseEmbeddingFunction,
    gemini_generate_embedding,
    LLM,
    gemini_llm_async_inference
)
        

def test_base_agent():
    """
    Test basic agent functionality with different logging configurations.
    
    Tests three configurations:
    1. Agent with Langfuse logging
    2. Agent with basic logging enabled
    3. Agent with logging disabled
    
    Each configuration tests message sending and response handling.
    """
    
    print(f'Testing langfuse langchain agent')
    config = AgentConfig(
        task_prompt = "You must respond like a pirate",
        logging = Logging.LANGFUSE
    )
    
    from llm.embeddings.gemini import _initialize_genai
    _initialize_genai()
    
    llm = LLM(
        gemini_llm_async_inference, 
        input_tokens = 2000, 
        max_tokens = 2000,
        system_prompt = "You must respond like a pirate"
    )
    
    # Create embedding function
    embedding_function = BaseEmbeddingFunction(
        gemini_generate_embedding,
        dimension = 768
    )
    
    agent = Agent(
        name = "Test Agent",
        llm = llm,
        config = config,
        embedding_function = embedding_function
    )
    
    # Verify embedding function propagation
    assert agent.state.embedding_function == embedding_function
    assert agent.state.file_store.embedding_function == embedding_function
    assert agent.state.metadata_store.embedding_function == embedding_function
    
    async def test_agent():
        # Create test data
        test_df = pl.DataFrame({
            "id": range(5),
            "value": ["a", "b", "c", "d", "e"],
            "number": [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        # Create metadata object
        metadata = Metadata(
            name="test_data",
            description="Sample test data",
            data=test_df
        )
        
        # Create a test file
        test_file = await File.create(
            data="Test file content",
            path="test.txt"
        )
        
        # Create message with direct objects
        message = Message(
            content="Hello, world!",
            metadata=[metadata],  # Direct metadata object
            files=[test_file]    # Direct file object
        )
        
        # Send to agent
        response = await agent(message)
        print(f'Response: {response}')
        
        # Verify ingestion
        stored_metadata = await agent.state.metadata_store.get(metadata.id)
        assert stored_metadata is not None
        assert stored_metadata.name == metadata.name
        
        stored_file = await agent.state.file_store.get_file(test_file.id)
        assert stored_file is not None
        assert stored_file.path == test_file.path
        
        print(f"Response: {response.content}")
        if agent.config.logging != Logging.DISABLED:
            log_text = await agent.text()
            print(log_text)

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
        output = await agent.text()
        print(f'AGENT OUTPUT: {output}')
     
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
    
    
    print("\n=== Testing Base Agent ===")
    
    # Initialize agent
    print("\nInitializing agent...")
    agent = Agent(name="test_agent")
    print(f"✓ Created agent: {agent.name}")
    
    async def test_agent():
        print("\nTesting message processing:")
        message = Message(content="Test message")
        
        try:
            response = await agent(message)
            print(f"✓ Processed message")
            print(f"  Response: {response.content}")
            
            # Add debug logging
            print("\nAgent Log State:")
            print(f"- Input logged: {bool(agent.agent_log.agent_input)}")
            print(f"- Output logged: {bool(agent.agent_log.agent_output)}")
            if agent.agent_log.agent_output:
                print(f"- Output message: {agent.agent_log.agent_output.output_message}")
            
            # Only try to print text if output exists
            if agent.agent_log.agent_output:
                print("\nFull Agent Log:")
                print(await agent.text())
            else:
                print("\nWarning: No output logged yet")
                
        except Exception as e:
            print(f"\nError in agent processing: {str(e)}")
            print(f"Agent state: {agent.agent_status}")
            raise
    
    asyncio.run(test_agent())
    print("\n=== Base Agent Tests Complete ===")



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
    from agents.storage.message import Message, Sender
    from agents.storage.metadata import Metadata

    async def _run_test():
        # Create test agent
        agent = Agent(name="TestAgent", type="test")
        
        # Add some test data
        test_message = Message(
            content="Hello, this is a test message",
            sender=Sender.USER
        )
        await agent.state.ingest_message_data(test_message)
        
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
        
        orig_metadata_len = len(agent.state.metadata_store.data)
        
        # Serialize agent
        test_path = Path("test_serialization")
        await agent.serialize(test_path)
    
        # Deserialize to new agent
        loaded_agent = await Agent.deserialize(test_path)
        
        # Verify data
        assert loaded_agent.name == agent.name
        assert loaded_agent.type == agent.type
        
        loaded_metadata_len = len(loaded_agent.state.metadata_store.data)
        
        assert loaded_metadata_len == orig_metadata_len, f"Metadata store length mismatch: {loaded_metadata_len} != {orig_metadata_len}"
        
        # Clean up
        shutil.rmtree(test_path)

    # Run async test in sync context
    asyncio.run(_run_test())


def test_react_agent():
    
    def add_values(
        a: int, 
        b: int, 
        c: float
    ) -> int:
        """
        Add two values a and b and raise to the power of c
        """
        intermediate = a + b
        return intermediate ** c
    
  
    llm = LLM(
        pulse_llm_async_inference, 
        input_tokens = 2000, 
        max_tokens = 2000,
    )
    config = AgentConfig()
    
    agent = ReactAgent(
        name = "Test Agent",
        llm = llm,
        config = config
    )
