from deployment.config import initialize_environment
initialize_environment()

import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import test functions
from tests.embeddings import (
    test_embeddings,
    test_embeddings_store,
    test_file_operations
)
from tests.llm import test_llm
from tests.agents import (
    test_base_agent,
    test_agent_serialization,
    test_react_agent
)


def run_tests():
    print("\n=== Running All Tests ===")
    print("=" * 50)
    
    # Embedding tests
    print("\n1. Testing Embeddings")
    print("-" * 50)
    test_embeddings()
    test_embeddings_store()
    asyncio.run(test_file_operations())

    # LLM tests
    print("\n2. Testing LLM")
    print("-" * 50)
    test_llm()

    
    # Agent tests
    print("\n3. Testing Agents")
    print("-" * 50)
    test_base_agent()
    test_agent_serialization()
    #test_react_agent()
    print("\n=== All Tests Complete ===")


if __name__ == "__main__":
    run_tests()
