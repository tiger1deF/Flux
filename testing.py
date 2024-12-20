"""
Test runner for agent system components.

This module provides a test runner for various components of the agent system,
including embeddings, LLM functionality, vector stores, and agent operations.
"""


# NOTE - Replace this with a file that initializes environmental variables
from deployment.config import initialize_environment
initialize_environment()

import asyncio

from agents import Agent

from tests.embeddings import test_embeddings
from tests.llm import test_llm
from tests.embeddings import test_embeddings_store
from tests.agents import test_base_agent


# PASSING TESTS

#test_embeddings()
#test_llm()
#test_embeddings_store()

# WIP TESTS
test_base_agent()

import sys
sys.exit()
###################
# AGENT TESTS #
###################

def test_tool(a: int, b: int) -> int:
    """
    Test tool function that adds two numbers.
    
    :param a: First number
    :type a: int
    :param b: Second number
    :type b: int
    :return: Sum of the numbers
    :rtype: int
    """
    return a + b

agent = Agent()

async def main():
    """
    Main test function for agent interaction.
    
    Tests basic agent functionality by sending a message and
    checking the response and log output.
    """
    pulse_response = await agent('Hello, how are you?')
    print(pulse_response)   
    print(agent.text())
   
   
if __name__ == '__main__':
    asyncio.run(main())
