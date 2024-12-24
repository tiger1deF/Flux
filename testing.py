# NOTE - Replace this with a file that initializes environmental variables
from deployment.config import initialize_environment
initialize_environment()

import asyncio
from pathlib import Path

from tests.embeddings import test_embeddings, test_embeddings_store
from tests.llm import test_llm
from tests.agents import test_base_agent, test_chat_agents, test_agent_serialization
from agents.agent.models import Agent

from agents.messages.message import Message, MessageType, Sender

from agents.messages.metadata import Metadata


# PASSING TESTS

# Embedding testing
test_embeddings_store()
test_embeddings()

test_llm()

# Agent testing
test_base_agent()
test_agent_serialization()


# WIP TESTS
#test_chat_agents()


'''
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
'''

# Run serialization test

if __name__ == '__main__':
    asyncio.run(test_agent_serialization())
