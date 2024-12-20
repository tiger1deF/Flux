Welcome to Flux Documentation
===========================

Flux is an advanced AI agent system that provides configurable agents with various capabilities, including LLM integration, vector storage, and monitoring.

.. toctree::
   :maxdepth: 2
   :caption: Core Components

   agents/index
   llm/index
   utils/index
   tests/index

Core Features
------------

Agents
^^^^^^
* Configurable AI agents with different logging modes
* Tool integration and React-style reasoning
* Message handling and state management
* Vector store integration for context management

LLM Integration
^^^^^^^^^^^^^
* Support for multiple LLM providers (Gemini, Pulse)
* Async and sync inference modes
* Embedding generation and management
* Token handling and context management

Utilities
^^^^^^^^
* Serialization and data handling
* Text summarization and truncation
* Shared tokenizer functionality
* File and metadata management

Testing
^^^^^^^
* Comprehensive test suites for all components
* Example implementations and usage patterns
* Integration tests for full agent workflows

Getting Started
-------------

Installation
^^^^^^^^^^^

.. code-block:: bash

   pip install -r requirements/requirements.txt

Basic Usage
^^^^^^^^^^

.. code-block:: python

   from agents import Agent, Message, AgentConfig, Logging
   
   # Create an agent with basic logging
   agent = Agent(
       name="Test Agent",
       config=AgentConfig(
           task_prompt="You must respond like a pirate",
           logging=Logging.ENABLED
       )
   )
   
   # Send a message
   response = await agent(Message(content="Hello, world!"))

Contributing
-----------
Contributions are welcome! Please feel free to submit a Pull Request.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`