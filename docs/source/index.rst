################################
Welcome to Flux Documentation
################################

.. image:: _static/flux_logo.png
   :alt: Flux Logo
   :align: center

|

**Flux** is an advanced AI agent system that provides configurable agents with various capabilities, including LLM integration, vector storage, and monitoring.

.. note::
   This documentation covers the latest version of Flux. For older versions, please refer to the version selector.

.. toctree::
   :maxdepth: 2
   :caption: Core Components
   :name: mastertoc

   agents/index
   llm/index
   utils/index
   tests/index

*************
Core Features
*************

Agents
======
* **Configurable AI Agents**: Flexible agents with different logging modes
* **Tool Integration**: React-style reasoning and execution
* **Message Handling**: Robust state management system
* **Vector Store**: Advanced context management with vector databases

LLM Integration
==============
* **Multiple Providers**: Support for Gemini, Pulse, and more
* **Inference Modes**: Both async and sync operations
* **Embedding Management**: Efficient vector operations
* **Context Control**: Smart token and context handling

Utilities
========
* **Data Processing**: Advanced serialization and handling
* **Text Operations**: Summarization and truncation utilities
* **Shared Resources**: Common tokenizer functionality
* **File Management**: Robust file and metadata handling

Testing
=======
* **Test Coverage**: Comprehensive test suites
* **Example Code**: Clear implementation patterns
* **Integration**: End-to-end workflow testing

***************
Getting Started
***************

Installation
===========

.. code-block:: bash
   :caption: Install via pip
   :emphasize-lines: 1

   pip install -r requirements/requirements.txt

Quick Start
==========

.. code-block:: python
   :caption: Basic agent usage
   :emphasize-lines: 11,19

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

************
Contributing
************

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

For more details, see our contribution guidelines.

********************
Indices and Tables
********************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`