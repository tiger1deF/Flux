Core Agent
==========

.. module:: agents.models
   :synopsis: Core agent models and state management.

Overview
--------
This module provides base models for agent configuration, state management,
and file handling with support for both synchronous and asynchronous operations.

Base Models
----------

Agent State
~~~~~~~~~~
.. autoclass:: agents.models.AgentState
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __adel__, cleanup, create_file, read_file, obtain_context
   :exclude-members: _metadata_lock, _executor, _run_in_executor, _cleanup_sync
   :member-order: bysource

Agent Config
~~~~~~~~~~
.. autoclass:: agents.models.AgentConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __repr__
   :exclude-members: _metadata_lock
   :member-order: bysource

Status
~~~~~
.. autoclass:: agents.models.AgentStatus
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Logging
~~~~~~
.. autoclass:: agents.models.Logging
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Agent Implementation
------------------

.. module:: agents.agent.models
   :synopsis: Core agent implementation and functionality.

Overview
~~~~~~~~
The agent module provides the base Agent class and related utilities for building
and managing agents in the system. It includes functionality for:

- Agent state and configuration management
- Message handling and communication
- LLM integration
- Logging and monitoring

Base Agent
~~~~~~~~~
.. autoclass:: agents.agent.models.Agent
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__, __repr__, __str__
   :exclude-members: _thread_local, _lock, Config
   :member-order: bysource

Configuration
~~~~~~~~~~~
.. autoclass:: agents.agent.models.Agent.Config
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Message Handling
-------------

.. automethod:: agents.agent.models.Agent.add_message

.. automethod:: agents.agent.models.Agent.get_messages

.. automethod:: agents.agent.models.Agent.send_message

.. automethod:: agents.agent.models.Agent.sync_send_message

.. automethod:: agents.agent.models.Agent.async_call

.. automethod:: agents.agent.models.Agent.sync_call

State Management
-------------

.. automethod:: agents.agent.models.Agent.clear_history

.. automethod:: agents.agent.models.Agent.update_context

.. automethod:: agents.agent.models.Agent.text

Utilities
--------

.. autofunction:: agents.agent.models.conditional_logging