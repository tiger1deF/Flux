Chat Agents
===========

This section covers the chat agent implementations available in the library.

.. note::
   Chat agents are responsible for managing conversations, context, and message history.

Base Chat Agent
--------------

The base chat agent provides core functionality for message handling and agent state management.

.. module:: agents.agent.chat.base
   :synopsis: Base chat agent implementation

.. autoclass:: BaseChatAgent
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
~~~~~~~~~~~~

Chat agents can be configured using the following settings:

- ``logging``: Enable/disable logging (default: True)
- ``state``: Agent state configuration
- ``config``: Agent configuration parameters

Example Usage
~~~~~~~~~~~~

.. code-block:: python

   from agents.agent.chat.base import BaseChatAgent
   
   agent = BaseChatAgent()
   response = await agent.send_message("Hello!")
   print(response.content)

See Also
--------

- :doc:`/agents/models` - Core agent models and types
- :doc:`/agents/messages` - Message handling and types
- :doc:`/agents/monitor` - Agent monitoring and logging 