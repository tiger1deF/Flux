Tool Models
===========

.. module:: agents.tools.models
   :synopsis: Tool models for agent function wrapping and parameter handling.

Overview
--------
This module provides models for wrapping functions as tools that can be used by agents.
Tools can be called like regular functions while maintaining parameter validation,
static parameters, and metadata. The module supports both synchronous and asynchronous
execution with proper thread safety and event loop management.

Models
------

Tool
~~~~
.. autoclass:: agents.tools.models.Tool
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__, __sync_call__, __get__, __repr__
   :exclude-members: _is_async, Config
   :member-order: bysource

Tool Parameter
~~~~~~~~~~~~
.. autoclass:: agents.tools.models.ToolParameter
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __repr__
   :exclude-members: Config
   :member-order: bysource

Configuration
-----------

Tool Config
~~~~~~~~~~
.. autoclass:: agents.tools.models.Tool.Config
   :members:
   :undoc-members:
   :show-inheritance:

Parameter Config
~~~~~~~~~~~~~
.. autoclass:: agents.tools.models.ToolParameter.Config
   :members:
   :undoc-members:
   :show-inheritance: