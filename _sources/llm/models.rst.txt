Language Models
===============

.. module:: llm
   :synopsis: Language model implementations and interfaces.

Overview
--------
The language models module provides base classes and utilities for working with
LLMs, including parameter management, type checking, and thread-safe execution.

Base LLM
--------
.. autoclass:: llm.LLM
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__
   :exclude-members: _build_call_implementation, _get_type_hints, _inspect_function, _async_lock

Parameter Models
--------------
.. automodule:: llm.models
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

See :doc:`embeddings` for embedding model implementations and :doc:`inference` for
inference implementations. 