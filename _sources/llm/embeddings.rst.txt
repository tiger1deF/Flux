Embeddings
=========

.. module:: llm.models
   :synopsis: Base embedding model implementations.

Overview
--------
The embeddings module provides implementations for various embedding models,
with support for both synchronous and asynchronous operations. Each implementation
is thread-safe and handles proper resource management.

Base Embedding Function
---------------------
.. autoclass:: llm.models.BaseEmbeddingFunction
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__

Provider Implementations
----------------------

Pulse Embeddings
---------------
.. automodule:: llm.embeddings.pulse
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: pulse_fetch_embedding

Gemini Embeddings
---------------
.. automodule:: llm.embeddings.gemini
   :members:
   :undoc-members:
   :show-inheritance:

Local Embeddings
--------------
.. automodule:: llm.embeddings.local
   :members:
   :undoc-members:
   :show-inheritance: 