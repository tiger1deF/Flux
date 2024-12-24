Vector Store
============

.. module:: agents.vectorstore.models
   :synopsis: Base vector store implementation for embedding storage and retrieval.

Overview
--------
This module provides an abstract base class for vector stores with thread-safe
operations and resource management. It includes support for both synchronous and
asynchronous operations, with proper cleanup of resources and thread pool management.

Base Store
---------

.. autoclass:: agents.vectorstore.models.BaseVectorStore
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __del__, acleanup_executor, cleanup_executor
   :exclude-members: _instance_lock, _executor, _run_in_executor
   :member-order: bysource

Implementations
-------------

HNSW Store
~~~~~~~~~
.. module:: agents.vectorstore.default.store
   :synopsis: HNSW vector store implementation for efficient local embeddings.

.. autoclass:: agents.vectorstore.default.store.HNSWStore
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __aenter__, __aexit__, __del__
   :exclude-members: _instance_lock, _executor, _run_in_executor, Config
   :member-order: bysource

Configuration
~~~~~~~~~~~
.. autoclass:: agents.vectorstore.default.store.HNSWStore.Config
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Thread Management
~~~~~~~~~~~~~~
.. autoclass:: agents.vectorstore.default.store.ContextAwareThreadPoolExecutor
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, submit
   :member-order: bysource