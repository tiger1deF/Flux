Language Models
=============

.. toctree::
   :maxdepth: 2
   :caption: Components

   models
   embeddings
   inference

Overview
--------

.. module:: llm
   :synopsis: Language model implementations and interfaces.

The LLM module provides a comprehensive framework for working with language models,
including inference, embeddings, and model management. The module is organized into
three main components:

Base Models
~~~~~~~~~~
* Base LLM interface for model inference
* Parameter and configuration management
* Thread-safe execution handling

See :doc:`models` for implementation details.

Embeddings
~~~~~~~~~
* Multiple embedding providers (Gemini, Pulse)
* Async and sync embedding generation
* Vector operations and caching

See :doc:`embeddings` for implementation details.

Inference
~~~~~~~~
* Multiple model providers (Gemini, Pulse, Azure)
* Async and sync inference
* Configurable generation parameters
* Thread-safe model management

See :doc:`inference` for implementation details.