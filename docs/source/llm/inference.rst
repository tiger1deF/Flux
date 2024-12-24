Inference
=========

Overview
--------
The inference module provides implementations for various LLM providers,
with support for both synchronous and asynchronous operations. Each implementation
is thread-safe and handles proper resource management.

Pulse Inference
-------------

Synchronous
~~~~~~~~~~
.. autofunction:: llm.pulse_llm_sync_inference

Asynchronous
~~~~~~~~~~~
.. autofunction:: llm.pulse_llm_async_inference

Gemini Inference
--------------

Synchronous
~~~~~~~~~~
.. autofunction:: llm.gemini_llm_sync_inference

Asynchronous
~~~~~~~~~~~
.. autofunction:: llm.gemini_llm_async_inference

Implementation Details
-------------------

Pulse Implementation
~~~~~~~~~~~~~~~~~~

.. module:: llm.inference.pulse.sync_inference
   :synopsis: Pulse model synchronous inference implementation.

.. automodule:: llm.inference.pulse.sync_inference
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _get_internal_model_list

.. module:: llm.inference.pulse.async_inference
   :synopsis: Pulse model asynchronous inference implementation.

.. automodule:: llm.inference.pulse.async_inference
   :members:
   :undoc-members:
   :show-inheritance:

Gemini Implementation
~~~~~~~~~~~~~~~~~~~

.. module:: llm.inference.gemini.sync_inference
   :synopsis: Gemini model synchronous inference implementation.

.. automodule:: llm.inference.gemini.sync_inference
   :members:
   :undoc-members:
   :show-inheritance:

.. module:: llm.inference.gemini.async_inference
   :synopsis: Gemini model asynchronous inference implementation.

.. automodule:: llm.inference.gemini.async_inference
   :members:
   :undoc-members:
   :show-inheritance: 