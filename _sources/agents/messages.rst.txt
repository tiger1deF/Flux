Message Components
================

Message Models
------------
.. automodule:: agents.messages.models
   :synopsis: Core message types and models for agent communication.

Base Message
~~~~~~~~~~
.. autoclass:: agents.messages.models.Message
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__, to_json, read_json
   :member-order: bysource

File Handling
-----------
.. autoclass:: agents.messages.models.File
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, to_dict, from_dict
   :member-order: bysource

Message Types
-----------
.. autoclass:: agents.messages.models.Sender
   :members:
   :undoc-members:
   :show-inheritance:

Examples
-------

Basic Usage
~~~~~~~~~

.. code-block:: python

   from agents.messages.models import Message, Sender
   
   # Create a user message
   message = Message(
       sender=Sender.USER,
       content="Hello agent!",
       metadata={"priority": "high"}
   )

   # Convert to JSON
   json_str = await message.to_json()

File Attachments
~~~~~~~~~~~~~

.. code-block:: python

   from agents.messages.models import File
   
   # Create a file attachment
   file = File(
       content="file contents",
       description="Example file",
       path="example.txt"
   )
   
   # Attach to message
   message.files.append(file) 