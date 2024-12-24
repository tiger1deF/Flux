"""
Test suite for embedding functionality and vector stores.

Tests embedding models in both sync and async contexts by calling them as functions.
"""
import asyncio
from typing import List
import time
import logging

from llm import (
    BaseEmbeddingFunction, 
    gemini_generate_embedding,
    pulse_embeddings,
    local_generate_embedding
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_embeddings():
    """Test embeddings in both sync and async contexts"""
    
    # Test Gemini embeddings
    logger.info("\n=== Testing Gemini embeddings ===")
    start_time = time.time()
    embedder = BaseEmbeddingFunction(
        gemini_generate_embedding,
        dimension = 768
    )
    logger.info(f"Embedder initialization took {time.time() - start_time:.2f}s")
    logger.info(f"Dimension: {embedder.dimension}\n")
    
    # Test in sync context
    start_time = time.time()
    embedding = embedder("Hello, how are you?")  # Direct sync call
    logger.info(f"Sync call took {time.time() - start_time:.2f}s (length: {len(embedding)})\n")
    
    # Test in async context
    start_time = time.time()
    async def async_test():
        return await embedder("Hello, how are you?")
    embedding = asyncio.run(async_test())  # Async call
    logger.info(f"Async call took {time.time() - start_time:.2f}s (length: {len(embedding)})\n")
    
    # Test batch operations
    texts = ["Hello", "World", "Test"]
    
    start_time = time.time()
    embeddings = [embedder(text) for text in texts]  # Sync batch
    logger.info(f"Sync batch took {time.time() - start_time:.2f}s (size: {len(embeddings)})\n")
    
    start_time = time.time()
    async def async_batch():
        return await asyncio.gather(*[embedder(text) for text in texts])
    embeddings = asyncio.run(async_batch())  # Async batch
    logger.info(f"Async batch took {time.time() - start_time:.2f}s (size: {len(embeddings)})\n")

    # Test Pulse embeddings
    logger.info("\n=== Testing Pulse embeddings ===")
    start_time = time.time()
    embedder = BaseEmbeddingFunction(pulse_embeddings)
    logger.info(f"Embedder initialization took {time.time() - start_time:.2f}s")
    logger.info(f"Dimension: {embedder.dimension}\n")
    
    start_time = time.time()
    embedding = embedder("What is up?")  # Direct sync call
    logger.info(f"Sync call took {time.time() - start_time:.2f}s (length: {len(embedding)})\n")
    
    start_time = time.time()
    async def async_test():
        return await embedder("What is up?")
    embedding = asyncio.run(async_test())  # Async call
    logger.info(f"Async call took {time.time() - start_time:.2f}s (length: {len(embedding)})\n")
    
    start_time = time.time()
    embeddings = [embedder(text) for text in texts]  # Sync batch
    logger.info(f"Sync batch took {time.time() - start_time:.2f}s (size: {len(embeddings)})\n")
    
    start_time = time.time()
    async def async_batch():
        return await asyncio.gather(*[embedder(text) for text in texts])
    embeddings = asyncio.run(async_batch())  # Async batch
    logger.info(f"Async batch took {time.time() - start_time:.2f}s (size: {len(embeddings)})\n")

    # Test Local embeddings
    logger.info("\n=== Testing Local embeddings ===")
    start_time = time.time()
    embedder = BaseEmbeddingFunction(local_generate_embedding)
    logger.info(f"Embedder initialization took {time.time() - start_time:.2f}s")
    logger.info(f"Dimension: {embedder.dimension}\n")
    
    start_time = time.time()
    embedding = embedder("What is up?")  # Direct sync call
    logger.info(f"Sync call took {time.time() - start_time:.2f}s (length: {len(embedding)})\n")
    
    start_time = time.time()
    async def async_test():
        return await embedder("What is up?")
    embedding = asyncio.run(async_test())  # Async call
    logger.info(f"Async call took {time.time() - start_time:.2f}s (length: {len(embedding)})\n")
    
    start_time = time.time()
    embeddings = [embedder(text) for text in texts]  # Sync batch
    logger.info(f"Sync batch took {time.time() - start_time:.2f}s (size: {len(embeddings)})\n")
    
    start_time = time.time()
    async def async_batch():
        return await asyncio.gather(*[embedder(text) for text in texts])
    embeddings = asyncio.run(async_batch())  # Async batch
    logger.info(f"Async batch took {time.time() - start_time:.2f}s (size: {len(embeddings)})\n")


def test_embeddings_store():
    """Test HNSW vector store operations"""
    import asyncio
    from agents import HNSWStore
    from llm import BaseEmbeddingFunction, gemini_generate_embedding
    embedder = BaseEmbeddingFunction(gemini_generate_embedding)
    
    from agents import Message
    from agents.messages.message import MessageType
    
    print(f'Testing HNSWStore...')
    store = HNSWStore(embedding_function = embedder)
    
    print(f'Adding items...')
    id_1 = asyncio.run(
        store.add(
            "Hello, how are you?",
            metadata = Message(
                content = "Hello, how are you?",
                type = MessageType.INPUT
            )
        )
    )
    id_2 = asyncio.run(
        store.add(
            "What is up?",
            metadata = Message(
                content = "What is up?",
                type = MessageType.OUTPUT
            )
        )
    )
    id_3 = asyncio.run(
        store.add(
            "Gosh darnit!",
            metadata = Message(
                content = "Gosh darnit!",
                type = MessageType.INPUT
            )
        )
    )
    
    # Get embeddings synchronously since we're in a sync context
    embeddings = embedder("What is popping")  # This will now return actual embeddings
   
    id_4 = asyncio.run(store.add(
        text = "What is popping",
        embeddings = embeddings,  # Note: store expects a single vector, not List[List[float]]
        metadata = Message(
            content = "What is popping",
            type = MessageType.INTERMEDIATE
        )
    ))
    
    print(f'Searching for items...')
    data = asyncio.run(store.search_relevant("Hellooo"))
    print(data)
    
    print(f'Deleting items...')
    asyncio.run(store.delete([id_1, id_2]))
    
    print(f'Searching for items...')
    data = asyncio.run(store.search_relevant("Hellooo"))
    print(data)
    
    print(f'Filtering items...')
    data = asyncio.run(store.search_attribute(
        "type", 
        [MessageType.INPUT]
    ))
    print(data)
