"""
Test suite for embedding functionality and vector stores.

This module provides tests for various embedding models and vector store operations,
including Gemini, Pulse, and local embeddings, as well as HNSW vector store functionality.
"""


def test_embeddings():
    """
    Test different embedding models.
    
    Tests three embedding implementations:
    1. Gemini embeddings
    2. Pulse embeddings
    3. Local embeddings
    
    For each model, tests dimension and embedding generation.
    """
    import asyncio

    from llm import (
        BaseEmbeddingFunction, 
        gemini_generate_embedding,
        pulse_embeddings,
        local_generate_embedding
    )

    print(f'Testing gemini embeddings...')
    embedder = BaseEmbeddingFunction(gemini_generate_embedding)
    print(embedder.dimension)
    embedding = asyncio.run(embedder("Hello, how are you?"))
    print(len(embedding))

    print(f'Testing pulse embeddings...')
    embedder = BaseEmbeddingFunction(pulse_embeddings)
    print(embedder.dimension)
    embedding = asyncio.run(embedder('What is up?'))
    print(len(embedding))

    print(f'Testing local embeddings...')
    embedder = BaseEmbeddingFunction(local_generate_embedding)
    print(embedder.dimension)
    embedding = asyncio.run(embedder('What is up?'))
    print(len(embedding))
    
    
def test_embeddings_store():
    """
    Test HNSW vector store operations.
    
    Tests the following vector store operations:
    1. Store initialization
    2. Adding items with and without metadata
    3. Similarity search
    4. Item deletion
    5. Search after deletion
    """
    import asyncio
    from agents import HNSWStore
    from llm import BaseEmbeddingFunction, gemini_generate_embedding
    embedder = BaseEmbeddingFunction(gemini_generate_embedding)
    
    print(f'Testing HNSWStore...')
    store = HNSWStore(embedding_function = embedder)
    
    print(f'Adding items...')
    id_1 = asyncio.run(store.add("Hello, how are you?"))
    id_2 = asyncio.run(store.add("What is up?"))
    id_3 = asyncio.run(store.add("Gosh darnit!"))
    id_4 = asyncio.run(store.add(
        text = "What is popping",
        embeddings = gemini_generate_embedding("What is popping"),
        metadata = {"name": "popping"}
    ))
    
    print(f'Searching for items...')
    data = asyncio.run(store.search("Hellooo"))
    print(data)
    
    print(f'Deleting items...')
    asyncio.run(store.delete([id_1, id_2]))
    
    print(f'Searching for items...')
    data = asyncio.run(store.search("Hellooo"))
    print(data)
    
