"""
Test suite for embedding functionality and vector stores.

Tests embedding models in both sync and async contexts by calling them as functions.
"""
import asyncio
import time
import logging
from pathlib import Path
import shutil
import tempfile

from llm import (
    BaseEmbeddingFunction, 
    gemini_generate_embedding,
    pulse_embeddings,
    local_generate_embedding
)

from agents.vectorstore.default.store import HNSWStore
from agents.storage.filestore import FileStore

from agents.storage.message import Message, MessageType
from agents.storage.file import File, FileType
from agents.config.models import ContextConfig
from agents.state.models import AgentState
from agents.agent.models import Agent
from agents.storage.models import Chunk
from utils.shared.tokenizer import encode_async


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedder = BaseEmbeddingFunction(gemini_generate_embedding, dimension=768)


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
    print("\n=== Testing Vector Store Operations ===")
    
    # Initialize store
    print("\nInitializing store...")
    store = HNSWStore(embedding_function = embedder)
    
    # Test adding items
    print("\nTesting item addition:")
    id_1 = asyncio.run(store.add(
        "Hello, how are you?",
        metadata = Message(content="Hello, how are you?", type=MessageType.INPUT)
    ))
    print(f"✓ Added item 1 (ID: {id_1})")
    
    id_2 = asyncio.run(store.add(
        "What is up?",
        metadata = Message(content="What is up?", type=MessageType.OUTPUT)
    ))
    print(f"✓ Added item 2 (ID: {id_2})")
    
    # Test search
    print("\nTesting similarity search:")
    results = asyncio.run(store.search_relevant("How are you doing?", k=2))
    print(f"✓ Found {len(results)} similar items")
    for i, result in enumerate(results, 1):
        print(f"  Result {i}:")
        print(f"  - Content: {result.content}")
        print(f"  - Score: {result.score:.3f}")
    
    # Test deletion
    print("\nTesting deletion:")
    asyncio.run(store.delete(id_1))
    print(f"✓ Deleted item (ID: {id_1})")
    
    # Verify deletion
    remaining = len(store.data)
    print(f"✓ Store contains {remaining} items after deletion")
    
    print("\n=== Vector Store Tests Complete ===")


# Test data
sample_code = """
def fibonacci(n: int) -> int:
    \"\"\"Calculate the nth Fibonacci number recursively.\"\"\"
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n: int) -> int:
    \"\"\"Calculate the factorial of n recursively.\"\"\"
    if n <= 1:
        return 1
    return n * factorial(n-1)

def is_prime(n: int) -> bool:
    \"\"\"Check if a number is prime.\"\"\"
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
"""

sample_text = """
Mathematical Concepts Overview:

1. Fibonacci Sequence
   - Each number is the sum of the two preceding ones
   - Applications in nature and computer science
   
2. Factorial Calculation
   - Product of all positive integers up to n
   - Important in probability and combinatorics
   
3. Prime Numbers
   - Numbers with exactly two factors
   - Fundamental to cryptography and number theory
"""

additional_text = """
Advanced Mathematical Topics:

4. Derivatives
   - Rate of change of a function
   - Applications in optimization
   
5. Integrals
   - Area under a curve
   - Used in physics and engineering
"""

async def test_file_operations():
    """Test file store operations with both lazy and eager indexing"""
    print("\n=== Testing File Operations ===")
    
    # Test both lazy and eager indexing
    for lazy in [False, True]:
        print(f"\nTesting with {'lazy' if lazy else 'eager'} indexing")
        print("-" * 50)
        
        # Create agent with proper embedding function
        agent = Agent(
            embedding_function = embedder,
            state = AgentState(
                context_config = ContextConfig(
                    chunk_size = 50,
                    chunk_overlap = 20,
                    item_count = 3
                )
            )
        )
        
        # Get state from properly initialized agent
        state = agent.state
        
        # Use file store from agent state
        store = state.file_store
        store.lazy_indexing = lazy  # Set indexing mode
        
        try:
            # File creation and addition
            print("\nTesting file creation and addition...")
            start_time = time.time()
            
            python_file = await File.create(
                data = sample_code,
                path = "math_utils.py",  # Simplified path - FileStore handles directory
                type = FileType.PYTHON
            )
            text_file = await File.create(
                data = sample_text,
                path = "math_concepts.txt",  # Simplified path
                type = FileType.TEXT
            )
            
            await store.add_file(python_file)
            await store.add_file(text_file)   
            print(f"✓ File creation and addition took {time.time() - start_time:.3f}s")
            print(f"Store contains {await store.length} files")
            
            # Verify files are in store's temp directory
            assert python_file.path.startswith(str(store.temp_dir))
            assert text_file.path.startswith(str(store.temp_dir))
            
            # File merging - use just filename for output
            print("\nTesting file merging...")
            start_time = time.time()
        
            # First merge
            merged_file = await store.merge_files(
                [python_file.id, text_file.id],
                "merged.txt"
            )
            
            print(f"✓ Initial merge took {time.time() - start_time:.3f}s")
            first_merge_id = merged_file.id
            print(f"Store contains {await store.length} files after first merge")
            
            await store.add_file(python_file)
            await store.add_file(text_file)  
            
            assert merged_file.path == merged_file.path
            assert merged_file.id == first_merge_id  # Should be same ID since it's an update
            assert "source_files" in merged_file.annotations
            assert len(merged_file.annotations["source_files"]) == 2
            
            # Check raw data contains actual file contents in correct order
            assert merged_file.data == f"{sample_code}\n\n{sample_text}"  # First merge order
        
            # Test chunk retrieval with context config
            print("\nTesting chunk retrieval...")
            start_time = time.time()
                        
            chunks = await store.search_chunks(
                query = "fibbing",
                config = state.context_config,
                file_id = merged_file.id
            )
            
            print(f"✓ File-based chunk retrieval took {time.time() - start_time:.3f}s")
            print(f'Number of file ids represented: {len(set([chunk.parent_id for chunk in chunks]))}')
            print(f"Retrieved {len(chunks)} chunks\n")
            
            chunks = await store.search_chunks(
                query = "fibbing",
                config = state.context_config
            )
               
            print(f"✓ Chunk retrieval took {time.time() - start_time:.3f}s")
            print(f'Number of file ids represented: {len(set([chunk.parent_id for chunk in chunks]))}')
            print(f"Retrieved {len(chunks)} chunks")    
            
            # Verify chunks respect context config
            assert len(chunks) <= state.context_config.item_count
            for i, chunk in enumerate(chunks, 1):
                assert isinstance(chunk, Chunk)
                print(f'Chunk # {i}')
                print(chunk)
     
            # Verify metadata
            assert "source_files" in merged_file.annotations
            assert len(merged_file.annotations["source_files"]) == 2
            assert "source_summaries" in merged_file.annotations
            
            print(f"✓ File merging took {time.time() - start_time:.3f}s")
            
            # Serialization - create temporary directory just for store data
            print("\nTesting serialization...")
            start_time = time.time()
            with tempfile.TemporaryDirectory() as store_dir:
                store_path = Path(store_dir) / "store"
                
                # Get original files for comparison
                original_files = await store.list_files()
                original_file_ids = {f.id for f in original_files}
                
                # Serialize
                await store.serialize(store_path)
                print(f"✓ Serialization took {time.time() - start_time:.3f}s")
                
                # Deserialization
                print("\nTesting deserialization...")
                start_time = time.time()
                loaded_store = await FileStore.deserialize(store_path)
                loaded_files = await loaded_store.list_files()
            
                # Verify files were restored correctly
                loaded_file_ids = {f.id for f in loaded_files}
                assert loaded_file_ids == original_file_ids, \
                    f"Mismatched files. Original: {original_file_ids}, Loaded: {loaded_file_ids}"
                
                # Verify file contents
                for orig_file in original_files:
                    loaded_file = next(f for f in loaded_files if f.id == orig_file.id)
                    assert loaded_file.data == orig_file.data, \
                        f"File {orig_file.id} content mismatch"
                    assert loaded_file.type == orig_file.type, \
                        f"File {orig_file.id} type mismatch"
                
                print(f"✓ Deserialization took {time.time() - start_time:.3f}s")
            
            # Cleanup
            print("\nTesting file removal and cleanup...")
            start_time = time.time()
            await store.remove_file(text_file.id)
         
            remaining_files = await store.list_files()
            print(f'{len(remaining_files)} files remaining')
            print(f"✓ Cleanup took {time.time() - start_time:.3f}s")
            
            # Test serialization of files
            print("\nTesting file serialization...")
            start_time = time.time()
            
            # Serialize individual file
            serialized = await python_file.serialize()
            deserialized = await File.deserialize(serialized)
            
            # Verify content preserved
            assert deserialized.data == python_file.data
            assert deserialized.type == python_file.type
            assert deserialized.path == python_file.path
            
            # Verify cached summary preserved if present
            if python_file._content_summary:
                assert deserialized._content_summary == python_file._content_summary
                
            # Test with binary data
            binary_file = await File.create(
                data=b"Binary content",
                path="test.bin",
                type=FileType.BINARY
            )
            
            serialized = await binary_file.serialize()
            deserialized = await File.deserialize(serialized)
            assert deserialized.data == binary_file.data
            
            print(f"✓ File serialization took {time.time() - start_time:.3f}s")
            
            # Test chunk retrieval
            print("\nTesting chunk retrieval...")
            start_time = time.time()
            
            # Search for chunks containing specific terms
            chunks = await store.search_chunks(
                query = "fibonacci sequence",
                config = state.context_config  # Use config from state
            )
            
            # Verify chunk properties
            assert len(chunks) > 0, "No chunks found"
            for chunk in chunks:
                # Verify chunk size
                tokens = await encode_async(chunk.content)
                assert len(tokens) <= state.context_config.chunk_size, \
                    f"Chunk too large: {len(tokens)} > {state.context_config.chunk_size}"
                
            print(f"✓ Found {len(chunks)} relevant chunks")
            print(f"✓ Chunk retrieval took {time.time() - start_time:.3f}s")
            
        finally:
            await store.cleanup()

async def test_embedding_serialization():
    """Test embedding function serialization/deserialization"""
    
    # Create embedding function
    embedder = BaseEmbeddingFunction(
        embedding_fn=gemini_generate_embedding,
        dimension=768
    )
    
    # Test serialization
    serialized = await embedder.serialize()
    
    # Test deserialization
    restored = await BaseEmbeddingFunction.deserialize(serialized)
    
    # Verify properties preserved
    assert restored.dimension == embedder.dimension
    assert restored._fn_module == embedder._fn_module
    assert restored._fn_name == embedder._fn_name
    assert restored.accepts_list == embedder.accepts_list
    
    # Verify function works
    test_text = "Test embedding serialization"
    original_embedding = await embedder([test_text])
    restored_embedding = await restored([test_text])
    
    assert len(original_embedding) == len(restored_embedding)
