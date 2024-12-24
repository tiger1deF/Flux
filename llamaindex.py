import logging
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.llms import CustomLLM, LLMMetadata, CompletionResponse
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.graph_stores.simple import SimpleGraphStore
from typing import Optional, Generator, Any, List
from pydantic import BaseModel, PrivateAttr
import networkx as nx
from datetime import datetime
import json
import asyncio
import time
import threading
from llama_index.core.storage import StorageContext


from deployment.config import initialize_environment
# Initialize environment
initialize_environment()



# Import your existing LLM and embedding implementations
from llm import (
    LLM, 
    pulse_llm_async_inference,
    BaseEmbeddingFunction,
    pulse_embeddings  # Your custom embeddings
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('graph_rag_tests.log')
logger.addHandler(file_handler)

class PulseEmbedding(BaseEmbedding):
    """Adapter for using Pulse embeddings with LlamaIndex"""
    
    def __init__(self):
        super().__init__()
        logger.info("Initializing PulseEmbedding adapter")
        self._embedder = BaseEmbeddingFunction(
            pulse_embeddings,
            batch_size=32
        )

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get multiple text embeddings synchronously"""
        start = time.time()
        logger.debug(f"Getting embeddings for {len(texts)} texts synchronously")
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logger.debug("Using existing event loop")
                # Create a new event loop in a separate thread for sync operations
                result_container = []
                
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(self._aget_text_embeddings(texts))
                        result_container.append(result)
                    finally:
                        new_loop.close()
                
                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join()
                return result_container[0]
            else:
                logger.debug("No running loop - using run_until_complete")
                return loop.run_until_complete(self._aget_text_embeddings(texts))
        except RuntimeError:
            logger.debug("No event loop - creating new one")
            return asyncio.run(self._aget_text_embeddings(texts))
        finally:
            logger.debug(f"Embedding completed in {time.time() - start:.2f}s")

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get single text embedding synchronously"""
        logger.debug("Getting single text embedding synchronously")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Use same thread-based approach for single embeddings
                result_container = []
                
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(self._aget_text_embedding(text))
                        result_container.append(result)
                    finally:
                        new_loop.close()
                
                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join()
                return result_container[0]
            else:
                return loop.run_until_complete(self._aget_text_embedding(text))
        except RuntimeError:
            return asyncio.run(self._aget_text_embedding(text))

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding synchronously"""
        logger.debug("Getting query embedding synchronously")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Use same thread-based approach for queries
                result_container = []
                
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(self._aget_query_embedding(query))
                        result_container.append(result)
                    finally:
                        new_loop.close()
                
                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join()
                return result_container[0]
            else:
                return loop.run_until_complete(self._aget_query_embedding(query))
        except RuntimeError:
            return asyncio.run(self._aget_query_embedding(query))

    def _sync_batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Synchronous batch embedding implementation"""
        logger.debug(f"Processing batch of {len(texts)} texts")
        results = []
        for text in texts:
            # Use the underlying sync function directly
            result = self._embedder._sync_impl(text)
            results.append(result[0])  # Get first vector for single texts
        return results

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get multiple text embeddings asynchronously"""
        logger.debug(f"Getting embeddings for {len(texts)} texts asynchronously")
        try:
            # Process in batches
            batch_size = self._embedder.batch_size or len(texts)
            results = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1} of {len(texts)//batch_size + 1}")
                
                batch_results = await asyncio.gather(*[
                    self._aget_text_embedding(text) for text in batch
                ])
                results.extend(batch_results)
            
            return results
        except Exception as e:
            logger.error(f"Error in async embedding: {str(e)}")
            raise

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get single text embedding asynchronously"""
        try:
            result = await self._embedder(text)
            return result[0]  # Return single vector
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            raise

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding asynchronously"""
        logger.debug("Getting query embedding asynchronously")
        result = await self._embedder(query)
        return result[0]

class PulseLlamaLLM(CustomLLM, BaseModel):
    """Adapter for using Pulse LLM with LlamaIndex"""
    
    _pulse_llm: Any = PrivateAttr()
    context_window: int = 4096
    max_new_tokens: int = 2000
    model_name: str = "Mixtral-8x22B-Instruct-v0.1"
    
    def __init__(self, **data):
        super().__init__(**data)
        self._pulse_llm = LLM(
            pulse_llm_async_inference,
            model_name=self.model_name,
            input_tokens=self.context_window,
            max_tokens=self.max_new_tokens
        )

    @property
    def metadata(self) -> LLMMetadata:
        """Return LLM metadata in the format LlamaIndex expects"""
        return LLMMetadata(
            context_window=self.context_window,
            max_new_tokens=self.max_new_tokens,
            model_name=self.model_name,
            is_chat_model=False,
            is_function_calling_model=False,
            model_backend="custom"
        )

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Required by LlamaIndex - sync completion"""
        from llama_index.core.llms import CompletionResponse
        raw_response = asyncio.run(self._pulse_llm(prompt))
        return CompletionResponse(text=raw_response)

    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Required by LlamaIndex - async completion"""
        from llama_index.core.llms import CompletionResponse
        raw_response = await self._pulse_llm(prompt)
        return CompletionResponse(text=raw_response)
        
    def stream_complete(self, prompt: str, **kwargs) -> Generator[CompletionResponse, None, None]:
        """Required by LlamaIndex - sync streaming"""
        from llama_index.core.llms import CompletionResponse
        response = self.complete(prompt, **kwargs)
        yield response

    async def astream_complete(self, prompt: str, **kwargs) -> Generator[CompletionResponse, None, None]:
        """Required by LlamaIndex - async streaming"""
        from llama_index.core.llms import CompletionResponse
        response = await self.acomplete(prompt, **kwargs)
        yield response

async def test_llama_index_graph():
    """Test ontology-based graph creation with LlamaIndex"""
    logger.info("Starting ontology-based graph creation...")
    
    # Step 1: Define ontology/schema
    event_relationships = {
        "influenced": {"source_type": "event", "target_type": "event"},
        "participated_in": {"source_type": "country", "target_type": "event"},
        "preceded": {"source_type": "event", "target_type": "event"}
    }
    
    logger.info("Defined ontology with relationships:")
    for rel, constraints in event_relationships.items():
        logger.info(f"  {rel}: {constraints['source_type']} -> {constraints['target_type']}")
    
    # Step 2: Initialize components
    logger.info("\nInitializing graph components...")
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    
    Settings.llm = PulseLlamaLLM()
    Settings.embed_model = PulseEmbedding()
    Settings.chunk_size = 1024
    
    # Step 3: Prepare documents with typed entities
    logger.info("\nPreparing typed documents...")
    test_docs = [
        Document(
            text="World War II was a global conflict from 1939-1945",
            id_="ww2",
            metadata={"type": "event", "start_year": 1939, "end_year": 1945}
        ),
        Document(
            text="The Cold War was a period of geopolitical tension",
            id_="cold_war",
            metadata={"type": "event", "start_year": 1947, "end_year": 1991}
        ),
        Document(
            text="The Soviet Union was a major participant in both wars",
            id_="soviet",
            metadata={"type": "country", "exists": True}
        )
    ]
    
    for doc in test_docs:
        logger.info(f"  Added document: {doc.id_} (type: {doc.metadata['type']})")
    
    # Step 4: Create vector index for semantic search
    logger.info("\nCreating vector index...")
    vector_index = VectorStoreIndex.from_documents(
        documents=test_docs,
        storage_context=storage_context,
        show_progress=True
    )
    
    try:
        # Step 5: Extract and validate relationships
        logger.info("\nExtracting relationships based on ontology...")
        relationships_to_add = [
            {
                "source": "ww2",
                "relation": "influenced",
                "target": "cold_war",
                "metadata": {"confidence": 0.9}
            },
            {
                "source": "soviet",
                "relation": "participated_in",
                "target": "ww2",
                "metadata": {"confidence": 0.95}
            },
            {
                "source": "soviet",
                "relation": "participated_in",
                "target": "cold_war",
                "metadata": {"confidence": 0.95}
            }
        ]
        
        # Step 6: Add validated relationships to graph
        logger.info("Adding validated relationships to graph...")
        for rel in relationships_to_add:
            # Validate against ontology
            rel_constraints = event_relationships.get(rel["relation"])
            if rel_constraints:
                source_doc = next(d for d in test_docs if d.id_ == rel["source"])
                target_doc = next(d for d in test_docs if d.id_ == rel["target"])
                
                if (source_doc.metadata["type"] == rel_constraints["source_type"] and 
                    target_doc.metadata["type"] == rel_constraints["target_type"]):
                    logger.info(f"  Adding valid relationship: {rel['source']} --[{rel['relation']}]--> {rel['target']}")
                    graph_store.add_triple(
                        subject=rel["source"],
                        predicate=rel["relation"],
                        object_=rel["target"],
                        metadata=rel["metadata"]
                    )
                else:
                    logger.warning(f"  Skipping invalid relationship: {rel['source']} --[{rel['relation']}]--> {rel['target']}")
        
        # Step 7: Query the graph
        logger.info("\nTesting graph query capabilities...")
        query = "Impact of World War II on Cold War"
        query_engine = vector_index.as_query_engine(
            include_metadata=True,
            response_mode="tree_summarize"
        )
        response = await query_engine.aquery(query)
        logger.info(f"Query result: {response}")
        
        # Step 8: Analyze graph structure
        logger.info("\nAnalyzing final graph structure...")
        try:
            # Get all relationships
            triples = graph_store.get_triples()
            logger.info("Graph relationships:")
            for triple in triples:
                logger.info(f"  {triple.subject} --[{triple.predicate}]--> {triple.object_}")
                if hasattr(triple, 'metadata') and triple.metadata:
                    logger.info(f"    Metadata: {triple.metadata}")
        except Exception as e:
            logger.error(f"Error getting triples: {str(e)}")
            # Fallback to relationship iteration if needed
            logger.info("Attempting to get relationships directly...")
            for rel in relationships_to_add:
                if graph_store.has_triple(rel["source"], rel["relation"], rel["target"]):
                    logger.info(f"  Verified: {rel['source']} --[{rel['relation']}]--> {rel['target']}")
        
    finally:
        if hasattr(vector_index, 'close'):
            vector_index.close()

if __name__ == "__main__":
    asyncio.run(test_llama_index_graph())