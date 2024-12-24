import os

# Force PyTorch/disable TensorFlow before any imports
os.environ['USE_TORCH'] = '1'
os.environ['NO_TENSORFLOW'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from typing import List
import threading
from functools import lru_cache
import numpy as np
from llm.models import BaseEmbeddingFunction
import asyncio


_model_lock = threading.Lock()


@lru_cache(maxsize = 1)
def _get_model():
    """
    Get the SentenceTransformer model for generating embeddings.
    
    :return: SentenceTransformer model
    :rtype: SentenceTransformer
    """
    try:
        import torch
        from sentence_transformers import SentenceTransformer
     
        model = SentenceTransformer(
            "BAAI/bge-large-en-v1.5",  # 1024d model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    
        model.to(device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return model
        
    except Exception as e:
        raise ValueError(f"Error loading model: {e}")


def local_generate_embedding(text: str) -> List[float]:
    """
    Generate embeddings for a given text using the local model.
    
    :param text: Text to embed
    :type text: str
    :return: Embedding vector
    :rtype: List[float]
    """
    with _model_lock:
        model = _get_model()
        embedding = model.encode(
            f"passage: {text}",  # BGE works better with this prefix
            normalize_embeddings = True,
            convert_to_numpy = True
        )
        return embedding.tolist()  # Convert numpy array to list


base_local_embedder = BaseEmbeddingFunction(
    embedding_fn = local_generate_embedding,
    dimension = 1024,
)
