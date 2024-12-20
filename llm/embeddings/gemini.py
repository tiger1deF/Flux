from typing import List
import os
import threading

from llm.models import BaseEmbeddingFunction


_genai_lock = threading.Lock()
_genai = None


def _initialize_genai():
    """
    Initialize the Google Generative AI client.
    
    :return: Google Generative AI client
    :rtype: google.generativeai.GenerativeModel
    """
    global _genai
    if _genai is None:
        with _genai_lock:
            if _genai is None:
                import google.generativeai as genai
                genai.configure(api_key = os.getenv("GEMINI_API_KEY"))
                _genai = genai
    return _genai


async def gemini_generate_embedding(text: str) -> List[float]:
    """
    Generate embeddings for a given text using Gemini API.
    
    :param text: Text to embed
    :type text: str
    :return: Embedding vector
    :rtype: List[float]
    """
    genai = _initialize_genai()
    with _genai_lock:
        response = genai.embed_content(
            model='models/embedding-001',
            content=text,
            task_type='retrieval_document'
        )
    return response['embedding']


# Let it run the test during initialization
base_gemini_embedder = BaseEmbeddingFunction(gemini_generate_embedding)

