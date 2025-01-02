import numpy as np
from typing import Union, List


def generate_static_embeddings(
    text: Union[str, List[str]], 
    dimension: int = 768
) -> np.ndarray:
    """
    Generate a static embedding array of ones for testing/default purposes.
    
    :param text: Input text (unused, kept for API compatibility)
    :type text: Union[str, List[str]]
    :param dimension: Dimension of the embedding vector
    :type dimension: int
    :return: Array of ones with specified dimension
    :rtype: np.ndarray
    """
    if isinstance(text, list):
        return np.ones((len(text), dimension))
    return np.ones(dimension) 