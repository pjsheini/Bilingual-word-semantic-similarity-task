"""Bilingual word semantic similarity: evaluate cross-lingual word embeddings."""

from .embeddings import Embeddings
from .evaluate import EvalResult, evaluate, load_dictionary
from .explore import MultiSpace

__all__ = ["Embeddings", "EvalResult", "MultiSpace", "evaluate", "load_dictionary"]
__version__ = "2.0.0"
