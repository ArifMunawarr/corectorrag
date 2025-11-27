"""Embedding module using SentenceTransformers."""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

from config import config


class EmbeddingModel:
    """Wrapper for SentenceTransformer embedding model."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
        print(f"✓ Embedding model loaded: {self.model_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = np.array(self.model.encode(text1))
        emb2 = np.array(self.model.encode(text2))
        
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(cosine_sim)


# Singleton instance
_embedding_model = None


def get_embedding_model() -> EmbeddingModel:
    """Get or create singleton embedding model instance."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model
