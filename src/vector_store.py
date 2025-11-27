"""Vector store module using ChromaDB."""

import json
import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from config import config
from src.embeddings import get_embedding_model


class VectorStore:
    """ChromaDB-based vector store for STT corrections."""
    
    def __init__(self):
        # Ensure persist directory exists
        os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=config.CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"description": "STT correction knowledge base"}
        )
        
        self.embedding_model = get_embedding_model()
        print(f"✓ Vector store initialized: {config.COLLECTION_NAME}")
    
    def add_correction(
        self,
        correct_phrase: str,
        common_mistakes: List[str],
        context: str = "",
        category: str = "",
        doc_id: Optional[str] = None
    ) -> str:
        """Add a correction entry to the vector store."""
        # Generate unique ID
        doc_id = doc_id or f"correction_{hash(correct_phrase) % 10000}"
        
        # Create searchable text combining all variants
        searchable_texts = [correct_phrase] + common_mistakes
        
        # Create metadata
        metadata = {
            "correct_phrase": correct_phrase,
            "common_mistakes": json.dumps(common_mistakes),
            "context": context,
            "category": category,
            "type": "correction"
        }
        
        # Add each variant as a separate document pointing to the correct phrase
        for i, text in enumerate(searchable_texts):
            variant_id = f"{doc_id}_{i}"
            embedding = self.embedding_model.embed_text(text)
            
            self.collection.upsert(
                ids=[variant_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata]
            )
        
        return doc_id
    
    def search(
        self,
        query: str,
        top_k: int = None,
        threshold: float = None
    ) -> List[Dict[str, Any]]:
        """Search for corrections matching the query."""
        top_k = top_k or config.TOP_K_RESULTS
        threshold = threshold or config.SIMILARITY_THRESHOLD
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_text(query)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Process results
        corrections = []
        seen_phrases = set()
        
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                # Convert distance to similarity (ChromaDB returns L2 distance)
                similarity = 1 / (1 + distance)
                
                if similarity >= threshold:
                    metadata = results["metadatas"][0][i]
                    correct_phrase = metadata["correct_phrase"]
                    
                    # Avoid duplicates
                    if correct_phrase not in seen_phrases:
                        seen_phrases.add(correct_phrase)
                        corrections.append({
                            "correct_phrase": correct_phrase,
                            "matched_text": results["documents"][0][i],
                            "common_mistakes": json.loads(metadata["common_mistakes"]),
                            "context": metadata["context"],
                            "category": metadata["category"],
                            "similarity": similarity
                        })
        
        return corrections
    
    def load_knowledge_base(self, filepath: str) -> int:
        """Load corrections from a JSON knowledge base file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        count = 0
        for entry in data.get("corrections", []):
            self.add_correction(
                correct_phrase=entry["correct_phrase"],
                common_mistakes=entry.get("common_mistakes", []),
                context=entry.get("context", ""),
                category=entry.get("category", "")
            )
            count += 1
        
        print(f"✓ Loaded {count} corrections from knowledge base")
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "collection_name": config.COLLECTION_NAME,
            "total_documents": self.collection.count(),
            "persist_directory": config.CHROMA_PERSIST_DIR
        }
    
    def clear(self):
        """Clear all documents from the collection."""
        self.client.delete_collection(config.COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=config.COLLECTION_NAME,
            metadata={"description": "STT correction knowledge base"}
        )
        print("✓ Vector store cleared")


# Singleton instance
_vector_store = None


def get_vector_store() -> VectorStore:
    """Get or create singleton vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
