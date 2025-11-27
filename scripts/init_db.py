"""Script to initialize the vector database with knowledge base."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store import get_vector_store


def main():
    print("Initializing STT Corrector Database...")
    
    # Get vector store
    vs = get_vector_store()
    
    # Load knowledge base
    kb_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "knowledge_base.json"
    )
    
    if os.path.exists(kb_path):
        count = vs.load_knowledge_base(kb_path)
        print(f"✓ Database initialized with {count} corrections")
    else:
        print(f"✗ Knowledge base not found at: {kb_path}")
        return 1
    
    # Print stats
    stats = vs.get_stats()
    print(f"✓ Total documents in vector store: {stats['total_documents']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
