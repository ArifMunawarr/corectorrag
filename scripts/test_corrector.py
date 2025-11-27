"""Test script for STT Corrector."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.corrector import get_corrector


def main():
    print("Testing STT Corrector...\n")
    
    # Initialize corrector
    corrector = get_corrector()
    
    # Load knowledge base
    kb_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "knowledge_base.json"
    )
    corrector.load_knowledge_base(kb_path)
    
    # Test cases
    test_cases = [
        "start eating",      # Should correct to "start meeting"
        "send mail",         # Should correct to "send email"
        "screen chair",      # Should correct to "screen share"
        "open fail",         # Should correct to "open file"
        "join eating",       # Should correct to "join meeting"
        "mute mike",         # Should correct to "mute microphone"
        "hello world",       # Should remain unchanged (no match)
    ]
    
    print("=" * 60)
    print(f"{'Input':<20} | {'Output':<20} | {'Method':<15}")
    print("=" * 60)
    
    for test_input in test_cases:
        result = corrector.correct(test_input, use_llm=False)  # Direct match only for speed
        corrected = result["corrected_text"]
        method = result["method"]
        
        # Color coding for terminal
        if result["correction_made"]:
            status = "✓"
        else:
            status = "-"
        
        print(f"{status} {test_input:<18} | {corrected:<20} | {method:<15}")
    
    print("=" * 60)
    print("\nTest completed!")
    
    # Get stats
    stats = corrector.get_stats()
    print(f"\nVector Store Stats: {stats['vector_store']['total_documents']} documents")
    print(f"LLM Model: {stats['llm_model']}")
    print(f"LLM Connected: {stats['llm_connected']}")


if __name__ == "__main__":
    main()
