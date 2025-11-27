"""Main STT Corrector RAG module."""

from typing import Dict, Any, List, Optional

from src.vector_store import get_vector_store, VectorStore
from src.llm import get_llm, OllamaLLM
from config import config


SYSTEM_PROMPT = """Kamu adalah asisten yang bertugas mengoreksi kesalahan Speech-to-Text (STT).
Tugasmu adalah menganalisis teks hasil STT yang mungkin salah dengar dan mengoreksinya berdasarkan daftar koreksi yang diberikan.

Aturan:
1. Jika ada kecocokan dengan daftar koreksi, gunakan frasa yang benar
2. Jika tidak ada kecocokan, kembalikan teks asli tanpa perubahan
3. Berikan hanya hasil koreksi, tanpa penjelasan tambahan
4. Pertahankan kapitalisasi dan tanda baca yang sesuai"""


USER_PROMPT_TEMPLATE = """Teks STT yang perlu dikoreksi: "{input_text}"

Daftar kemungkinan koreksi berdasarkan knowledge base:
{corrections}

Berdasarkan daftar koreksi di atas, apa hasil koreksi yang tepat untuk teks STT tersebut?
Jawab hanya dengan teks hasil koreksi, tanpa penjelasan."""


class STTCorrector:
    """RAG-based Speech-to-Text Corrector."""
    
    def __init__(self):
        self.vector_store: VectorStore = get_vector_store()
        self.llm: OllamaLLM = get_llm()
    
    def correct(
        self,
        input_text: str,
        use_llm: bool = True,
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        Correct STT text using RAG.
        
        Args:
            input_text: The STT text to correct
            use_llm: Whether to use LLM for final correction (False = direct match only)
            top_k: Number of candidates to retrieve
            
        Returns:
            Dictionary with correction results
        """
        # Step 1: Retrieve relevant corrections from knowledge base
        candidates = self.vector_store.search(
            query=input_text,
            top_k=top_k or config.TOP_K_RESULTS
        )
        
        result = {
            "input_text": input_text,
            "corrected_text": input_text,
            "candidates": candidates,
            "correction_made": False,
            "method": "none"
        }
        
        if not candidates:
            return result
        
        # Step 2: Check for high-confidence direct match (berbasis similarity embedding)
        best_match = candidates[0]
        direct_threshold = getattr(config, "DIRECT_MATCH_THRESHOLD", 0.7)
        if best_match["similarity"] >= direct_threshold:
            # High confidence - use direct match
            result["corrected_text"] = best_match["correct_phrase"]
            result["correction_made"] = True
            result["method"] = "direct_match"
            result["confidence"] = best_match["similarity"]
            return result
        
        # Step 3: Use LLM for ambiguous cases
        if use_llm and candidates:
            corrections_text = self._format_candidates(candidates)
            
            prompt = USER_PROMPT_TEMPLATE.format(
                input_text=input_text,
                corrections=corrections_text
            )
            
            llm_response = self.llm.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.1
            )
            
            corrected = llm_response.strip().strip('"').strip("'")
            
            if corrected and corrected.lower() != input_text.lower():
                result["corrected_text"] = corrected
                result["correction_made"] = True
                result["method"] = "llm_rag"
                result["confidence"] = best_match["similarity"]
        
        return result
    
    def correct_in_text(
        self,
        input_text: str,
        use_llm: bool = False,
        max_ngram: int = 3
    ) -> Dict[str, Any]:
        """Correct misheard phrases inside a longer text using n-gram search.

        Contoh: "tolong stat meeting jam tiga" -> "tolong start meeting jam tiga".
        """
        tokens = input_text.split()
        if not tokens:
            return {
                "input_text": input_text,
                "corrected_text": input_text,
                "candidates": [],
                "correction_made": False,
                "method": "none",
            }

        n = len(tokens)
        ngram_sizes = list(range(min(max_ngram, n), 0, -1))

        direct_threshold = getattr(config, "DIRECT_MATCH_THRESHOLD", 0.7)
        replacements = {}  # start_idx -> (end_idx, replacement_tokens, best_match)
        used_indices = set()

        # Cari n-gram yang mirip dengan entri di knowledge base
        for size in ngram_sizes:
            if size <= 0:
                continue
            for i in range(0, n - size + 1):
                # Lewati jika posisi ini sudah termasuk dalam replacement lain
                if any(idx in used_indices for idx in range(i, i + size)):
                    continue

                phrase = " ".join(tokens[i : i + size])
                candidates = self.vector_store.search(query=phrase, top_k=1)
                if not candidates:
                    continue

                best = candidates[0]
                if best["similarity"] < direct_threshold:
                    continue

                correct_phrase = best["correct_phrase"]
                # Jika sudah sama (case-insensitive), tidak perlu diganti
                if correct_phrase.lower() == phrase.lower():
                    continue

                # Daftarkan replacement untuk n-gram ini
                replacement_tokens = correct_phrase.split()
                replacements[i] = (i + size, replacement_tokens, best)
                for idx in range(i, i + size):
                    used_indices.add(idx)

        # Jika tidak ada replacement, fallback ke koreksi biasa
        if not replacements:
            base = self.correct(input_text=input_text, use_llm=use_llm)
            return base

        # Bangun kembali teks dengan replacement
        corrected_tokens: List[str] = []
        i = 0
        applied_candidates: List[Dict[str, Any]] = []

        while i < n:
            if i in replacements:
                end_idx, rep_tokens, best = replacements[i]
                corrected_tokens.extend(rep_tokens)
                applied_candidates.append(best)
                i = end_idx
            else:
                corrected_tokens.append(tokens[i])
                i += 1

        corrected_text = " ".join(corrected_tokens)

        return {
            "input_text": input_text,
            "corrected_text": corrected_text,
            "candidates": applied_candidates,
            "correction_made": corrected_text != input_text,
            "method": "ngram_direct_match",
        }
    
    def _format_candidates(self, candidates: List[Dict]) -> str:
        """Format candidates for LLM prompt."""
        lines = []
        for i, c in enumerate(candidates, 1):
            mistakes = ", ".join(c["common_mistakes"][:3])
            lines.append(
                f"{i}. Frasa benar: \"{c['correct_phrase']}\" "
                f"(kesalahan umum: {mistakes}) - Konteks: {c['context']}"
            )
        return "\n".join(lines)
    
    def correct_batch(
        self,
        texts: List[str],
        use_llm: bool = True
    ) -> List[Dict[str, Any]]:
        """Correct multiple STT texts."""
        return [self.correct(text, use_llm=use_llm) for text in texts]
    
    def add_correction(
        self,
        correct_phrase: str,
        common_mistakes: List[str],
        context: str = "",
        category: str = ""
    ) -> str:
        """Add a new correction to the knowledge base."""
        return self.vector_store.add_correction(
            correct_phrase=correct_phrase,
            common_mistakes=common_mistakes,
            context=context,
            category=category
        )
    
    def load_knowledge_base(self, filepath: str) -> int:
        """Load corrections from JSON file."""
        return self.vector_store.load_knowledge_base(filepath)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "vector_store": self.vector_store.get_stats(),
            "llm_model": self.llm.model_name,
            "llm_connected": self.llm.check_connection()
        }


# Singleton instance
_corrector = None


def get_corrector() -> STTCorrector:
    """Get or create singleton corrector instance."""
    global _corrector
    if _corrector is None:
        _corrector = STTCorrector()
    return _corrector
