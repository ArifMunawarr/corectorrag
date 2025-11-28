"""Main STT Corrector RAG module."""

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Dict, Any, List, Optional

from src.vector_store import get_vector_store, VectorStore
from config import config


class STTCorrector:
    """RAG-based Speech-to-Text Corrector."""
    
    def __init__(self):
        self.vector_store: VectorStore = get_vector_store()
    
    def correct(
        self,
        input_text: str,
        use_llm: bool = False,
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
            if use_llm:
                llm_text = self._call_llm_normalize(input_text, [])
                if llm_text:
                    result["corrected_text"] = llm_text
                    result["correction_made"] = llm_text != input_text
                    result["method"] = "llm"
            return result
        
        # Step 2: Check for high-confidence direct match (berbasis similarity embedding)
        best_match = candidates[0]
        direct_threshold = getattr(config, "DIRECT_MATCH_THRESHOLD", 0.7)

        if not use_llm:
            if best_match["similarity"] >= direct_threshold:
                # High confidence - use direct match
                result["corrected_text"] = best_match["correct_phrase"]
                result["correction_made"] = True
                result["method"] = "direct_match"
                result["confidence"] = best_match["similarity"]
                return result
            
            # Tidak ada LLM: jika tidak lolos direct match, kembalikan teks asli
            return result

        base_text = input_text
        if best_match["similarity"] >= direct_threshold:
            base_text = best_match["correct_phrase"]
            result["confidence"] = best_match["similarity"]
            result["method"] = "direct_match"

        llm_text = self._call_llm_normalize(base_text, candidates)
        if llm_text:
            result["corrected_text"] = llm_text
            result["correction_made"] = llm_text != input_text
            if result["method"] == "none":
                result["method"] = "llm"
            else:
                result["method"] = "llm_with_rag"
            
            # Post-processing: Pastikan frasa dari knowledge base tetap exact match
            for candidate in candidates:
                correct_phrase = candidate["correct_phrase"]
                pattern = re.compile(re.escape(correct_phrase), re.IGNORECASE)
                result["corrected_text"] = pattern.sub(correct_phrase, result["corrected_text"])
        
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

                # Strip punctuation dari token untuk search
                clean_tokens = [re.sub(r'[^\w\s-]', '', tok) for tok in tokens[i : i + size]]
                phrase = " ".join(clean_tokens)
                
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

        # Jika tidak ada replacement, fallback ke koreksi berbasis kalimat penuh
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
                # Ambil punctuation dari token terakhir yang diganti
                last_token_orig = tokens[end_idx - 1]
                trailing_punct = re.findall(r'[^\w\s-]+$', last_token_orig)
                
                # Tambahkan replacement tokens
                corrected_tokens.extend(rep_tokens[:-1] if len(rep_tokens) > 1 else [])
                # Token terakhir + punctuation asli
                last_rep = rep_tokens[-1] if rep_tokens else ""
                if trailing_punct:
                    last_rep += trailing_punct[0]
                corrected_tokens.append(last_rep)
                
                applied_candidates.append(best)
                i = end_idx
            else:
                corrected_tokens.append(tokens[i])
                i += 1

        corrected_text = " ".join(corrected_tokens)

        final_text = corrected_text
        method = "ngram_direct_match"
        if use_llm:
            llm_text = self._call_llm_normalize(corrected_text, applied_candidates)
            if llm_text:
                final_text = llm_text
                method = "ngram_direct_match+llm"
                
                # Post-processing: Pastikan frasa dari knowledge base tetap exact match
                # (LLM kadang mengubah kapitalisasi)
                for candidate in applied_candidates:
                    correct_phrase = candidate["correct_phrase"]
                    # Case-insensitive replace dengan exact casing dari knowledge base
                    pattern = re.compile(re.escape(correct_phrase), re.IGNORECASE)
                    final_text = pattern.sub(correct_phrase, final_text)

        return {
            "input_text": input_text,
            "corrected_text": final_text,
            "candidates": applied_candidates,
            "correction_made": final_text != input_text,
            "method": method,
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
    
    def _call_llm_normalize(
        self,
        input_text: str,
        candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        base_url = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")
        model_name = getattr(config, "LLM_MODEL", "llama3.2:latest")  

        parts: List[str] = [
            "Kamu adalah sistem koreksi ejaan untuk teks speech-to-text bahasa Indonesia.",
            "",
            "ATURAN PENTING:",
            "- HANYA perbaiki kata yang jelas typo atau salah dengar",
            "- JANGAN tambah kata baru yang tidak ada di input",
            "- JANGAN hapus kata apapun dari input - semua kata harus tetap ada dalam output (meskipun dikoreksi)",
            "- JANGAN ganti kata dengan kata lain yang mengubah makna kalimat",
            "- JANGAN ubah struktur kalimat atau urutan kata",
            "- BOLEH memperbaiki kapitalisasi dan spasi yang salah (misalnya 'kitMulai' → 'kita mulai')",
            "- KHUSUS untuk istilah/\"Frasa benar\" dari knowledge base, JANGAN ubah penulisan sama sekali (huruf besar/kecil, spasi, tanda hubung)",
            "- Koreksi harus minimal: hanya perbaiki huruf/ejaan, kapitalisasi, dan spasi yang salah",
            "- OUTPUT: Hanya tulis teks hasil koreksi, TANPA label, TANPA penjelasan",
            "",
            "POLA KESALAHAN UMUM:",
            "- Huruf hilang: 'say' → 'saya', 'kit' → 'kita', 'bso' → 'baso', 'beso' → 'besok', 'tida' → 'tidak'",
            "- Salah dengar: 'k' → 'ke', 'kmarin' → 'kemarin', 'gmana' → 'gimana'",
            "- Singkatan lisan: 'bru' → 'baru', 'org' → 'orang', 'tgl' → 'tanggal'",
            "- Spasi/kapitalisasi: 'kitMulai' → 'kita mulai', 'kitaMulai' → 'kita mulai'",
        ]

        # Candidates jadi contoh belajar, bukan hardcode
        if candidates:
            parts.append("")
            parts.append("ISTILAH KHUSUS yang mungkin salah dengar:")
            parts.append("Gunakan FRASA BENAR persis seperti tertulis, jangan diubah penulisannya (huruf besar/kecil, spasi, tanda hubung).")
            # Ambil max 5-10 contoh teratas aja
            top_candidates = candidates[:10] if len(candidates) > 10 else candidates
            parts.append(self._format_candidates(top_candidates))

        parts.append("")
        parts.append("===== CONTOH POLA KOREKSI =====")
        parts.append("Perbaiki huruf/ejaan, kapitalisasi, dan spasi yang salah; jangan ganti makna kalimat:")
        parts.append("")
        parts.append("'say' → 'saya' (huruf 'a' hilang)")
        parts.append("'kit' → 'kita' (huruf 'a' hilang)")
        parts.append("'beso' → 'besok' (huruf 'k' hilang)")
        parts.append("'kmarin' → 'kemarin' (huruf 'e' hilang)")
        parts.append("'gmana' → 'gimana' (huruf 'i' hilang)")
        parts.append("'dgan' → 'dengan' (huruf 'en' hilang)")
        parts.append("'bru' → 'baru' (huruf 'a' hilang)")
        parts.append("'kitMulai' → 'kita mulai' (spasi + huruf)")
        parts.append("")
        parts.append("Contoh kalimat:")
        parts.append("'kmarin kit rapat dgan manajer' → 'kemarin kita rapat dengan manajer'")
        parts.append("'beso kitaMulai pelatihan nek ji, tida di kantor' → 'besok kita mulai pelatihan Next-G tidak di kantor'")
        parts.append("")
        parts.append("INGAT: Semua kata dalam input harus ada dalam output (meskipun dikoreksi ejaannya)!")
        parts.append("")
        parts.append("===== TUGAS KAMU =====")
        parts.append("")
        parts.append("Teks: " + input_text)
        parts.append("Koreksi:")

        prompt = "\n".join(parts)

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.05,
                "top_p": 0.1,
                "num_predict": 150,
            },
        }
        
        data = json.dumps(payload).encode("utf-8")
        url = f"{base_url.rstrip('/')}/api/generate"

        try:
            request_obj = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request_obj, timeout=30) as response:
                raw = response.read().decode("utf-8")

            resp_json = json.loads(raw)
            text = resp_json.get("response", "").strip()
            if not text:
                return None

            if (text.startswith("\"") and text.endswith("\"")) or (
                text.startswith("'") and text.endswith("'")
            ):
                text = text[1:-1].strip()

            return text
        except Exception:
            logging.exception("Failed to call LLM for normalization")
            return None
    
    def correct_batch(
        self,
        texts: List[str],
        use_llm: bool = False
    ) -> List[Dict[str, Any]]:
        """Correct multiple STT texts."""
        return [self.correct(text, use_llm=False) for text in texts]
    
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
            "llm_enabled": True,
            "llm_model": getattr(config, "LLM_MODEL", None),
            "llm_connected": False,
        }


# Singleton instance
_corrector = None


def get_corrector() -> STTCorrector:
    """Get or create singleton corrector instance."""
    global _corrector
    if _corrector is None:
        _corrector = STTCorrector()
    return _corrector
