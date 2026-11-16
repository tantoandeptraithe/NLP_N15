"""Vietnamese TTS Intelligibility evaluation (CER / Syllable WER) via `vinai/PhoWhisper-large`."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

_VI_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_vietnamese_for_asr(text: str) -> str:
    """Lowercase and strip punctuation while preserving all Vietnamese tone diacritics."""
    cleaned = _VI_PUNCT_RE.sub(" ", text.lower())
    return " ".join(cleaned.split())


class VietnameseIntelligibilityScorer:
    """Back-transcribes synthesized Vietnamese audio with `vinai/PhoWhisper-large` and computes CER/WER."""

    def __init__(
        self, model_id: str = "vinai/PhoWhisper-large", device: str = "cuda"
    ) -> None:
        self.model_id = model_id
        self.device_name = device
        self._pipe: Any = None

    def load(self) -> None:
        if self._pipe is not None:
            return
        import torch
        from transformers import pipeline

        use_cuda = self.device_name == "cuda" and torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            torch_dtype=dtype,
            device=0 if use_cuda else -1,
        )
        print(f"[Intelligibility] Loaded `{self.model_id}`")

    def score_file(self, audio_path: str | Path, target_vi_text: str) -> Dict[str, Any]:
        import jiwer

        if not audio_path or not Path(audio_path).exists() or not target_vi_text.strip():
            return {}
        self.load()
        res = self._pipe(str(audio_path))
        hyp_raw = str(res.get("text", "")).strip()
        ref_norm = normalize_vietnamese_for_asr(target_vi_text)
        hyp_norm = normalize_vietnamese_for_asr(hyp_raw)
        if not ref_norm:
            return {}
        return {
            "back_asr_vi_text": hyp_raw,
            "vi_syllable_wer": round(float(jiwer.wer(ref_norm, hyp_norm)), 4),
            "vi_cer": round(float(jiwer.cer(ref_norm, hyp_norm)), 4),
        }
