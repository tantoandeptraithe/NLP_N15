"""Word Error Rate (WER) evaluation stratified by speaker and accent."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List

_PUNCT_RE = re.compile(r"[^\w\s']")


def normalize_english_for_wer(text: str) -> str:
    """Normalize English transcript (Whisper EnglishTextNormalizer if installed, else fallback)."""
    try:
        from whisper_normalizer.english import EnglishTextNormalizer

        normalizer = EnglishTextNormalizer()
        return str(normalizer(text)).strip()
    except Exception:
        cleaned = _PUNCT_RE.sub(" ", text.lower())
        return " ".join(cleaned.split())


def compute_wer_breakdown(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute overall, per-accent, and per-speaker WER using `jiwer`."""
    import jiwer

    refs_all: List[str] = []
    hyps_all: List[str] = []
    by_accent: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"refs": [], "hyps": []})
    by_speaker: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"refs": [], "hyps": []})
    per_utt: List[Dict[str, Any]] = []

    for rec in records:
        ref = normalize_english_for_wer(str(rec.get("gold_en_text", "")))
        hyp = normalize_english_for_wer(str(rec.get("stt_en_text", "")))
        if not ref:
            continue
        utt_wer = float(jiwer.wer(ref, hyp))
        utt_cer = float(jiwer.cer(ref, hyp))

        accent = str(rec.get("accent", "Unknown"))
        spk = str(rec.get("speaker_id", "Unknown"))
        refs_all.append(ref)
        hyps_all.append(hyp)
        by_accent[accent]["refs"].append(ref)
        by_accent[accent]["hyps"].append(hyp)
        by_speaker[spk]["refs"].append(ref)
        by_speaker[spk]["hyps"].append(hyp)

        per_utt.append(
            {
                "utt_id": rec.get("utt_id"),
                "speaker_id": spk,
                "accent": accent,
                "wer": round(utt_wer, 4),
                "cer": round(utt_cer, 4),
                "norm_ref": ref,
                "norm_hyp": hyp,
            }
        )

    overall_wer = float(jiwer.wer(refs_all, hyps_all)) if refs_all else 0.0
    accent_summary = {
        acc: {
            "num_utterances": len(vals["refs"]),
            "wer": round(float(jiwer.wer(vals["refs"], vals["hyps"])), 4),
            "cer": round(float(jiwer.cer(vals["refs"], vals["hyps"])), 4),
        }
        for acc, vals in sorted(by_accent.items())
    }
    speaker_summary = {
        spk: {
            "num_utterances": len(vals["refs"]),
            "wer": round(float(jiwer.wer(vals["refs"], vals["hyps"])), 4),
        }
        for spk, vals in sorted(by_speaker.items())
    }

    return {
        "overall_wer": round(overall_wer, 4),
        "num_evaluated": len(refs_all),
        "by_accent": accent_summary,
        "by_speaker": speaker_summary,
        "per_utterance": per_utt,
    }
