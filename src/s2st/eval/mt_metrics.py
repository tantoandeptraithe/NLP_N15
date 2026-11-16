"""Machine Translation metrics: COMET (`Unbabel/wmt22-comet-da`), sacreBLEU, chrF++, and proper-noun accuracy."""

from __future__ import annotations

import re
from typing import Any, Dict, List


_CAP_WORDS_RE = re.compile(r"\b[A-Z][a-zA-Z0-9_-]{2,}\b|\b\d+(?:[.,]\d+)?\b")


def extract_proper_nouns_and_numbers(en_text: str) -> List[str]:
    """Extract non-initial capitalized words and numbers that should be preserved in translation."""
    words = en_text.strip().split()
    if len(words) <= 1:
        return []
    tail = " ".join(words[1:])
    return _CAP_WORDS_RE.findall(tail)


def score_proper_noun_preservation(en_src: str, vi_hyp: str) -> float | None:
    """Return fraction of source proper nouns/numbers preserved in `vi_hyp` (or None if none present)."""
    entities = extract_proper_nouns_and_numbers(en_src)
    if not entities:
        return None
    hyp_lower = vi_hyp.lower()
    hits = sum(1 for ent in entities if ent.lower() in hyp_lower)
    return float(hits) / len(entities)


def evaluate_mt_manifest(
    records: List[Dict[str, Any]],
    comet_model_id: str = "Unbabel/wmt22-comet-da",
    use_comet: bool = True,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Compute sacreBLEU, chrF++, proper-noun accuracy, and neural COMET score."""
    import sacrebleu

    valid = [
        r
        for r in records
        if r.get("mt_vi_text") and r.get("ref_vi_text") and r.get("gold_en_text")
    ]
    if not valid:
        return {"error": "No records with both `mt_vi_text` and `ref_vi_text` found."}

    srcs = [str(r.get("mt_source_text") or r.get("gold_en_text")).strip() for r in valid]
    hyps = [str(r["mt_vi_text"]).strip() for r in valid]
    refs = [str(r["ref_vi_text"]).strip() for r in valid]

    bleu_res = sacrebleu.corpus_bleu(hyps, [refs])
    chrf_res = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)  # chrF++

    pn_scores = [
        score_proper_noun_preservation(str(r["gold_en_text"]), str(r["mt_vi_text"]))
        for r in valid
    ]
    pn_valid = [s for s in pn_scores if s is not None]
    pn_acc = float(sum(pn_valid) / len(pn_valid)) if pn_valid else None

    comet_system_score: float | None = None
    comet_seg_scores: List[float | None] = [None] * len(valid)

    if use_comet:
        try:
            import torch
            from comet import download_model, load_from_checkpoint

            ckpt_path = download_model(comet_model_id)
            model = load_from_checkpoint(ckpt_path)
            data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(srcs, hyps, refs)]
            gpus = 1 if torch.cuda.is_available() else 0
            pred = model.predict(data, batch_size=batch_size, gpus=gpus)
            comet_system_score = float(pred.system_score)
            comet_seg_scores = [round(float(x), 4) for x in pred.scores]
        except Exception as exc:
            print(f"[eval_mt] Warning: COMET evaluation skipped/failed ({exc})")

    per_sentence: List[Dict[str, Any]] = []
    for rec, c_score, pn in zip(valid, comet_seg_scores, pn_scores):
        s_bleu = sacrebleu.sentence_bleu(str(rec["mt_vi_text"]), [str(rec["ref_vi_text"])]).score
        per_sentence.append(
            {
                "utt_id": rec.get("utt_id"),
                "mt_model_name": rec.get("mt_model_name"),
                "sentence_bleu": round(float(s_bleu), 2),
                "comet": c_score,
                "proper_noun_acc": round(pn, 4) if pn is not None else None,
            }
        )

    return {
        "num_sentences": len(valid),
        "sacrebleu": round(float(bleu_res.score), 2),
        "chrf2_pp": round(float(chrf_res.score), 2),
        "comet_wmt22_da": round(comet_system_score, 4) if comet_system_score is not None else None,
        "proper_noun_accuracy": round(pn_acc, 4) if pn_acc is not None else None,
        "proper_noun_error_rate": round(1.0 - pn_acc, 4) if pn_acc is not None else None,
        "per_sentence": per_sentence,
    }
