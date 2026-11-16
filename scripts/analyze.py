#!/usr/bin/env python3
"""Aggregate evaluation manifests and run statistical hypothesis tests for E1-E4."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from s2st.eval.stats import bootstrap_ci_mean, paired_wilcoxon_and_effect_size, spearman_correlation
from s2st.manifest import read_manifest


METRICS_TO_SUMMARIZE = [
    "sim_ecapa_prompt",
    "sim_ecapa_heldout",
    "sim_wavlm_prompt",
    "sim_wavlm_heldout",
    "utmos",
    "vi_cer",
    "vi_syllable_wer",
    "accent_probe_sim_to_src",
    "tts_rtf",
]


def summarize_evaluated_manifests(manifest_paths: List[Path]) -> Dict[str, Any]:
    all_records: List[Dict[str, Any]] = []
    for p in manifest_paths:
        all_records.extend(read_manifest(p))

    by_sys: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_sys_accent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for rec in all_records:
        sys_key = f"{rec.get('tts_model_name', 'tts')}__{rec.get('tts_condition_tag', 'default')}"
        acc = str(rec.get("accent", "Unknown"))
        by_sys[sys_key].append(rec)
        by_sys_accent[f"{sys_key} | {acc}"].append(rec)

    system_summary: Dict[str, Any] = {}
    for sys_key, recs in sorted(by_sys.items()):
        system_summary[sys_key] = {
            m: bootstrap_ci_mean([float(r[m]) for r in recs if r.get(m) is not None])
            for m in METRICS_TO_SUMMARIZE
        }

    accent_summary: Dict[str, Any] = {}
    for sa_key, recs in sorted(by_sys_accent.items()):
        accent_summary[sa_key] = {
            m: bootstrap_ci_mean([float(r[m]) for r in recs if r.get(m) is not None])
            for m in METRICS_TO_SUMMARIZE
        }

    # Test Hypothesis H3 (Prosody transfer: speaking rate vs F0 IQR correlation from EN source to VI TTS)
    h3_prosody_transfer: Dict[str, Any] = {}
    for sys_key, recs in sorted(by_sys.items()):
        src_rate = [r.get("src_speaking_rate_w_per_s") for r in recs]
        tts_rate = [r.get("tts_speaking_rate_w_per_s") for r in recs]
        src_f0_iqr = [r.get("src_f0_iqr_semitones") for r in recs]
        tts_f0_iqr = [r.get("tts_f0_iqr_semitones") for r in recs]
        h3_prosody_transfer[sys_key] = {
            "speaking_rate_spearman": spearman_correlation(src_rate, tts_rate),
            "f0_iqr_spearman": spearman_correlation(src_f0_iqr, tts_f0_iqr),
        }

    # Paired Wilcoxon comparison if >= 2 systems share utt_id
    paired_tests: Dict[str, Any] = {}
    sys_names = sorted(by_sys.keys())
    if len(sys_names) >= 2:
        for i in range(len(sys_names)):
            for j in range(i + 1, len(sys_names)):
                s1, s2 = sys_names[i], sys_names[j]
                m1 = {str(r["utt_id"]): r for r in by_sys[s1]}
                m2 = {str(r["utt_id"]): r for r in by_sys[s2]}
                common_ids = sorted(set(m1.keys()) & set(m2.keys()))
                if not common_ids:
                    continue
                pair_key = f"{s1}__VS__{s2}"
                paired_tests[pair_key] = {}
                for m in ("sim_ecapa_heldout", "sim_wavlm_heldout", "utmos", "vi_cer"):
                    x = [m1[u].get(m) for u in common_ids]
                    y = [m2[u].get(m) for u in common_ids]
                    paired_tests[pair_key][m] = paired_wilcoxon_and_effect_size(x, y)

    # Identify worst-K samples by SIM-heldout for E4 qualitative error inspection
    scored_with_sim = [r for r in all_records if r.get("sim_ecapa_heldout") is not None]
    scored_with_sim.sort(key=lambda r: float(r["sim_ecapa_heldout"]))
    worst_cases = [
        {
            "utt_id": r.get("utt_id"),
            "speaker_id": r.get("speaker_id"),
            "accent": r.get("accent"),
            "system": f"{r.get('tts_model_name')}__{r.get('tts_condition_tag')}",
            "sim_ecapa_heldout": r.get("sim_ecapa_heldout"),
            "utmos": r.get("utmos"),
            "vi_cer": r.get("vi_cer"),
            "tts_audio_path": r.get("tts_audio_path"),
        }
        for r in scored_with_sim[:30]
    ]

    return {
        "system_summary": system_summary,
        "by_system_and_accent": accent_summary,
        "paired_wilcoxon_tests": paired_tests,
        "h3_prosody_transfer_correlations": h3_prosody_transfer,
        "e4_worst_30_sim_cases": worst_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze evaluated audio manifests.")
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    res = summarize_evaluated_manifests([Path(p) for p in args.manifests])
    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[analyze] Saved summary & statistical tests -> {out_p}")


if __name__ == "__main__":
    main()
