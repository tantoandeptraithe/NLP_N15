#!/usr/bin/env python3
"""Evaluate STT (WER), MT (COMET/BLEU/chrF++/Proper-noun), and Audio (SIM/UTMOS/CER/Prosody/Accent).

Supports incremental per-utterance checkpointing so long audio evaluations on Colab
can be resumed seamlessly if interrupted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from s2st.io import ProjectPaths
from s2st.manifest import append_manifest_record, deduplicate_manifest, get_completed_keys, read_manifest


def run_audio_eval_resumable(
    input_manifest: Path,
    output_manifest: Path,
    eval_sim: bool = True,
    eval_mos: bool = True,
    eval_cer: bool = True,
    eval_prosody: bool = True,
    eval_accent_probe: bool = True,
) -> List[Dict[str, Any]]:
    """Run all automated audio metrics over `input_manifest` with per-item resume."""
    records = read_manifest(input_manifest)
    completed = get_completed_keys(
        output_manifest,
        key_field="utt_id",
        required_non_empty_fields=["eval_completed"],
    )
    pending = [r for r in records if str(r.get("utt_id")) not in completed]
    print(f"[eval_audio] Total={len(records)} | Done={len(completed)} | Pending={len(pending)}")
    if not pending:
        return read_manifest(output_manifest)

    sim_scorer = None
    mos_scorer = None
    cer_scorer = None
    accent_probe = None

    if eval_sim:
        from s2st.eval.spk_sim import SpeakerSimilarityScorer

        sim_scorer = SpeakerSimilarityScorer()
    if eval_mos:
        from s2st.eval.mos_auto import AutomatedMOSScorer

        mos_scorer = AutomatedMOSScorer()
    if eval_cer:
        from s2st.eval.intelligibility import VietnameseIntelligibilityScorer

        cer_scorer = VietnameseIntelligibilityScorer()
    if eval_accent_probe:
        from s2st.eval.accent_probe import AccentEmbeddingProbe

        accent_probe = AccentEmbeddingProbe()
    if eval_prosody:
        from s2st.eval.prosody import extract_prosody_features

    for rec in tqdm(pending, desc="Evaluating Audio"):
        tts_path = rec.get("tts_audio_path")
        src_path = rec.get("source_audio_path")
        metrics: Dict[str, Any] = {}

        if sim_scorer and tts_path:
            metrics.update(sim_scorer.score_record(rec))
        if mos_scorer and tts_path:
            metrics["utmos"] = mos_scorer.score_file(tts_path)
        if cer_scorer and tts_path and rec.get("tts_language", "vi") == "vi":
            target_vi = str(rec.get("tts_input_text") or rec.get("mt_vi_text") or "")
            metrics.update(cer_scorer.score_file(tts_path, target_vi))
        if eval_prosody and tts_path:
            tts_pros = extract_prosody_features(
                tts_path, text=str(rec.get("tts_input_text") or "")
            )
            for k, v in tts_pros.items():
                metrics[f"tts_{k}"] = v
            if src_path and Path(src_path).exists():
                src_pros = extract_prosody_features(
                    src_path, text=str(rec.get("gold_en_text") or "")
                )
                for k, v in src_pros.items():
                    metrics[f"src_{k}"] = v
        if accent_probe and tts_path and src_path:
            metrics["accent_probe_sim_to_src"] = accent_probe.score_pair(tts_path, src_path)

        merged = {**rec, **metrics, "eval_completed": True}
        append_manifest_record(merged, output_manifest)

    return deduplicate_manifest(output_manifest, key_field="utt_id")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automated evaluations.")
    parser.add_argument("--mode", required=True, choices=["wer", "mt", "audio"])
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-file", required=True, help="Output JSON/JSONL path")
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--no-comet", action="store_true")
    parser.add_argument("--no-cer", action="store_true")
    parser.add_argument("--no-accent-probe", action="store_true")
    args = parser.parse_args()

    paths = ProjectPaths(args.paths_config)
    paths.ensure_dirs()

    in_path = Path(args.input_manifest)
    if not in_path.exists():
        in_path = paths.drive_root / args.input_manifest
    out_path = Path(args.output_file)
    if not out_path.is_absolute():
        out_path = paths.drive_root / args.output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "wer":
        from s2st.eval.wer import compute_wer_breakdown

        res = compute_wer_breakdown(read_manifest(in_path))
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval:wer] Overall WER={res['overall_wer']} | By accent={res['by_accent']}")
        return

    if args.mode == "mt":
        from s2st.eval.mt_metrics import evaluate_mt_manifest

        res = evaluate_mt_manifest(read_manifest(in_path), use_comet=not args.no_comet)
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[eval:mt] BLEU={res.get('sacrebleu')} | chrF++={res.get('chrf2_pp')} | COMET={res.get('comet_wmt22_da')}"
        )
        return

    if args.mode == "audio":
        run_audio_eval_resumable(
            input_manifest=in_path,
            output_manifest=out_path,
            eval_cer=not args.no_cer,
            eval_accent_probe=not args.no_accent_probe,
        )


if __name__ == "__main__":
    main()
