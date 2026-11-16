#!/usr/bin/env python3
"""Unified CLI for running any pipeline stage with Google Drive persistence and auto-resume.

Examples:
  # 1. Prepare VCTK dataset (smoke test: 2 speakers x 3 utterances)
  python scripts/run_stage.py --stage prepare_vctk --config configs/data/vctk.yaml --smoke

  # 2. Run STT (faster-whisper large-v3)
  python scripts/run_stage.py --stage stt --config configs/stt/faster_whisper_large_v3.yaml \
      --input-manifest processed/manifests/vctk_smoke.jsonl \
      --output-manifest outputs/stt/vctk_smoke_stt.jsonl

  # 3. Run MT (VinAI v2 / NLLB 1.3B / Qwen3.5-9B)
  python scripts/run_stage.py --stage mt --config configs/mt/vinai_v2.yaml \
      --input-manifest outputs/stt/vctk_smoke_stt.jsonl \
      --output-manifest outputs/mt/vctk_smoke_vinai.jsonl

  # 4. Run Zero-Shot TTS (VieNeu-TTS v3 Turbo or viXTTS)
  python scripts/run_stage.py --stage tts --config configs/tts/vieneu_v3turbo.yaml \
      --input-manifest outputs/mt/vctk_smoke_vinai.jsonl \
      --output-manifest outputs/tts/vctk_smoke_vieneu.jsonl \
      --condition-tag clone_8s_clean_denoise_on --ref-duration 8s --ref-quality clean
"""

from __future__ import annotations

import argparse
from pathlib import Path

from s2st.io import ProjectPaths, load_yaml
from s2st.runmeta import save_run_metadata


def _resolve_manifest_path(paths: ProjectPaths, p: str | None) -> Path:
    if not p:
        raise ValueError("Manifest path is required for this stage.")
    cand = Path(p)
    if cand.is_absolute() or cand.exists():
        return cand
    return paths.drive_root / cand


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a resumable S2ST pipeline stage.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["prepare_vctk", "prepare_fleurs", "prepare_globe", "stt", "mt", "tts", "e3_grid"],
    )
    parser.add_argument("--config", required=True, help="Path to stage YAML config.")
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--run-id", default="run_default")
    parser.add_argument("--input-manifest", default=None)
    parser.add_argument("--output-manifest", default=None)
    parser.add_argument("--source-text-field", default="stt_en_text")
    parser.add_argument("--condition-tag", default="clone_8s_clean_denoise_on")
    parser.add_argument("--ref-duration", default="8s", help="3s | 5s | 8s")
    parser.add_argument("--ref-quality", default="clean", help="clean | noisy_snr10db")
    parser.add_argument("--denoise-ref", choices=["true", "false", "auto"], default="auto")
    parser.add_argument("--preset-anchor", action="store_true")
    parser.add_argument("--same-lang-en", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run small 6-sample smoke test")
    args = parser.parse_args()

    paths = ProjectPaths(args.paths_config)
    paths.ensure_dirs()
    cfg = load_yaml(args.config)
    save_run_metadata(paths.get_dir("runs"), args.run_id, args.stage, cfg)

    if args.stage == "prepare_vctk":
        from s2st.data.prepare_vctk import prepare_vctk_dataset

        prepare_vctk_dataset(args.config, args.paths_config, smoke_test=args.smoke)
        return

    if args.stage == "prepare_fleurs":
        from s2st.data.prepare_fleurs import prepare_fleurs_flores_mt_benchmark

        vctk_m = _resolve_manifest_path(paths, args.input_manifest) if args.input_manifest else None
        prepare_fleurs_flores_mt_benchmark(
            args.config,
            args.paths_config,
            vctk_manifest_path=vctk_m,
            limit_flores=10 if args.smoke else None,
        )
        return

    if args.stage == "prepare_globe":
        from s2st.data.prepare_globe import prepare_globe_dataset

        prepare_globe_dataset(args.config, args.paths_config)
        return

    in_m = _resolve_manifest_path(paths, args.input_manifest)
    out_m = _resolve_manifest_path(paths, args.output_manifest)

    if args.stage == "stt":
        from s2st.stt.faster_whisper import run_stt_stage

        run_stt_stage(in_m, out_m, args.config)
        return

    if args.stage == "mt":
        from s2st.mt.base import run_mt_stage

        run_mt_stage(in_m, out_m, args.config, source_text_field=args.source_text_field)
        return

    if args.stage == "tts":
        from s2st.tts.base import run_tts_stage

        denoise_val = None if args.denoise_ref == "auto" else (args.denoise_ref == "true")
        run_tts_stage(
            input_manifest_path=in_m,
            output_manifest_path=out_m,
            tts_config_path=args.config,
            paths_cfg_path=args.paths_config,
            run_id=args.run_id,
            condition_tag=args.condition_tag,
            ref_duration_tag=args.ref_duration,
            ref_quality=args.ref_quality,
            denoise_ref=denoise_val,
            use_preset_anchor=args.preset_anchor,
            same_lang_en_control=args.same_lang_en,
        )
        return

    if args.stage == "e3_grid":
        # Run full E3 3x4 reference ablation grid automatically
        from s2st.tts.base import run_tts_stage

        e3_cfg = load_yaml("configs/experiments/e3_reference.yaml")
        durations = [f"{int(d)}s" for d in e3_cfg["factors"]["ref_durations_sec"]]
        q_conds = e3_cfg["factors"]["quality_conditions"]
        for d_tag in durations:
            for qc in q_conds:
                c_tag = f"e3_{d_tag}_{qc['name']}"
                sub_out = out_m.parent / f"{out_m.stem}__{c_tag}.jsonl"
                run_tts_stage(
                    input_manifest_path=in_m,
                    output_manifest_path=sub_out,
                    tts_config_path=args.config,
                    paths_cfg_path=args.paths_config,
                    run_id=args.run_id,
                    condition_tag=c_tag,
                    ref_duration_tag=d_tag,
                    ref_quality=qc["input_type"],
                    denoise_ref=bool(qc["vieneu_denoise"]),
                )


if __name__ == "__main__":
    main()
