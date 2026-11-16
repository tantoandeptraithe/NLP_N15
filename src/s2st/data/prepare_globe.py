"""Prepare secondary multi-accent English evaluation subset from `MushanW/GLOBE_V2` (CC0-1.0)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from s2st.data.make_refs import build_speaker_reference_clips
from s2st.data.prepare_vctk import _decode_hf_audio_cell
from s2st.io import ProjectPaths, load_yaml, write_audio_atomic
from s2st.manifest import write_manifest_atomic


def prepare_globe_dataset(
    config_path: str | Path = "configs/data/globe.yaml",
    paths_cfg_path: str | Path = "configs/paths.yaml",
) -> Path:
    """Stream `MushanW/GLOBE_V2` and build a robustness evaluation manifest."""
    from datasets import load_dataset

    cfg = load_yaml(config_path)
    paths = ProjectPaths(paths_cfg_path)
    paths.ensure_dirs()

    sel = cfg.get("selection", {})
    target_accents = set(sel.get("target_accents", ["United States English", "England English"]))
    spk_per_accent = int(sel.get("speakers_per_accent", 6))
    utts_per_spk = int(sel.get("utterances_per_speaker", 3))
    min_words = int(sel.get("min_source_words", 8))
    target_sr = int(cfg.get("sample_rate", 44100))

    ds = load_dataset(cfg.get("hf_repo_id", "MushanW/GLOBE_V2"), split=cfg.get("hf_split", "test"), streaming=True)
    speaker_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    speaker_accent: Dict[str, str] = {}
    selected_by_accent: Dict[str, List[str]] = defaultdict(list)

    for row in ds:
        acc = str(row.get("accent", "")).strip()
        if acc not in target_accents:
            continue
        txt = str(row.get("transcript", row.get("text", ""))).strip()
        if len(txt.split()) < min_words:
            continue
        spk = str(row.get("speaker_id", "")).strip()
        if not spk:
            continue
        if spk not in speaker_accent:
            if len(selected_by_accent[acc]) >= spk_per_accent:
                continue
            speaker_accent[spk] = acc
            selected_by_accent[acc].append(spk)
        if len(speaker_rows[spk]) < utts_per_spk + 2:
            speaker_rows[spk].append(row)

    raw_dir = paths.get_dir("raw") / "globe_v2"
    refs_dir = paths.get_dir("refs") / "globe_v2"
    records: List[Dict[str, Any]] = []

    for spk, rows in speaker_rows.items():
        if len(rows) < utts_per_spk + 1:
            continue
        eval_rows = rows[:utts_per_spk]
        ref_rows = rows[utts_per_spk:]
        ref_paths: List[Path] = []
        for idx, r_row in enumerate(ref_rows):
            wav, sr = _decode_hf_audio_cell(r_row["audio"], target_sr=target_sr)
            rp = raw_dir / spk / f"{spk}_ref_{idx}.wav"
            write_audio_atomic(rp, wav, sr)
            ref_paths.append(rp)

        ref_bundle = build_speaker_reference_clips(
            candidate_audio_paths=ref_paths,
            output_dir=refs_dir,
            speaker_id=spk,
            durations_sec=(3.0, 5.0, 8.0),
            target_sr=target_sr,
            make_noisy_snr_db=None,
        )
        default_ref = ref_bundle.get("8s") or next(iter(ref_bundle.values()))

        for idx, e_row in enumerate(eval_rows):
            utt_id = f"globe_{spk}_{idx:02d}"
            wav, sr = _decode_hf_audio_cell(e_row["audio"], target_sr=target_sr)
            sp = raw_dir / spk / f"{utt_id}.wav"
            write_audio_atomic(sp, wav, sr)
            records.append(
                {
                    "utt_id": utt_id,
                    "dataset": "globe_v2",
                    "speaker_id": spk,
                    "accent": speaker_accent[spk],
                    "gender": str(e_row.get("gender", "U")),
                    "source_audio_path": str(sp),
                    "source_sample_rate": sr,
                    "source_duration_sec": round(len(wav) / sr, 3),
                    "gold_en_text": str(e_row.get("transcript", e_row.get("text", ""))).strip(),
                    "ref_audio_path": default_ref["clean_ref_path"],
                    "ref_duration_tag": default_ref["duration_tag"],
                    "heldout_same_speaker_audio_path": str(ref_paths[0]),
                    "refs_by_duration": ref_bundle,
                }
            )

    out_manifest = paths.get_dir("manifests") / "globe_robustness.jsonl"
    write_manifest_atomic(records, out_manifest)
    return out_manifest
