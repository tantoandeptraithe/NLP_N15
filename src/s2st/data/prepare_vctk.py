"""Download and prepare the VCTK US/UK controlled evaluation set & speaker references.

Uses the verified Parquet mirror `sanchit-gandhi/vctk` (CC BY 4.0) on Hugging Face,
fetching only the required shards via `huggingface_hub` + `pyarrow`/`datasets` so that
smoke tests (e.g. 2-4 speakers) complete in seconds without downloading all 11 GB.
"""

from __future__ import annotations

import io
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import soundfile as sf

from s2st.data.make_refs import build_speaker_reference_clips
from s2st.io import ProjectPaths, load_yaml, write_audio_atomic
from s2st.manifest import write_manifest_atomic


def _decode_hf_audio_cell(audio_cell: Any, target_sr: int = 48000):
    """Decode an audio cell from a Hugging Face Parquet row ({'bytes': ..., 'path': ...})."""
    import numpy as np

    if isinstance(audio_cell, dict):
        if audio_cell.get("array") is not None:
            arr = np.asarray(audio_cell["array"], dtype=np.float32)
            sr = int(audio_cell.get("sampling_rate", target_sr))
            return arr, sr
        if audio_cell.get("bytes"):
            wav, sr = sf.read(io.BytesIO(audio_cell["bytes"]), dtype="float32", always_2d=True)
            return wav.mean(axis=1), int(sr)
        if audio_cell.get("path") and Path(audio_cell["path"]).exists():
            wav, sr = sf.read(str(audio_cell["path"]), dtype="float32", always_2d=True)
            return wav.mean(axis=1), int(sr)
    raise ValueError(f"Unsupported audio cell format: {type(audio_cell)}")


def prepare_vctk_dataset(
    config_path: str | Path = "configs/data/vctk.yaml",
    paths_cfg_path: str | Path = "configs/paths.yaml",
    smoke_test: bool = False,
    smoke_speakers_per_accent: int = 1,
    smoke_utterances_per_speaker: int = 3,
) -> Path:
    """Prepare VCTK evaluation manifest and 3s/5s/8s reference clips (clean + noisy).

    When `smoke_test=True`, streams rows until it collects `smoke_speakers_per_accent`
    American and English speakers (e.g., 1 US + 1 UK x 3 utterances = 6 samples)
    so the notebook runs end-to-end in under 2 minutes on Colab.
    """
    from datasets import load_dataset

    cfg = load_yaml(config_path)
    paths = ProjectPaths(paths_cfg_path)
    paths.ensure_dirs()

    sel = cfg["selection"]
    target_accents = set(sel.get("target_accents", ["American", "English"]))
    spk_per_accent = (
        smoke_speakers_per_accent if smoke_test else int(sel.get("speakers_per_accent", 12))
    )
    utts_per_spk = (
        smoke_utterances_per_speaker
        if smoke_test
        else int(sel.get("utterances_per_speaker", 5))
    )
    min_words = int(sel.get("min_source_words", 8))
    preferred_mic = cfg.get("preferred_mic", "mic1")
    target_sr = int(cfg.get("sample_rate", 48000))
    ref_cfg = cfg.get("reference_slicing", {})
    ref_durations = ref_cfg.get("durations_sec", [3.0, 5.0, 8.0])
    snr_db = (
        float(ref_cfg["noise_augmentation"]["snr_db"])
        if ref_cfg.get("noise_augmentation", {}).get("enabled", True)
        else None
    )

    repo_id = cfg.get("hf_repo_id", "sanchit-gandhi/vctk")
    split = cfg.get("hf_split", "train")

    print(f"[prepare_vctk] Streaming `{repo_id}` (smoke_test={smoke_test})...")
    ds = load_dataset(repo_id, split=split, streaming=True)

    # Group candidate utterances per speaker (need eval utts + extra utts for reference slicing)
    needed_utts_per_spk = utts_per_spk + 4
    speaker_meta: Dict[str, Dict[str, str]] = {}
    speaker_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    selected_speakers_by_accent: Dict[str, List[str]] = defaultdict(list)

    for row in ds:
        accent = str(row.get("accent", "")).strip()
        if accent not in target_accents:
            continue
        file_field = str(row.get("file", ""))
        if preferred_mic and preferred_mic not in file_field:
            continue
        text = str(row.get("text", "")).strip()
        if len(text.split()) < min_words:
            continue

        spk_id = str(row.get("speaker_id", "")).strip()
        if not spk_id:
            continue

        # Check if we can accept this speaker
        if spk_id not in speaker_meta:
            if len(selected_speakers_by_accent[accent]) >= spk_per_accent:
                # Check if all already-selected speakers have enough utterances
                all_done = all(
                    len(selected_speakers_by_accent[a]) >= spk_per_accent
                    and all(
                        len(speaker_rows[s]) >= needed_utts_per_spk
                        for s in selected_speakers_by_accent[a]
                    )
                    for a in target_accents
                )
                if all_done:
                    break
                continue
            speaker_meta[spk_id] = {
                "speaker_id": spk_id,
                "accent": accent,
                "gender": str(row.get("gender", "U")).strip(),
                "age": str(row.get("age", "")).strip(),
                "region": str(row.get("region", "")).strip(),
            }
            selected_speakers_by_accent[accent].append(spk_id)

        if len(speaker_rows[spk_id]) < needed_utts_per_spk:
            speaker_rows[spk_id].append(row)

    raw_vctk_dir = paths.get_dir("raw") / "vctk"
    refs_dir = paths.get_dir("refs") / "vctk"
    manifest_records: List[Dict[str, Any]] = []

    for accent in sorted(target_accents):
        for spk_id in selected_speakers_by_accent[accent]:
            rows = speaker_rows[spk_id]
            if len(rows) < utts_per_spk + 1:
                print(f"[prepare_vctk] Warning: speaker {spk_id} has only {len(rows)} rows; skipping")
                continue

            eval_rows = rows[:utts_per_spk]
            ref_rows = rows[utts_per_spk:]

            # Save reference source WAVs first
            ref_source_paths: List[Path] = []
            for r_row in ref_rows:
                tid = str(r_row.get("text_id", "ref"))
                wav, sr = _decode_hf_audio_cell(r_row["audio"], target_sr=target_sr)
                r_path = raw_vctk_dir / spk_id / f"{spk_id}_{tid}_refsrc.wav"
                write_audio_atomic(r_path, wav, sr)
                ref_source_paths.append(r_path)

            ref_bundle = build_speaker_reference_clips(
                candidate_audio_paths=ref_source_paths,
                output_dir=refs_dir,
                speaker_id=spk_id,
                durations_sec=ref_durations,
                target_sr=target_sr,
                make_noisy_snr_db=snr_db,
            )
            # Also designate the first unused utterance as a separate held-out utterance for SIM-heldout
            heldout_audio_path = str(ref_source_paths[0])

            for e_row in eval_rows:
                tid = str(e_row.get("text_id", "000"))
                utt_id = f"vctk_{spk_id}_{tid}"
                wav, sr = _decode_hf_audio_cell(e_row["audio"], target_sr=target_sr)
                src_wav_path = raw_vctk_dir / spk_id / f"{utt_id}.wav"
                write_audio_atomic(src_wav_path, wav, sr)

                self_ref_mode = bool(ref_cfg.get("self_reference_mode", True))
                if self_ref_mode:
                    # Self-reference architecture: build 3s/5s/8s/10s reference clips
                    # starting directly from `src_wav_path` itself (padded with same-speaker
                    # audio if a single VCTK utterance is shorter than 8s-10s).
                    utt_ref_bundle = build_speaker_reference_clips(
                        candidate_audio_paths=[src_wav_path, *ref_source_paths],
                        output_dir=refs_dir / "self_ref",
                        speaker_id=utt_id,
                        durations_sec=ref_durations,
                        target_sr=target_sr,
                        make_noisy_snr_db=snr_db,
                    )
                else:
                    utt_ref_bundle = ref_bundle

                default_ref = (
                    utt_ref_bundle.get("10s")
                    or utt_ref_bundle.get("8s")
                    or next(iter(utt_ref_bundle.values()))
                )
                manifest_records.append(
                    {
                        "utt_id": utt_id,
                        "dataset": "vctk",
                        "speaker_id": spk_id,
                        "accent": speaker_meta[spk_id]["accent"],
                        "gender": speaker_meta[spk_id]["gender"],
                        "region": speaker_meta[spk_id]["region"],
                        "source_audio_path": str(src_wav_path),
                        "source_sample_rate": sr,
                        "source_duration_sec": round(len(wav) / sr, 3),
                        "gold_en_text": str(e_row.get("text", "")).strip(),
                        "text_id": tid,
                        "self_reference_mode": self_ref_mode,
                        "ref_audio_path": default_ref["clean_ref_path"],
                        "ref_duration_tag": default_ref["duration_tag"],
                        "heldout_same_speaker_audio_path": heldout_audio_path,
                        "refs_by_duration": utt_ref_bundle,
                    }
                )

    manifest_name = "vctk_smoke.jsonl" if smoke_test else "vctk_main.jsonl"
    out_manifest = paths.get_dir("manifests") / manifest_name
    write_manifest_atomic(manifest_records, out_manifest)
    print(f"[prepare_vctk] Saved {len(manifest_records)} records -> {out_manifest}")
    return out_manifest
