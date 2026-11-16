#!/usr/bin/env python3
"""Build randomized, balanced listening-test stimulus packages + consistency repeats.

Design for < 10 listeners (§6 of Technical Plan):
- 12 speakers (6 US, 6 UK; balanced gender) x 3 conditions (VieNeu clone, viXTTS clone, VieNeu native preset anchor) = 36 items
- + 6 items from E3 reference length/noise ablation
- + 3 duplicate items for intra-rater reliability check
= 45 stimuli per listener (~15-20 minutes).
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from s2st.manifest import read_manifest


def build_stimuli(
    manifests: List[Path],
    output_dir: Path,
    speakers_per_accent: int = 6,
    num_repeat_checks: int = 3,
    seed: int = 42,
) -> Path:
    rng = random.Random(seed)
    audio_out_dir = output_dir / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    by_utt_system: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for m_path in manifests:
        for rec in read_manifest(m_path):
            if rec.get("tts_audio_path") and Path(rec["tts_audio_path"]).exists():
                by_utt_system[str(rec["utt_id"])].append(rec)

    # Select balanced speakers (US & UK)
    speakers_by_accent: Dict[str, List[str]] = defaultdict(list)
    utt_for_speaker: Dict[str, str] = {}
    for utt_id, recs in sorted(by_utt_system.items()):
        r0 = recs[0]
        spk = str(r0.get("speaker_id"))
        acc = str(r0.get("accent", "Unknown"))
        if spk not in utt_for_speaker and len(speakers_by_accent[acc]) < speakers_per_accent:
            speakers_by_accent[acc].append(spk)
            utt_for_speaker[spk] = utt_id

    chosen_utts = set(utt_for_speaker.values())
    stimuli_rows: List[Dict[str, Any]] = []

    for utt_id in sorted(chosen_utts):
        for rec in by_utt_system[utt_id]:
            sys_name = f"{rec.get('tts_model_name')}__{rec.get('tts_condition_tag')}"
            stim_id = f"stim_{utt_id}__{sys_name}"
            src_wav = Path(rec["tts_audio_path"])
            ref_wav = Path(rec.get("ref_audio_path") or rec.get("source_audio_path"))

            dst_tts = audio_out_dir / f"{stim_id}_tts.wav"
            dst_ref = audio_out_dir / f"{utt_id}_ref_en.wav"
            shutil.copy2(src_wav, dst_tts)
            if ref_wav.exists() and not dst_ref.exists():
                shutil.copy2(ref_wav, dst_ref)

            stimuli_rows.append(
                {
                    "stimulus_id": stim_id,
                    "utt_id": utt_id,
                    "speaker_id": rec.get("speaker_id"),
                    "accent": rec.get("accent"),
                    "gender": rec.get("gender"),
                    "system": sys_name,
                    "vi_text": rec.get("tts_input_text") or rec.get("mt_vi_text", ""),
                    "en_text": rec.get("gold_en_text", ""),
                    "tts_audio_relpath": f"audio/{dst_tts.name}",
                    "ref_audio_relpath": f"audio/{dst_ref.name}",
                    "is_repeat_check": False,
                }
            )

    # Add intra-rater reliability repeat stimuli
    if len(stimuli_rows) >= num_repeat_checks:
        for rep in rng.sample(stimuli_rows, k=num_repeat_checks):
            dup = dict(rep)
            dup["stimulus_id"] = f"{rep['stimulus_id']}__rep"
            dup["is_repeat_check"] = True
            stimuli_rows.append(dup)

    rng.shuffle(stimuli_rows)
    csv_path = output_dir / "stimuli.csv"
    if stimuli_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(stimuli_rows[0].keys()))
            writer.writeheader()
            writer.writerows(stimuli_rows)
    print(f"[listening_test] Wrote {len(stimuli_rows)} stimuli -> {csv_path}")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build listening test stimuli.")
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output-dir", default="listening_test")
    args = parser.parse_args()
    build_stimuli([Path(m) for m in args.manifests], Path(args.output_dir))


if __name__ == "__main__":
    main()
