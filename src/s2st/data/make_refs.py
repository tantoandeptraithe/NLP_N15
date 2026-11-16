"""Build clean and noisy reference clips (3s / 5s / 8s) per speaker for zero-shot cloning.

Crucial design rules:
1. Reference clips are ALWAYS built from held-out utterances of the same speaker
   (never the source utterance being translated/evaluated).
2. Leading/trailing silence is trimmed via energy/VAD before concatenation so that
   a "5.0s reference" contains ~5.0s of actual speech rather than dead air.
3. Since VieNeu-TTS v3 Turbo automatically trims references to <= 8 seconds,
   standard durations are [3.0, 5.0, 8.0] seconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from s2st.data.add_noise import create_noisy_reference_file
from s2st.io import read_audio_mono, write_audio_atomic


def trim_silence_energy(
    wav: np.ndarray,
    sample_rate: int,
    top_db: float = 30.0,
    frame_ms: int = 25,
    hop_ms: int = 10,
) -> np.ndarray:
    """Trim leading and trailing silence using frame RMS threshold (`top_db` below peak)."""
    if len(wav) == 0:
        return wav
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    hop_len = max(1, int(sample_rate * hop_ms / 1000))
    if len(wav) < frame_len:
        return wav

    # Compute short-time energy
    num_frames = 1 + (len(wav) - frame_len) // hop_len
    energies = np.empty(num_frames, dtype=np.float32)
    for i in range(num_frames):
        seg = wav[i * hop_len : i * hop_len + frame_len]
        energies[i] = float(np.sqrt(np.mean(seg * seg) + 1e-12))

    peak_e = float(np.max(energies))
    if peak_e < 1e-7:
        return wav
    thresh = peak_e * (10.0 ** (-top_db / 20.0))
    active = np.where(energies >= thresh)[0]
    if len(active) == 0:
        return wav

    start_sample = max(0, int(active[0]) * hop_len)
    end_sample = min(len(wav), int(active[-1]) * hop_len + frame_len)
    return wav[start_sample:end_sample]


def build_speaker_reference_clips(
    candidate_audio_paths: Sequence[str | Path],
    output_dir: str | Path,
    speaker_id: str,
    durations_sec: Sequence[float] = (3.0, 5.0, 8.0),
    target_sr: int = 48000,
    silence_gap_sec: float = 0.15,
    make_noisy_snr_db: float | None = 10.0,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Build exact-duration clean (and optionally noisy) reference WAVs for one speaker.

    Returns a dictionary keyed by duration tag (e.g. '3s', '5s', '8s') with paths and metadata.
    """
    out_root = Path(output_dir)
    gap = np.zeros(int(target_sr * silence_gap_sec), dtype=np.float32)

    trimmed_chunks: List[np.ndarray] = []
    used_files: List[str] = []
    max_needed_samples = int(max(durations_sec) * target_sr)
    accumulated = 0

    for p in candidate_audio_paths:
        wav, _ = read_audio_mono(p, target_sr=target_sr)
        trimmed = trim_silence_energy(wav, target_sr, top_db=28.0)
        if len(trimmed) < int(0.4 * target_sr):
            continue
        trimmed_chunks.append(trimmed)
        used_files.append(str(p))
        accumulated += len(trimmed) + len(gap)
        if accumulated >= max_needed_samples + int(target_sr * 1.0):
            break

    if not trimmed_chunks:
        raise ValueError(f"No valid reference audio chunks found for speaker {speaker_id}")

    # Concatenate chunks with a short natural pause
    interleaved: List[np.ndarray] = []
    for idx, ch in enumerate(trimmed_chunks):
        if idx > 0:
            interleaved.append(gap)
        interleaved.append(ch)
    long_stream = np.concatenate(interleaved, axis=0)

    # Peak normalize to -1 dBFS (0.891) for consistent conditioning
    peak = float(np.max(np.abs(long_stream)))
    if peak > 1e-6:
        long_stream = (long_stream * (0.891 / peak)).astype(np.float32)

    results: Dict[str, Dict[str, Any]] = {}
    for dur in durations_sec:
        tag = f"{int(dur) if float(dur).is_integer() else dur}s"
        n_target = int(round(dur * target_sr))
        clip = long_stream[:n_target]
        actual_dur = len(clip) / target_sr

        clean_path = out_root / "clean" / speaker_id / f"{tag}.wav"
        write_audio_atomic(clean_path, clip, target_sr)

        entry: Dict[str, Any] = {
            "duration_tag": tag,
            "requested_duration_sec": float(dur),
            "actual_duration_sec": round(actual_dur, 3),
            "clean_ref_path": str(clean_path),
            "num_concatenated_utterances": len(used_files),
            "source_ref_files": used_files,
        }

        if make_noisy_snr_db is not None:
            noisy_path = out_root / f"noisy_snr{int(make_noisy_snr_db)}db" / speaker_id / f"{tag}.wav"
            _, achieved_snr = create_noisy_reference_file(
                clean_path,
                noisy_path,
                snr_db=float(make_noisy_snr_db),
                seed=seed + int(dur * 10),
            )
            entry["noisy_ref_path"] = str(noisy_path)
            entry["noisy_snr_db"] = round(achieved_snr, 2)

        results[tag] = entry

    return results
