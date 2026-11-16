"""Prosody extraction (speaking rate, pause ratio, F0 median/IQR/slope) using Praat-Parselmouth.

Used in Experiment E4 to test Hypothesis H3:
- Vietnamese is a 6-tone language where lexical tone heavily constrains F0 contour,
  whereas speaking rate and pause rhythm can still transfer from the English speaker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from s2st.io import read_audio_mono


def extract_prosody_features(
    audio_path: str | Path,
    text: str = "",
    f0_min_hz: float = 75.0,
    f0_max_hz: float = 500.0,
) -> Dict[str, float | None]:
    """Extract F0 statistics (in Hz and semitones), voiced/pause ratio, and speaking rate."""
    if not audio_path or not Path(audio_path).exists():
        return {}
    wav, sr = read_audio_mono(audio_path)
    duration_sec = len(wav) / float(sr) if sr > 0 else 0.0
    if duration_sec < 0.2:
        return {}

    word_or_syllable_count = len(text.strip().split()) if text else 0
    speaking_rate = (word_or_syllable_count / duration_sec) if word_or_syllable_count > 0 else None

    try:
        import parselmouth

        snd = parselmouth.Sound(wav.astype(np.float64), sampling_frequency=float(sr))
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=f0_min_hz, pitch_ceiling=f0_max_hz)
        f0 = pitch.selected_array["frequency"]
        voiced = f0[f0 > 0]
        voiced_fraction = float(len(voiced)) / max(len(f0), 1)
        pause_ratio = 1.0 - voiced_fraction

        if len(voiced) < 5:
            return {
                "speaking_rate_w_per_s": round(speaking_rate, 3) if speaking_rate else None,
                "pause_ratio": round(pause_ratio, 4),
                "f0_median_hz": None,
                "f0_iqr_semitones": None,
                "f0_slope_st_per_sec": None,
            }

        f0_med = float(np.median(voiced))
        # Convert voiced F0 to semitones relative to 100 Hz: 12 * log2(f / 100)
        st = 12.0 * np.log2(voiced / 100.0)
        q75, q25 = np.percentile(st, [75, 25])
        f0_iqr_st = float(q75 - q25)

        t_voiced = np.where(f0 > 0)[0] * 0.01
        slope = float(np.polyfit(t_voiced, st, 1)[0]) if len(t_voiced) >= 5 else 0.0

        return {
            "speaking_rate_w_per_s": round(speaking_rate, 3) if speaking_rate else None,
            "pause_ratio": round(pause_ratio, 4),
            "f0_median_hz": round(f0_med, 2),
            "f0_iqr_semitones": round(f0_iqr_st, 3),
            "f0_slope_st_per_sec": round(slope, 3),
        }
    except Exception as exc:
        print(f"[prosody] parselmouth failed on {audio_path}: {exc}")
        return {
            "speaking_rate_w_per_s": round(speaking_rate, 3) if speaking_rate else None,
            "pause_ratio": None,
            "f0_median_hz": None,
            "f0_iqr_semitones": None,
            "f0_slope_st_per_sec": None,
        }
