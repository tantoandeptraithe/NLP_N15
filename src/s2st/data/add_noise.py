"""Add controlled noise at a target SNR (dB) to reference audio for Experiment E3."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from s2st.io import read_audio_mono, write_audio_atomic


def compute_rms(wav: np.ndarray) -> float:
    """Compute root-mean-square energy of a waveform."""
    return float(np.sqrt(np.mean(np.square(wav.astype(np.float64))) + 1e-12))


def generate_pink_babble_noise(num_samples: int, sample_rate: int, seed: int = 42) -> np.ndarray:
    """Generate deterministic pink/cafe-like stationary+modulated noise (1/f spectrum).

    Used when no external noise file is provided so E3 is 100% reproducible without
    requiring large external noise corpora.
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(num_samples)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(num_samples, d=1.0 / sample_rate)
    # 1 / sqrt(f) shaping inside speech band (80 Hz - 8000 Hz)
    scale = 1.0 / np.sqrt(np.maximum(freqs, 80.0))
    pink = np.fft.irfft(spectrum * scale, n=num_samples)
    # Slow amplitude modulation (~3 Hz syllabic envelope) to mimic room babble
    t = np.arange(num_samples, dtype=np.float64) / sample_rate
    env = 1.0 + 0.35 * np.sin(2.0 * np.pi * 3.1 * t + rng.uniform(0, 2 * np.pi))
    out = (pink * env).astype(np.float32)
    peak = np.max(np.abs(out)) + 1e-9
    return out / peak


def mix_at_snr(
    clean_wav: np.ndarray,
    noise_wav: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """Mix `clean_wav` and `noise_wav` at exact `snr_db` decibels."""
    n = len(clean_wav)
    if len(noise_wav) < n:
        reps = int(np.ceil(n / max(len(noise_wav), 1)))
        noise_wav = np.tile(noise_wav, reps)[:n]
    else:
        noise_wav = noise_wav[:n]

    clean_rms = compute_rms(clean_wav)
    noise_rms = compute_rms(noise_wav)
    if clean_rms < 1e-8 or noise_rms < 1e-8:
        return clean_wav.copy()

    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    scaled_noise = noise_wav * (target_noise_rms / noise_rms)
    mixed = clean_wav + scaled_noise
    # Prevent clipping while preserving relative SNR
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)


def create_noisy_reference_file(
    clean_ref_path: str | Path,
    noisy_ref_path: str | Path,
    snr_db: float = 10.0,
    external_noise_path: str | Path | None = None,
    seed: int = 42,
) -> Tuple[Path, float]:
    """Create a noisy reference WAV at `snr_db` and return `(output_path, achieved_snr_db)`."""
    clean_wav, sr = read_audio_mono(clean_ref_path)
    if external_noise_path and Path(external_noise_path).exists():
        noise_wav, _ = read_audio_mono(external_noise_path, target_sr=sr)
    else:
        noise_wav = generate_pink_babble_noise(len(clean_wav), sr, seed=seed)

    mixed = mix_at_snr(clean_wav, noise_wav, snr_db=snr_db)
    out_path = write_audio_atomic(noisy_ref_path, mixed, sr)
    noise_component = mixed - clean_wav * (
        compute_rms(mixed) / max(compute_rms(clean_wav), 1e-9)
        if np.max(np.abs(clean_wav + noise_wav)) > 0.99
        else 1.0
    )
    achieved_snr = 20.0 * np.log10(
        max(compute_rms(clean_wav), 1e-9) / max(compute_rms(mixed - clean_wav), 1e-9)
    )
    return out_path, float(achieved_snr)
