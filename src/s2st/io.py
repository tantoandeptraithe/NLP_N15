"""I/O utilities: YAML config loading, Google Drive path resolution, and audio read/write."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import soundfile as sf
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a UTF-8 YAML configuration file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(data: Dict[str, Any], path: str | Path) -> None:
    """Atomically write a dictionary to a UTF-8 YAML file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_suffix(p.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    tmp_path.replace(p)


class ProjectPaths:
    """Resolves persistent Google Drive directories and local SSD caches."""

    def __init__(self, paths_cfg_path: str | Path = "configs/paths.yaml") -> None:
        cfg = load_yaml(paths_cfg_path) if Path(paths_cfg_path).exists() else {}
        drive_env = os.environ.get("S2ST_DRIVE_ROOT")
        cache_env = os.environ.get("S2ST_CACHE_ROOT")

        default_drive = cfg.get("drive_root", "/content/drive/MyDrive/s2st_vi")
        # If not running on Colab with mounted Drive, fall back to ./workspace_drive
        if not drive_env and not Path("/content/drive").exists():
            default_drive = str(Path("./workspace_drive").resolve())

        self.drive_root = Path(drive_env or default_drive)
        self.cache_root = Path(cache_env or cfg.get("local_cache_root", "./.cache/s2st"))
        self.subdirs: Dict[str, str] = cfg.get(
            "subdirs",
            {
                "raw": "raw",
                "manifests": "processed/manifests",
                "refs": "refs",
                "outputs_stt": "outputs/stt",
                "outputs_mt": "outputs/mt",
                "outputs_tts": "outputs/tts",
                "metrics": "metrics",
                "listening": "listening",
                "runs": "runs",
            },
        )

    def ensure_dirs(self) -> None:
        """Create all standard directories on Drive and local cache."""
        self.drive_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        for rel in self.subdirs.values():
            (self.drive_root / rel).mkdir(parents=True, exist_ok=True)

    def get_dir(self, key: str) -> Path:
        rel = self.subdirs.get(key, key)
        p = self.drive_root / rel
        p.mkdir(parents=True, exist_ok=True)
        return p


def read_audio_mono(path: str | Path, target_sr: int | None = None) -> Tuple[np.ndarray, int]:
    """Read an audio file as float32 mono waveform, optionally resampling."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav_mono = wav.mean(axis=1)
    if target_sr is not None and sr != target_sr:
        try:
            import soxr

            wav_mono = soxr.resample(wav_mono, sr, target_sr)
        except ImportError:
            import librosa

            wav_mono = librosa.resample(wav_mono, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return wav_mono.astype(np.float32), int(sr)


def write_audio_atomic(
    path: str | Path, wav: np.ndarray, sample_rate: int, subtype: str = "PCM_16"
) -> Path:
    """Atomically write a mono or stereo waveform to WAV/FLAC to survive Colab disconnects."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_name(f".tmp_{p.name}")
    sf.write(str(tmp_path), np.asarray(wav, dtype=np.float32), int(sample_rate), subtype=subtype)
    tmp_path.replace(p)
    return p
