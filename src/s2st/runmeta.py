"""Capture reproducibility metadata (git commit, pip freeze, GPU info, resolved config)."""

from __future__ import annotations

import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def _run_cmd(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=15)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return "unavailable"


def collect_run_metadata(
    run_id: str, stage: str, resolved_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Gather environment, git, and hardware info for reproducibility."""
    gpu_info = _run_cmd(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
    )
    git_commit = _run_cmd(["git", "rev-parse", "HEAD"])
    git_dirty = bool(_run_cmd(["git", "status", "--porcelain"]))
    pip_freeze = _run_cmd([sys.executable, "-m", "pip", "freeze"]).splitlines()

    return {
        "run_id": run_id,
        "stage": stage,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "gpu_info": gpu_info,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "pip_freeze": pip_freeze,
        "resolved_config": resolved_config,
    }


def save_run_metadata(
    runs_dir: str | Path, run_id: str, stage: str, resolved_config: Dict[str, Any]
) -> Path:
    """Save `meta_<stage>.json` inside `runs/<run_id>/`."""
    out_dir = Path(runs_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = collect_run_metadata(run_id, stage, resolved_config)
    out_path = out_dir / f"meta_{stage}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return out_path
