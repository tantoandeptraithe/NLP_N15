"""JSONL manifest read/write with atomic updates and idempotent resume support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


def read_manifest(path: str | Path) -> List[Dict[str, Any]]:
    """Read a JSONL manifest file, skipping blank or corrupted trailing lines."""
    p = Path(path)
    if not p.exists():
        return []
    records: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError:
                # If a Colab disconnect truncated the last line, skip it safely
                print(f"[manifest] Warning: skipping truncated JSON at {p}:{line_no}")
    return records


def write_manifest_atomic(records: Iterable[Dict[str, Any]], path: str | Path) -> Path:
    """Atomically write an entire list of records to a JSONL file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_suffix(p.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
    tmp_path.replace(p)
    return p


def append_manifest_record(record: Dict[str, Any], path: str | Path) -> None:
    """Append a single record to a JSONL manifest and flush immediately to Drive."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def get_completed_keys(
    manifest_path: str | Path,
    key_field: str = "utt_id",
    required_non_empty_fields: List[str] | None = None,
    verify_file_field: str | None = None,
) -> Set[str]:
    """Return the set of `key_field` values already completed in `manifest_path`.

    If `verify_file_field` is specified (e.g. 'tts_audio_path'), also verifies
    that the referenced file exists on disk and has size > 0 bytes.
    """
    records = read_manifest(manifest_path)
    completed: Set[str] = set()
    for rec in records:
        key = rec.get(key_field)
        if not key:
            continue
        if required_non_empty_fields:
            if any(rec.get(field) in (None, "") for field in required_non_empty_fields):
                continue
        if verify_file_field:
            fpath = rec.get(verify_file_field)
            if not fpath or not Path(fpath).exists() or Path(fpath).stat().st_size == 0:
                continue
        completed.add(str(key))
    return completed


def deduplicate_manifest(
    path: str | Path, key_field: str = "utt_id"
) -> List[Dict[str, Any]]:
    """Deduplicate a JSONL manifest keeping the latest entry per `key_field`."""
    records = read_manifest(path)
    by_key: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        k = str(rec.get(key_field, ""))
        if k:
            by_key[k] = rec
    deduped = list(by_key.values())
    if Path(path).exists():
        write_manifest_atomic(deduped, path)
    return deduped
