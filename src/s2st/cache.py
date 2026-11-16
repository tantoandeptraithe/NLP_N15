"""Deterministic configuration hashing and result caching utilities."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def compute_config_hash(config: Dict[str, Any], length: int = 10) -> str:
    """Compute a short deterministic SHA-256 hex digest of a nested dictionary."""
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=True, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:length]


def make_item_cache_key(utt_id: str, stage: str, config: Dict[str, Any]) -> str:
    """Create a composite key `<utt_id>__<stage>__<config_hash>` for idempotent runs."""
    cfg_hash = compute_config_hash(config, length=8)
    return f"{utt_id}__{stage}__{cfg_hash}"
