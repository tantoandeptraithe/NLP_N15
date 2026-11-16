"""Base interface and runner for resumable Machine Translation stages."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from s2st.io import load_yaml
from s2st.manifest import append_manifest_record, deduplicate_manifest, get_completed_keys, read_manifest

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def clean_translation_output(raw_text: str) -> str:
    """Strip `<think>...</think>` blocks, markdown quotes, and multi-line chatter from LLM output."""
    cleaned = _THINK_BLOCK_RE.sub("", raw_text).strip()
    # If an unclosed </think> tag exists, keep only what follows it
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if not lines:
        return ""
    # Take the first non-empty content line if the model did not add prefix labels
    first = lines[0]
    for prefix in ("Bản dịch:", "Tiếng Việt:", "Vietnamese:", "Translation:"):
        if first.lower().startswith(prefix.lower()):
            first = first[len(prefix) :].strip()
    return first.strip("\"'“”")


class BaseTranslator(ABC):
    """Abstract base class for EN->VI translators."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model_name = config.get("model_name", "mt_model")
        self.model_id = config["model_id"]

    @abstractmethod
    def translate_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Translate a batch of English strings into Vietnamese dicts with text & latency."""


def build_translator(config: Dict[str, Any]) -> BaseTranslator:
    backend = config.get("backend", "seq2seq_nllb")
    if backend in ("seq2seq_vinai", "seq2seq_nllb"):
        from s2st.mt.seq2seq import Seq2SeqTranslator

        return Seq2SeqTranslator(config)
    if backend == "llm_chat":
        from s2st.mt.llm import LLMChatTranslator

        return LLMChatTranslator(config)
    raise ValueError(f"Unknown MT backend: {backend}")


def run_mt_stage(
    input_manifest_path: str | Path,
    output_manifest_path: str | Path,
    mt_config_path: str | Path,
    source_text_field: str = "stt_en_text",
) -> Path:
    """Run resumable batched EN->VI translation over `input_manifest_path`."""
    cfg = load_yaml(mt_config_path)
    records = read_manifest(input_manifest_path)
    completed = get_completed_keys(
        output_manifest_path,
        key_field="utt_id",
        required_non_empty_fields=["mt_vi_text"],
    )
    pending = [r for r in records if str(r.get("utt_id")) not in completed]
    print(
        f"[MT:{cfg.get('model_name')}] Total={len(records)} | Done={len(completed)} | Pending={len(pending)}"
    )
    if not pending:
        return Path(output_manifest_path)

    translator = build_translator(cfg)
    batch_size = int(cfg.get("batch_size", 8))

    for i in tqdm(range(0, len(pending), batch_size), desc=f"MT ({cfg.get('model_name')})"):
        batch_recs = pending[i : i + batch_size]
        src_texts = [
            str(r.get(source_text_field) or r.get("gold_en_text") or "").strip()
            for r in batch_recs
        ]
        outputs = translator.translate_batch(src_texts)
        for rec, out in zip(batch_recs, outputs):
            merged = {
                **rec,
                "mt_model_name": cfg.get("model_name"),
                "mt_model_id": cfg.get("model_id"),
                "mt_source_field": source_text_field,
                **out,
            }
            append_manifest_record(merged, output_manifest_path)

    deduplicate_manifest(output_manifest_path, key_field="utt_id")
    return Path(output_manifest_path)
