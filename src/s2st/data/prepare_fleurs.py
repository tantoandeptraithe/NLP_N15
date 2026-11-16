"""Prepare the 100-sentence MT Benchmark (Set A: 60 FLEURS/FLORES+ + Set B: 40 VCTK)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from s2st.io import ProjectPaths, load_yaml
from s2st.manifest import read_manifest, write_manifest_atomic

_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b|\b\d+(?:[.,]\d+)?\b")


def has_proper_noun_or_number(text: str) -> bool:
    """Heuristic check if an English sentence contains non-initial proper nouns or numbers."""
    tokens = text.strip().split()
    if len(tokens) <= 1:
        return False
    tail = " ".join(tokens[1:])
    return bool(_PROPER_NOUN_RE.search(tail))


def prepare_fleurs_flores_mt_benchmark(
    config_path: str | Path = "configs/data/fleurs.yaml",
    paths_cfg_path: str | Path = "configs/paths.yaml",
    vctk_manifest_path: str | Path | None = None,
    limit_flores: int | None = None,
) -> Path:
    """Build the MT benchmark manifest combining FLORES+ (with gold VI) and in-domain VCTK."""
    from huggingface_hub import hf_hub_download

    cfg = load_yaml(config_path)
    paths = ProjectPaths(paths_cfg_path)
    paths.ensure_dirs()

    n_flores = limit_flores or int(cfg.get("selection", {}).get("num_mt_benchmark_sentences", 60))
    flores_repo = cfg.get("flores_repo_id", "openlanguagedata/flores_plus")

    records: List[Dict[str, Any]] = []
    try:
        en_path = hf_hub_download(
            repo_id=flores_repo,
            filename=cfg.get("flores_en_file", "devtest/eng_Latn.jsonl"),
            repo_type="dataset",
        )
        vi_path = hf_hub_download(
            repo_id=flores_repo,
            filename=cfg.get("flores_vi_file", "devtest/vie_Latn.jsonl"),
            repo_type="dataset",
        )
        en_rows = [json.loads(line) for line in Path(en_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        vi_rows = [json.loads(line) for line in Path(vi_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        vi_by_id = {r["id"]: r["text"] for r in vi_rows}

        for r in en_rows[:n_flores]:
            sid = r["id"]
            en_txt = r["text"].strip()
            vi_txt = vi_by_id.get(sid, "").strip()
            records.append(
                {
                    "utt_id": f"flores_devtest_{sid:04d}",
                    "subset": "A_flores_plus",
                    "gold_en_text": en_txt,
                    "ref_vi_text": vi_txt,
                    "has_proper_noun_or_number": has_proper_noun_or_number(en_txt),
                }
            )
    except Exception as exc:
        print(
            f"[prepare_fleurs] Note: Could not download gated `{flores_repo}` ({exc}). "
            "Ensure HF_TOKEN is set and you accepted the license on Hugging Face."
        )

    # Append Subset B (40 unique VCTK sentences) if VCTK manifest is provided
    if vctk_manifest_path and Path(vctk_manifest_path).exists():
        vctk_recs = read_manifest(vctk_manifest_path)
        seen_texts = set()
        b_count = 0
        for rec in vctk_recs:
            txt = rec.get("gold_en_text", "").strip()
            if not txt or txt in seen_texts:
                continue
            seen_texts.add(txt)
            records.append(
                {
                    "utt_id": f"mt_vctk_{rec['utt_id']}",
                    "subset": "B_vctk_indomain",
                    "gold_en_text": txt,
                    "stt_en_text": rec.get("stt_en_text", ""),
                    "ref_vi_text": rec.get("ref_vi_text", ""),  # Filled by human translator per guideline
                    "has_proper_noun_or_number": has_proper_noun_or_number(txt),
                }
            )
            b_count += 1
            if b_count >= 40:
                break

    out_path = paths.get_dir("manifests") / "mt_benchmark_100.jsonl"
    write_manifest_atomic(records, out_path)
    print(f"[prepare_fleurs] Wrote {len(records)} MT benchmark records -> {out_path}")
    return out_path
