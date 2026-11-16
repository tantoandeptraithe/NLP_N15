"""Seq2Seq translators: `vinai/vinai-translate-en2vi-v2` (mBART) and `facebook/nllb-200-*` (M2M100)."""

from __future__ import annotations

import time
from typing import Any, Dict, List

from s2st.mt.base import BaseTranslator


class Seq2SeqTranslator(BaseTranslator):
    """Handles both VinAI Translate (mBART) and Meta NLLB-200 (M2M100) with verified lang codes."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.backend = config.get("backend", "seq2seq_nllb")
        self.src_lang = config.get("src_lang", "en_XX" if "vinai" in self.backend else "eng_Latn")
        self.tgt_lang = config.get("tgt_lang", "vi_VN" if "vinai" in self.backend else "vie_Latn")
        self.device_name = config.get("device", "cuda")
        self.gen_cfg = config.get("generation", {})
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        dtype_str = self.config.get("torch_dtype", "float16")
        use_cuda = self.device_name == "cuda" and torch.cuda.is_available()
        self._device = torch.device("cuda" if use_cuda else "cpu")
        dtype = getattr(torch, dtype_str, torch.float16) if use_cuda else torch.float32

        # Compatibility patch for older local PyTorch (<2.6) when loading legacy .bin checkpoints
        # such as `vinai/vinai-translate-en2vi-v2` under transformers>=4.50 (CVE-2025-32434 check).
        try:
            import transformers.modeling_utils as _mu
            import transformers.utils.import_utils as _iu

            _iu.check_torch_load_is_safe = lambda: None
            _mu.check_torch_load_is_safe = lambda: None
        except Exception:
            pass

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, src_lang=self.src_lang)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, torch_dtype=dtype)
        self._model.to(self._device)
        self._model.eval()
        print(f"[MT:Seq2Seq] Loaded `{self.model_id}` on {self._device} ({dtype})")

    def _resolve_target_lang_id(self) -> int:
        if hasattr(self._tokenizer, "lang_code_to_id") and self.tgt_lang in self._tokenizer.lang_code_to_id:
            return int(self._tokenizer.lang_code_to_id[self.tgt_lang])
        return int(self._tokenizer.convert_tokens_to_ids(self.tgt_lang))

    def translate_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        import torch

        self.load()
        t0 = time.perf_counter()
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self._device)

        tgt_lang_id = self._resolve_target_lang_id()
        gen_kwargs: Dict[str, Any] = {
            "num_beams": int(self.gen_cfg.get("num_beams", 5)),
            "max_new_tokens": int(self.gen_cfg.get("max_new_tokens", 256)),
            "early_stopping": bool(self.gen_cfg.get("early_stopping", True)),
        }
        if self.backend == "seq2seq_vinai":
            # Verified from VinAIResearch/VinAI_Translate README
            gen_kwargs["decoder_start_token_id"] = tgt_lang_id
        else:
            gen_kwargs["forced_bos_token_id"] = tgt_lang_id

        with torch.inference_mode():
            out_ids = self._model.generate(**inputs, **gen_kwargs)

        decoded = self._tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        elapsed_per_item = (time.perf_counter() - t0) / max(len(texts), 1)
        return [
            {
                "mt_vi_text": txt.strip(),
                "mt_latency_sec": round(elapsed_per_item, 4),
            }
            for txt in decoded
        ]
