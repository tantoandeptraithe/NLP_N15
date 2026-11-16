"""Open-weight LLM translator (`Qwen/Qwen3.5-9B`, `tencent/Hunyuan-MT-7B`, `google/gemma-4-12B-it`).

Key verified detail for `Qwen/Qwen3.5-9B`:
- Qwen3.5 operates in thinking mode (`<think>...</think>`) by default and does NOT use `/nothink`.
- To disable thinking, pass `enable_thinking=False` to `tokenizer.apply_chat_template(...)`,
  and also strip any residual `<think>...</think>` blocks via `clean_translation_output()`.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from s2st.mt.base import BaseTranslator, clean_translation_output


class LLMChatTranslator(BaseTranslator):
    """Translates English sentences to Vietnamese using instruction/chat LLMs."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.device_name = config.get("device", "cuda")
        self.load_in_4bit = bool(config.get("load_in_4bit", False))
        self.prompt_cfg = config.get("prompt", {})
        self.gen_cfg = config.get("generation", {})
        self.chat_template_kwargs = config.get("chat_template_kwargs", {"enable_thinking": False})
        self._tokenizer: Any = None
        self._model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_str = self.config.get("torch_dtype", "bfloat16")
        dtype = getattr(torch, dtype_str, torch.bfloat16)

        model_kwargs: Dict[str, Any] = {
            "device_map": "auto" if self.device_name == "cuda" else None,
            "torch_dtype": dtype,
        }
        if self.load_in_4bit and self.device_name == "cuda":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        try:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        except Exception:
            # Qwen3.5 uses Qwen3_5ForConditionalGeneration under AutoModelForImageTextToText
            from transformers import AutoModelForImageTextToText

            self._model = AutoModelForImageTextToText.from_pretrained(self.model_id, **model_kwargs)

        self._model.eval()
        print(f"[MT:LLM] Loaded `{self.model_id}` (4bit={self.load_in_4bit})")

    def _build_prompt(self, source_text: str) -> str:
        sys_msg = str(self.prompt_cfg.get("system", "")).strip()
        user_tpl = str(
            self.prompt_cfg.get("user_template", "Translate into Vietnamese:\n{source_text}")
        )
        user_msg = user_tpl.format(source_text=source_text)
        messages: List[Dict[str, str]] = []
        if sys_msg:
            messages.append({"role": "system", "content": sys_msg})
        messages.append({"role": "user", "content": user_msg})

        try:
            return str(
                self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **self.chat_template_kwargs,
                )
            )
        except TypeError:
            return str(
                self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

    def translate_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        import torch

        self.load()
        t0 = time.perf_counter()
        prompts = [self._build_prompt(t) for t in texts]
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[1]
        do_sample = bool(self.gen_cfg.get("do_sample", False))
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(self.gen_cfg.get("max_new_tokens", 256)),
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if not do_sample:
            gen_kwargs["temperature"] = None
            gen_kwargs["top_p"] = None
            gen_kwargs["top_k"] = None

        with torch.inference_mode():
            out_ids = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = out_ids[:, input_len:]
        raw_decoded = self._tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        elapsed_per_item = (time.perf_counter() - t0) / max(len(texts), 1)

        results: List[Dict[str, Any]] = []
        for raw in raw_decoded:
            results.append(
                {
                    "mt_vi_text": clean_translation_output(raw),
                    "mt_raw_output": raw.strip(),
                    "mt_latency_sec": round(elapsed_per_item, 4),
                }
            )
        return results
