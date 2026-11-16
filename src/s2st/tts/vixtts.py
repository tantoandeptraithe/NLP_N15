"""viXTTS (`capleaf/viXTTS`) Baseline Zero-Shot Cross-Lingual TTS Wrapper.

Verified details from `thinhlpg/vixtts-demo` and `idiap/coqui-ai-TTS`:
1. Model files required from `capleaf/viXTTS`: `model.pth`, `config.json`, `vocab.json`,
   plus `speakers_xtts.pth` from `coqui/XTTS-v2`.
2. In standard `coqui-tts==0.28.0`, `VoiceBpeTokenizer.preprocess_text` raises
   `NotImplementedError("Language 'vi' is not supported")` because `thinhlpg/TTS`
   added `elif lang == "vi": txt = basic_cleaners(txt)` on a custom branch.
   We patch `VoiceBpeTokenizer.preprocess_text` and `char_limits["vi"] = 250` at runtime
   so `capleaf/viXTTS` runs cleanly on standard `coqui-tts==0.28.0` without needing a
   legacy git submodule build!
3. Short Vietnamese sentences (<10 words) suffer trailing artifacts on viXTTS; we port
   `calculate_keep_len()` and `normalize_vietnamese_text()` directly from `vixtts_demo.py`.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from s2st.io import write_audio_atomic
from s2st.tts.base import BaseZeroShotTTS


def calculate_keep_len(text: str, lang: str = "vi") -> int:
    """Ported from `thinhlpg/vixtts-demo`: trim trailing noise on short Vietnamese clauses."""
    if lang in ("ja", "zh-cn"):
        return -1
    word_count = len(text.split())
    num_punct = text.count(".") + text.count("!") + text.count("?") + text.count(",")
    if word_count < 5:
        return 15000 * word_count + 2000 * num_punct
    if word_count < 10:
        return 13000 * word_count + 2000 * num_punct
    return -1


def normalize_vietnamese_text(text: str) -> str:
    """Normalize Vietnamese numbers/abbreviations using `vinorm` if installed."""
    try:
        from vinorm import TTSnorm

        text = TTSnorm(text, unknown=False, lower=False, rule=True)
    except Exception:
        pass
    for old, new in (
        ("..", "."),
        ("!.", "!"),
        ("?.", "?"),
        (" .", "."),
        (" ,", ","),
        ('"', ""),
        ("'", ""),
        ("AI", "ây ai"),
        ("A.I", "ây ai"),
    ):
        text = text.replace(old, new)
    return text.strip()


def _patch_coqui_xtts_tokenizer_for_vietnamese() -> None:
    """Monkey-patch `VoiceBpeTokenizer` in `coqui-tts` to support `lang='vi'`."""
    from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer, basic_cleaners

    orig_init = VoiceBpeTokenizer.__init__
    orig_preprocess = VoiceBpeTokenizer.preprocess_text

    if getattr(VoiceBpeTokenizer, "_s2st_vi_patched", False):
        return

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        if hasattr(self, "char_limits") and isinstance(self.char_limits, dict):
            self.char_limits.setdefault("vi", 250)

    def patched_preprocess(self: Any, txt: str, lang: str) -> str:
        if lang.split("-")[0] == "vi":
            return str(basic_cleaners(txt))
        return str(orig_preprocess(self, txt, lang))

    VoiceBpeTokenizer.__init__ = patched_init  # type: ignore[method-assign]
    VoiceBpeTokenizer.preprocess_text = patched_preprocess  # type: ignore[method-assign]
    VoiceBpeTokenizer._s2st_vi_patched = True  # type: ignore[attr-defined]


class ViXTTSEngine(BaseZeroShotTTS):
    """Wrapper for `capleaf/viXTTS` (24 kHz) with conditioning latent cache."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.model_id = config.get("model_id", "capleaf/viXTTS")
        self.base_speaker_repo_id = config.get("base_speaker_repo_id", "coqui/XTTS-v2")
        self.inf_cfg = config.get("inference", {})
        self.do_norm_text = bool(config.get("normalize_vietnamese_text", True))
        self.do_keep_len = bool(config.get("apply_short_sentence_keep_len", True))
        self._model: Any = None
        self._cond_cache: Dict[str, Tuple[Any, Any]] = {}

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from huggingface_hub import hf_hub_download, snapshot_download
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        _patch_coqui_xtts_tokenizer_for_vietnamese()

        ckpt_dir = Path(
            snapshot_download(
                repo_id=self.model_id,
                allow_patterns=["model.pth", "config.json", "vocab.json"],
            )
        )
        spk_file = Path(ckpt_dir) / "speakers_xtts.pth"
        if not spk_file.exists():
            downloaded_spk = hf_hub_download(
                repo_id=self.base_speaker_repo_id,
                filename="speakers_xtts.pth",
                local_dir=str(ckpt_dir),
            )
            spk_file = Path(downloaded_spk)

        xtts_cfg = XttsConfig()
        xtts_cfg.load_json(str(ckpt_dir / "config.json"))
        if hasattr(xtts_cfg, "languages") and "vi" not in xtts_cfg.languages:
            xtts_cfg.languages.append("vi")

        model = Xtts.init_from_config(xtts_cfg)
        model.load_checkpoint(
            xtts_cfg,
            checkpoint_dir=str(ckpt_dir),
            vocab_path=str(ckpt_dir / "vocab.json"),
            speaker_file_path=str(spk_file),
            use_deepspeed=False,
        )
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        self._model = model
        print(f"[TTS:viXTTS] Loaded `{self.model_id}` @ {self.sample_rate} Hz")

    def _get_latents(self, ref_audio_path: str) -> Tuple[Any, Any]:
        if ref_audio_path not in self._cond_cache:
            gpt_cond_latent, speaker_embedding = self._model.get_conditioning_latents(
                audio_path=ref_audio_path,
                gpt_cond_len=self._model.config.gpt_cond_len,
                max_ref_length=self._model.config.max_ref_len,
                sound_norm_refs=self._model.config.sound_norm_refs,
            )
            self._cond_cache[ref_audio_path] = (gpt_cond_latent, speaker_embedding)
        return self._cond_cache[ref_audio_path]

    def synthesize_to_file(
        self,
        text: str,
        output_wav_path: str | Path,
        ref_audio_path: str | Path | None = None,
        preset_voice: str | None = None,
        denoise_ref: bool = True,
        language: str = "vi",
    ) -> Dict[str, Any]:
        import torch

        self.load()
        if not ref_audio_path or not os.path.exists(ref_audio_path):
            raise ValueError("viXTTS requires a valid `ref_audio_path` for zero-shot synthesis.")

        t0 = time.perf_counter()
        gpt_cond_latent, speaker_embedding = self._get_latents(str(ref_audio_path))

        proc_text = normalize_vietnamese_text(text) if (self.do_norm_text and language == "vi") else text
        try:
            from underthesea import sent_tokenize

            sentences = sent_tokenize(proc_text) if language == "vi" else [proc_text]
        except Exception:
            sentences = [proc_text]

        wav_chunks: List[np.ndarray] = []
        with torch.inference_mode():
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                out = self._model.inference(
                    text=sent,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=float(self.inf_cfg.get("temperature", 0.3)),
                    length_penalty=float(self.inf_cfg.get("length_penalty", 1.0)),
                    repetition_penalty=float(self.inf_cfg.get("repetition_penalty", 10.0)),
                    top_k=int(self.inf_cfg.get("top_k", 30)),
                    top_p=float(self.inf_cfg.get("top_p", 0.85)),
                    enable_text_splitting=bool(self.inf_cfg.get("enable_text_splitting", True)),
                )
                chunk = np.asarray(out["wav"], dtype=np.float32)
                if self.do_keep_len and language == "vi":
                    keep = calculate_keep_len(sent, language)
                    if keep > 0:
                        chunk = chunk[:keep]
                wav_chunks.append(chunk)

        full_wav = np.concatenate(wav_chunks, axis=0) if wav_chunks else np.zeros(2400, dtype=np.float32)
        elapsed = time.perf_counter() - t0
        out_path = write_audio_atomic(output_wav_path, full_wav, self.sample_rate)
        dur = len(full_wav) / float(self.sample_rate)
        rtf = elapsed / max(dur, 1e-6)

        return {
            "tts_audio_path": str(out_path),
            "tts_sample_rate": self.sample_rate,
            "tts_duration_sec": round(dur, 3),
            "tts_latency_sec": round(elapsed, 4),
            "tts_rtf": round(rtf, 4),
            "tts_preset_voice": None,
        }
