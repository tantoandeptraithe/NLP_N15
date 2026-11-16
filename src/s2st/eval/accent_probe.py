"""Exploratory Accent Embedding Probe using `Jzuluaga/accent-id-commonaccent_ecapa`.

Caveat (documented in §3):
- CommonAccent ECAPA is trained on English accents, so Vietnamese audio is out-of-domain.
- We use it strictly as a secondary exploratory probe by measuring cosine similarity between
  the synthesized Vietnamese audio embedding and the English source speaker's accent embedding,
  normalized against the native Vietnamese preset anchor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from s2st.eval.spk_sim import cosine_sim
from s2st.io import read_audio_mono


class AccentEmbeddingProbe:
    """Extracts embeddings from `Jzuluaga/accent-id-commonaccent_ecapa`."""

    def __init__(
        self,
        model_id: str = "Jzuluaga/accent-id-commonaccent_ecapa",
        device: str = "cuda",
    ) -> None:
        self.model_id = model_id
        self.device_name = device
        self._classifier: Any = None
        self._cache: Dict[str, np.ndarray] = {}

    def load(self) -> None:
        if self._classifier is not None:
            return
        import torch

        try:
            from speechbrain.inference.classifiers import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier

        run_opts = (
            {"device": "cuda"}
            if (self.device_name == "cuda" and torch.cuda.is_available())
            else {"device": "cpu"}
        )
        self._classifier = EncoderClassifier.from_hparams(source=self.model_id, run_opts=run_opts)
        print(f"[AccentProbe] Loaded `{self.model_id}`")

    def get_embedding(self, audio_path: str | Path) -> np.ndarray:
        key = str(audio_path)
        if key in self._cache:
            return self._cache[key]
        import torch

        self.load()
        wav, _ = read_audio_mono(audio_path, target_sr=16000)
        tensor = torch.from_numpy(wav).unsqueeze(0)
        with torch.inference_mode():
            emb = self._classifier.encode_batch(tensor).squeeze().detach().cpu().numpy()
        emb = emb.astype(np.float32)
        self._cache[key] = emb
        return emb

    def score_pair(
        self, tts_audio_path: str | Path, source_en_audio_path: str | Path
    ) -> float | None:
        if not Path(tts_audio_path).exists() or not Path(source_en_audio_path).exists():
            return None
        try:
            e_tts = self.get_embedding(tts_audio_path)
            e_src = self.get_embedding(source_en_audio_path)
            return round(cosine_sim(e_tts, e_src), 4)
        except Exception as exc:
            print(f"[AccentProbe] Failed on {tts_audio_path}: {exc}")
            return None
