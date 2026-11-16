"""Cross-Lingual Speaker Similarity using ECAPA-TDNN and WavLM-Base-Plus-SV.

Methodological Safeguards (documented in §3 of the Technical Plan):
- Monolingual English speaker encoders suffer a systematic score drop when comparing
  English reference speech against synthesized Vietnamese speech (language-mismatch penalty).
- Therefore, for every synthesized file we compute:
    1. `sim_to_prompt_ref`: Cosine similarity to the exact reference clip fed to TTS.
    2. `sim_to_heldout_spk`: Cosine similarity to a *different* English utterance of the
       same speaker (prevents acoustic prompt overfitting bias).
- And we support both `speechbrain/spkrec-ecapa-voxceleb` and `microsoft/wavlm-base-plus-sv`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from s2st.io import read_audio_mono


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


class SpeakerSimilarityScorer:
    """Extracts speaker embeddings (ECAPA-TDNN & WavLM-SV) with audio path caching."""

    def __init__(
        self,
        ecapa_id: str = "speechbrain/spkrec-ecapa-voxceleb",
        wavlm_id: str = "microsoft/wavlm-base-plus-sv",
        device: str = "cuda",
    ) -> None:
        self.ecapa_id = ecapa_id
        self.wavlm_id = wavlm_id
        self.device_name = device
        self._ecapa: Any = None
        self._wavlm_fe: Any = None
        self._wavlm_model: Any = None
        self._ecapa_cache: Dict[str, np.ndarray] = {}
        self._wavlm_cache: Dict[str, np.ndarray] = {}

    def _load_ecapa(self) -> None:
        if self._ecapa is not None:
            return
        import torch

        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier

        run_opts = {"device": "cuda"} if (self.device_name == "cuda" and torch.cuda.is_available()) else {"device": "cpu"}
        self._ecapa = EncoderClassifier.from_hparams(source=self.ecapa_id, run_opts=run_opts)

    def _load_wavlm(self) -> None:
        if self._wavlm_model is not None:
            return
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

        use_cuda = self.device_name == "cuda" and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self._wavlm_fe = AutoFeatureExtractor.from_pretrained(self.wavlm_id)
        self._wavlm_model = AutoModelForAudioXVector.from_pretrained(self.wavlm_id).to(device)
        self._wavlm_model.eval()

    def get_ecapa_embedding(self, audio_path: str | Path) -> np.ndarray:
        key = str(audio_path)
        if key in self._ecapa_cache:
            return self._ecapa_cache[key]
        import torch

        self._load_ecapa()
        wav, _ = read_audio_mono(audio_path, target_sr=16000)
        tensor = torch.from_numpy(wav).unsqueeze(0)
        with torch.inference_mode():
            emb = self._ecapa.encode_batch(tensor).squeeze().detach().cpu().numpy()
        emb = emb.astype(np.float32)
        self._ecapa_cache[key] = emb
        return emb

    def get_wavlm_embedding(self, audio_path: str | Path) -> np.ndarray:
        key = str(audio_path)
        if key in self._wavlm_cache:
            return self._wavlm_cache[key]
        import torch

        self._load_wavlm()
        wav, _ = read_audio_mono(audio_path, target_sr=16000)
        inputs = self._wavlm_fe(wav, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._wavlm_model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            emb = self._wavlm_model(**inputs).embeddings.squeeze().detach().cpu().numpy()
        emb = emb.astype(np.float32)
        self._wavlm_cache[key] = emb
        return emb

    def score_record(self, rec: Dict[str, Any]) -> Dict[str, float | None]:
        tts_path = rec.get("tts_audio_path")
        prompt_ref = rec.get("tts_used_ref_audio_path") or rec.get("ref_audio_path")
        heldout_ref = rec.get("heldout_same_speaker_audio_path") or rec.get("source_audio_path")

        if not tts_path or not Path(tts_path).exists():
            return {}

        out: Dict[str, float | None] = {}
        try:
            tts_ecapa = self.get_ecapa_embedding(tts_path)
            if prompt_ref and Path(prompt_ref).exists():
                out["sim_ecapa_prompt"] = round(
                    cosine_sim(tts_ecapa, self.get_ecapa_embedding(prompt_ref)), 4
                )
            if heldout_ref and Path(heldout_ref).exists():
                out["sim_ecapa_heldout"] = round(
                    cosine_sim(tts_ecapa, self.get_ecapa_embedding(heldout_ref)), 4
                )
        except Exception as exc:
            print(f"[spk_sim] ECAPA error on {tts_path}: {exc}")

        try:
            tts_wavlm = self.get_wavlm_embedding(tts_path)
            if prompt_ref and Path(prompt_ref).exists():
                out["sim_wavlm_prompt"] = round(
                    cosine_sim(tts_wavlm, self.get_wavlm_embedding(prompt_ref)), 4
                )
            if heldout_ref and Path(heldout_ref).exists():
                out["sim_wavlm_heldout"] = round(
                    cosine_sim(tts_wavlm, self.get_wavlm_embedding(heldout_ref)), 4
                )
        except Exception as exc:
            print(f"[spk_sim] WavLM error on {tts_path}: {exc}")

        return out
