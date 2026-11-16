"""VieNeu-TTS v3 Turbo (48 kHz) Zero-Shot Voice Cloning wrapper.

Verified SDK usage from `github.com/pnnbao97/VieNeu-TTS`:
    from vieneu import Vieneu
    tts = Vieneu()  # Default = v3 Turbo (48 kHz), auto-detects PyTorch on CUDA or ONNX on CPU
    audio = tts.infer(text, ref_audio="ref.wav", denoise=True)
    tts.save(audio, "out.wav")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from s2st.io import write_audio_atomic
from s2st.tts.base import BaseZeroShotTTS


class VieNeuTTSEngine(BaseZeroShotTTS):
    """Wrapper around `vieneu.Vieneu` (v3 Turbo, 48 kHz) supporting cloning & preset anchors."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.sdk_mode = config.get("sdk_mode", "v3turbo")
        self.backend_pref = config.get("sdk_backend", None)  # None = auto (GPU PyTorch / CPU ONNX)
        self._engine: Any = None

    def load(self) -> None:
        if self._engine is not None:
            return
        from vieneu import Vieneu

        kwargs: Dict[str, Any] = {}
        if self.backend_pref:
            kwargs["backend"] = self.backend_pref
        self._engine = Vieneu(**kwargs)
        print(f"[TTS:VieNeu] Initialized VieNeu-TTS ({self.sdk_mode}) @ {self.sample_rate} Hz")

    def synthesize_to_file(
        self,
        text: str,
        output_wav_path: str | Path,
        ref_audio_path: str | Path | None = None,
        preset_voice: str | None = None,
        denoise_ref: bool = True,
        language: str = "vi",
    ) -> Dict[str, Any]:
        self.load()
        t0 = time.perf_counter()

        if preset_voice:
            audio = self._engine.infer(text, voice=preset_voice)
        elif ref_audio_path:
            audio = self._engine.infer(
                text,
                ref_audio=str(ref_audio_path),
                denoise=bool(denoise_ref),
            )
        else:
            audio = self._engine.infer(text)

        elapsed = time.perf_counter() - t0
        wav = np.asarray(audio, dtype=np.float32)
        out_path = write_audio_atomic(output_wav_path, wav, self.sample_rate)
        dur = len(wav) / float(self.sample_rate)
        rtf = elapsed / max(dur, 1e-6)

        return {
            "tts_audio_path": str(out_path),
            "tts_sample_rate": self.sample_rate,
            "tts_duration_sec": round(dur, 3),
            "tts_latency_sec": round(elapsed, 4),
            "tts_rtf": round(rtf, 4),
            "tts_preset_voice": preset_voice,
        }
