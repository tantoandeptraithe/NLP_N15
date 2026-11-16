"""Automated Naturalness MOS prediction via UTMOSv2 (`sarulab-speech/UTMOSv2`) and SpeechMOS fallback.

Verified UTMOSv2 API from `github.com/sarulab-speech/UTMOSv2`:
    import utmosv2
    model = utmosv2.create_model(pretrained=True)
    mos = model.predict(input_path="/path/to/file.wav")
Fallback lightweight predictor from `github.com/tarepan/SpeechMOS`:
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from s2st.io import read_audio_mono


class AutomatedMOSScorer:
    """Computes UTMOSv2 (or fallback UTMOS22) naturalness score for synthesized audio."""

    def __init__(self, prefer_utmosv2: bool = True, device: str = "cuda") -> None:
        self.prefer_utmosv2 = prefer_utmosv2
        self.device_name = device
        self._utmosv2_model: Any = None
        self._utmos22_model: Any = None
        self._backend: str = "uninitialized"

    def load(self) -> None:
        if self._backend != "uninitialized":
            return
        if self.prefer_utmosv2:
            try:
                import utmosv2

                self._utmosv2_model = utmosv2.create_model(pretrained=True)
                self._backend = "utmosv2"
                print("[MOS] Loaded UTMOSv2 (`sarulab-speech/UTMOSv2`)")
                return
            except Exception as exc:
                print(f"[MOS] UTMOSv2 unavailable ({exc}); falling back to SpeechMOS `utmos22_strong`")

        import torch

        self._utmos22_model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
        )
        if self.device_name == "cuda" and torch.cuda.is_available():
            self._utmos22_model = self._utmos22_model.cuda()
        self._utmos22_model.eval()
        self._backend = "utmos22_strong"
        print("[MOS] Loaded SpeechMOS `utmos22_strong`")

    def score_file(self, audio_path: str | Path) -> float | None:
        if not audio_path or not Path(audio_path).exists():
            return None
        self.load()
        if self._backend == "utmosv2":
            val = self._utmosv2_model.predict(input_path=str(audio_path))
            if hasattr(val, "item"):
                val = val.item()
            elif isinstance(val, (list, tuple)):
                val = val[0]
            return round(float(val), 4)

        import torch

        wav, sr = read_audio_mono(audio_path, target_sr=16000)
        tensor = torch.from_numpy(wav).unsqueeze(0)
        if self.device_name == "cuda" and torch.cuda.is_available():
            tensor = tensor.cuda()
        with torch.inference_mode():
            score = self._utmos22_model(tensor, sr)
        return round(float(score.squeeze().cpu().item()), 4)
