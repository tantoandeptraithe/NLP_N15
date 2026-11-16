"""Single-File Self-Reference Speech-to-Speech Translation (S2ST) Pipeline.

Implements the exact user-specified architecture:
  Input English Audio (~10s)
      ├──> [1] STT (Whisper) -> English Text -> [2] MT -> Vietnamese Text
      └──> [3] Reference Extractor (uses the input audio itself, optionally sliced to 3s/5s/8s/10s)
                    │
                    ▼
          [4] Zero-Shot Voice Cloning TTS (VieNeu-TTS v3 Turbo or viXTTS)
                    │
                    ▼
          Output Vietnamese WAV (preserving the input speaker's timbre & prosody)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

from s2st.data.make_refs import build_speaker_reference_clips
from s2st.io import load_yaml
from s2st.mt.base import build_translator
from s2st.stt.faster_whisper import WhisperTranscriber
from s2st.tts.base import build_tts_engine


def slice_self_reference_from_input(
    input_audio_path: str | Path,
    work_dir: str | Path,
    utt_id: str = "input",
    durations_sec: Sequence[float] = (3.0, 5.0, 8.0, 10.0),
    target_sr: int = 48000,
    make_noisy_snr_db: float | None = 10.0,
    extra_same_speaker_paths: Sequence[str | Path] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Slice 3s / 5s / 8s / 10s reference clips directly from `input_audio_path`.

    If `input_audio_path` is shorter than a requested duration (e.g. a 6s VCTK clip when
    requesting a 10s reference) and `extra_same_speaker_paths` are provided, appends
    additional speech from the same speaker after the input utterance so `input_audio_path`
    is always the primary leading audio in the reference clip.
    """
    candidates = [Path(input_audio_path)]
    if extra_same_speaker_paths:
        for p in extra_same_speaker_paths:
            if Path(p) != Path(input_audio_path):
                candidates.append(Path(p))

    return build_speaker_reference_clips(
        candidate_audio_paths=candidates,
        output_dir=Path(work_dir) / "self_refs",
        speaker_id=utt_id,
        durations_sec=durations_sec,
        target_sr=target_sr,
        make_noisy_snr_db=make_noisy_snr_db,
    )


class SelfReferenceS2STPipeline:
    """End-to-end runner taking a single English audio file (~10s) and outputting cloned Vietnamese WAV."""

    def __init__(
        self,
        stt_config_path: str | Path = "configs/stt/faster_whisper_large_v3.yaml",
        mt_config_path: str | Path = "configs/mt/vinai_v2.yaml",
        tts_config_path: str | Path = "configs/tts/vieneu_v3turbo.yaml",
    ) -> None:
        self.stt_cfg = load_yaml(stt_config_path)
        self.mt_cfg = load_yaml(mt_config_path)
        self.tts_cfg = load_yaml(tts_config_path)
        self._stt = WhisperTranscriber(self.stt_cfg)
        self._mt = build_translator(self.mt_cfg)
        self._tts = build_tts_engine(self.tts_cfg)

    def translate_audio_file(
        self,
        input_en_wav_path: str | Path,
        output_vi_wav_path: str | Path,
        ref_duration_sec: float | None = None,
        denoise_ref: bool = True,
        work_dir: str | Path | None = None,
    ) -> Dict[str, Any]:
        """Run full cascade on `input_en_wav_path` using the input audio itself as `ref_audio`."""
        in_path = Path(input_en_wav_path)
        out_path = Path(output_vi_wav_path)
        w_dir = Path(work_dir) if work_dir else out_path.parent / "_work"

        # 1. STT: English Audio -> English Text
        stt_res = self._stt.transcribe_file(in_path)
        en_text = stt_res["stt_en_text"]

        # 2. MT: English Text -> Vietnamese Text
        mt_res = self._mt.translate_batch([en_text])[0]
        vi_text = mt_res["mt_vi_text"]

        # 3. Reference Audio Selection: use the input audio itself (or a specific 3s/5s/8s/10s slice of it)
        if ref_duration_sec is not None:
            slices = slice_self_reference_from_input(
                input_audio_path=in_path,
                work_dir=w_dir,
                utt_id=in_path.stem,
                durations_sec=[float(ref_duration_sec)],
                target_sr=int(self.tts_cfg.get("sample_rate", 48000)),
                make_noisy_snr_db=None,
            )
            ref_info = next(iter(slices.values()))
            used_ref_path = ref_info["clean_ref_path"]
        else:
            used_ref_path = str(in_path)

        # 4. Zero-shot Voice Cloning TTS: (Vietnamese Text + Self-Reference Audio) -> Vietnamese WAV
        tts_res = self._tts.synthesize_to_file(
            text=vi_text,
            output_wav_path=out_path,
            ref_audio_path=used_ref_path,
            preset_voice=None,
            denoise_ref=denoise_ref,
            language="vi",
        )

        return {
            "input_en_audio_path": str(in_path),
            "used_self_ref_audio_path": used_ref_path,
            "output_vi_audio_path": str(out_path),
            "stt_en_text": en_text,
            "mt_vi_text": vi_text,
            **stt_res,
            **mt_res,
            **tts_res,
        }
