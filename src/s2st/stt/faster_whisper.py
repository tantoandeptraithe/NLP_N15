"""Whisper STT wrapper using `faster-whisper` (with automatic fallback to HF Transformers)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from s2st.io import load_yaml
from s2st.manifest import append_manifest_record, deduplicate_manifest, get_completed_keys, read_manifest


class WhisperTranscriber:
    """Transcribes English audio files and extracts word-level timestamps."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model_id = config.get("model_id", "Systran/faster-whisper-large-v3")
        self.fallback_hf_id = config.get("fallback_hf_model_id", "openai/whisper-large-v3")
        self.device = config.get("device", "cuda")
        self.compute_type = config.get("compute_type", "float16")
        self.decode_opts = config.get("decode_options", {})
        self._backend = "faster_whisper"
        self._model: Any = None
        self._hf_pipe: Any = None

    def load(self) -> None:
        if self._model is not None or self._hf_pipe is not None:
            return
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_id,
                device=self.device,
                compute_type=self.compute_type,
            )
            self._backend = "faster_whisper"
            print(f"[STT] Loaded faster-whisper `{self.model_id}` ({self.compute_type})")
        except Exception as exc:
            print(f"[STT] faster-whisper failed ({exc}); falling back to HF `{self.fallback_hf_id}`")
            import torch
            from transformers import pipeline

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._hf_pipe = pipeline(
                "automatic-speech-recognition",
                model=self.fallback_hf_id,
                torch_dtype=dtype,
                device=0 if self.device == "cuda" else -1,
            )
            self._backend = "hf_transformers"

    def transcribe_file(self, audio_path: str | Path) -> Dict[str, Any]:
        self.load()
        t0 = time.perf_counter()
        if self._backend == "faster_whisper":
            segments_gen, info = self._model.transcribe(
                str(audio_path),
                language=self.decode_opts.get("language", "en"),
                task=self.decode_opts.get("task", "transcribe"),
                beam_size=int(self.decode_opts.get("beam_size", 5)),
                best_of=int(self.decode_opts.get("best_of", 5)),
                temperature=float(self.decode_opts.get("temperature", 0.0)),
                word_timestamps=bool(self.decode_opts.get("word_timestamps", True)),
                vad_filter=bool(self.decode_opts.get("vad_filter", True)),
                vad_parameters=self.decode_opts.get("vad_parameters"),
                condition_on_previous_text=bool(
                    self.decode_opts.get("condition_on_previous_text", False)
                ),
            )
            texts: List[str] = []
            words: List[Dict[str, Any]] = []
            for seg in segments_gen:
                texts.append(seg.text.strip())
                if seg.words:
                    for w in seg.words:
                        words.append(
                            {
                                "word": w.word.strip(),
                                "start": round(float(w.start), 3),
                                "end": round(float(w.end), 3),
                                "probability": round(float(w.probability), 4),
                            }
                        )
            elapsed = time.perf_counter() - t0
            return {
                "stt_en_text": " ".join(texts).strip(),
                "stt_words": words,
                "stt_language_prob": round(float(getattr(info, "language_probability", 1.0)), 4),
                "stt_latency_sec": round(elapsed, 4),
                "stt_backend": self._backend,
                "stt_model_id": self.model_id,
            }

        # Fallback HF pipeline
        res = self._hf_pipe(
            str(audio_path),
            generate_kwargs={"language": "english", "task": "transcribe"},
            return_timestamps="word",
        )
        elapsed = time.perf_counter() - t0
        return {
            "stt_en_text": str(res.get("text", "")).strip(),
            "stt_words": res.get("chunks", []),
            "stt_language_prob": 1.0,
            "stt_latency_sec": round(elapsed, 4),
            "stt_backend": self._backend,
            "stt_model_id": self.fallback_hf_id,
        }


def run_stt_stage(
    input_manifest_path: str | Path,
    output_manifest_path: str | Path,
    stt_config_path: str | Path = "configs/stt/faster_whisper_large_v3.yaml",
) -> Path:
    """Run resumable STT over an input manifest and write enriched records to `output_manifest_path`."""
    cfg = load_yaml(stt_config_path)
    records = read_manifest(input_manifest_path)
    completed = get_completed_keys(
        output_manifest_path,
        key_field="utt_id",
        required_non_empty_fields=["stt_en_text"],
    )
    pending = [r for r in records if str(r.get("utt_id")) not in completed]
    print(f"[STT] Total={len(records)} | Already done={len(completed)} | Pending={len(pending)}")
    if not pending:
        return Path(output_manifest_path)

    transcriber = WhisperTranscriber(cfg)
    for rec in tqdm(pending, desc=f"STT ({cfg.get('model_name', 'whisper')})"):
        stt_out = transcriber.transcribe_file(rec["source_audio_path"])
        merged = {**rec, **stt_out}
        append_manifest_record(merged, output_manifest_path)

    deduplicate_manifest(output_manifest_path, key_field="utt_id")
    return Path(output_manifest_path)
