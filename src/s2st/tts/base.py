"""Base interface and runner for resumable zero-shot TTS synthesis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from tqdm import tqdm

from s2st.io import ProjectPaths, load_yaml
from s2st.manifest import append_manifest_record, deduplicate_manifest, get_completed_keys, read_manifest


class BaseZeroShotTTS(ABC):
    """Abstract base class for zero-shot voice cloning TTS systems."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model_name = config.get("model_name", "tts_model")
        self.sample_rate = int(config.get("sample_rate", 48000))

    @abstractmethod
    def synthesize_to_file(
        self,
        text: str,
        output_wav_path: str | Path,
        ref_audio_path: str | Path | None = None,
        preset_voice: str | None = None,
        denoise_ref: bool = True,
        language: str = "vi",
    ) -> Dict[str, Any]:
        """Synthesize `text` into `output_wav_path` and return synthesis metadata."""


def build_tts_engine(config: Dict[str, Any]) -> BaseZeroShotTTS:
    backend = config.get("backend", "vieneu")
    if backend == "vieneu":
        from s2st.tts.vieneu import VieNeuTTSEngine

        return VieNeuTTSEngine(config)
    if backend == "vixtts":
        from s2st.tts.vixtts import ViXTTSEngine

        return ViXTTSEngine(config)
    raise ValueError(f"Unknown TTS backend: {backend}")


def _resolve_ref_audio_for_record(
    rec: Dict[str, Any],
    ref_duration_tag: str = "8s",
    ref_quality: str = "clean",
) -> str:
    """Resolve the requested reference clip (`3s`/`5s`/`8s` and `clean`/`noisy`) from a manifest row."""
    refs_by_dur = rec.get("refs_by_duration") or {}
    if ref_duration_tag in refs_by_dur:
        entry = refs_by_dur[ref_duration_tag]
        if "noisy" in ref_quality and entry.get("noisy_ref_path"):
            return str(entry["noisy_ref_path"])
        if entry.get("clean_ref_path"):
            return str(entry["clean_ref_path"])
    return str(rec.get("ref_audio_path", ""))


def run_tts_stage(
    input_manifest_path: str | Path,
    output_manifest_path: str | Path,
    tts_config_path: str | Path,
    paths_cfg_path: str | Path = "configs/paths.yaml",
    run_id: str = "default",
    condition_tag: str = "clone_8s_clean_denoise_on",
    ref_duration_tag: str = "8s",
    ref_quality: str = "clean",
    denoise_ref: bool | None = None,
    use_preset_anchor: bool = False,
    same_lang_en_control: bool = False,
) -> Path:
    """Run resumable zero-shot TTS over `input_manifest_path`.

    Automatically skips utterances whose output WAV file already exists on Drive.
    """
    cfg = load_yaml(tts_config_path)
    paths = ProjectPaths(paths_cfg_path)
    paths.ensure_dirs()

    records = read_manifest(input_manifest_path)
    completed = get_completed_keys(
        output_manifest_path,
        key_field="utt_id",
        required_non_empty_fields=["tts_audio_path"],
        verify_file_field="tts_audio_path",
    )
    pending = [r for r in records if str(r.get("utt_id")) not in completed]
    model_name = cfg.get("model_name", "tts")
    print(
        f"[TTS:{model_name}:{condition_tag}] Total={len(records)} | Done={len(completed)} | Pending={len(pending)}"
    )
    if not pending:
        return Path(output_manifest_path)

    engine = build_tts_engine(cfg)
    effective_denoise = (
        bool(cfg.get("default_denoise", True)) if denoise_ref is None else bool(denoise_ref)
    )
    out_audio_dir = paths.get_dir("outputs_tts") / run_id / f"{model_name}__{condition_tag}"
    out_audio_dir.mkdir(parents=True, exist_ok=True)

    anchors = cfg.get("anchors", {})
    male_anchor = anchors.get("native_vi_male_north", "Hải Đăng")
    female_anchor = anchors.get("native_vi_female_north", "Mai Anh")

    for rec in tqdm(pending, desc=f"TTS ({model_name}:{condition_tag})"):
        utt_id = str(rec["utt_id"])
        out_wav = out_audio_dir / f"{utt_id}.wav"

        if same_lang_en_control:
            text_to_speak = str(rec.get("gold_en_text") or rec.get("stt_en_text") or "").strip()
            lang = "en"
        else:
            text_to_speak = str(rec.get("mt_vi_text") or rec.get("ref_vi_text") or "").strip()
            lang = "vi"

        if use_preset_anchor:
            gender = str(rec.get("gender", "M")).upper()
            preset_name = female_anchor if gender.startswith("F") else male_anchor
            meta = engine.synthesize_to_file(
                text=text_to_speak,
                output_wav_path=out_wav,
                ref_audio_path=None,
                preset_voice=preset_name,
                denoise_ref=False,
                language=lang,
            )
            used_ref = None
        else:
            used_ref = _resolve_ref_audio_for_record(
                rec, ref_duration_tag=ref_duration_tag, ref_quality=ref_quality
            )
            meta = engine.synthesize_to_file(
                text=text_to_speak,
                output_wav_path=out_wav,
                ref_audio_path=used_ref,
                preset_voice=None,
                denoise_ref=effective_denoise,
                language=lang,
            )

        merged = {
            **rec,
            "tts_model_name": model_name,
            "tts_condition_tag": condition_tag,
            "tts_input_text": text_to_speak,
            "tts_language": lang,
            "tts_used_ref_audio_path": used_ref,
            "tts_ref_duration_tag": ref_duration_tag,
            "tts_ref_quality": ref_quality,
            "tts_denoise_ref": effective_denoise,
            **meta,
        }
        append_manifest_record(merged, output_manifest_path)

    deduplicate_manifest(output_manifest_path, key_field="utt_id")
    return Path(output_manifest_path)
