#!/usr/bin/env python3
"""1-Click CLI for Self-Reference Speech-to-Speech Translation (English Audio ~10s -> Cloned Vietnamese WAV).

Usage:
  python scripts/translate_speech.py \
      --input-wav path/to/english_10s.wav \
      --output-wav path/to/vietnamese_cloned.wav \
      --mt-config configs/mt/vinai_v2.yaml \
      --tts-config configs/tts/vieneu_v3turbo.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from s2st.pipeline import SelfReferenceS2STPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate an English audio clip (~10s) into Vietnamese speech while preserving speaker voice."
    )
    parser.add_argument("--input-wav", required=True, help="Input English audio file (~3-10s)")
    parser.add_argument("--output-wav", required=True, help="Output cloned Vietnamese WAV path")
    parser.add_argument("--stt-config", default="configs/stt/faster_whisper_large_v3.yaml")
    parser.add_argument("--mt-config", default="configs/mt/vinai_v2.yaml")
    parser.add_argument("--tts-config", default="configs/tts/vieneu_v3turbo.yaml")
    parser.add_argument(
        "--ref-duration",
        type=float,
        default=None,
        help="Optional slice duration (e.g. 3.0, 5.0, 8.0, 10.0) from the input audio. Default: use full input audio.",
    )
    parser.add_argument("--no-denoise", action="store_true", help="Disable reference denoising")
    args = parser.parse_args()

    pipe = SelfReferenceS2STPipeline(
        stt_config_path=args.stt_config,
        mt_config_path=args.mt_config,
        tts_config_path=args.tts_config,
    )
    res = pipe.translate_audio_file(
        input_en_wav_path=Path(args.input_wav),
        output_vi_wav_path=Path(args.output_wav),
        ref_duration_sec=args.ref_duration,
        denoise_ref=not args.no_denoise,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
