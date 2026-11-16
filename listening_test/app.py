"""Gradio Listening Test App for evaluating Naturalness, Speaker Similarity, and Accentedness.

Run on Google Colab (with `share=True`) or deploy on a free Hugging Face CPU Space:
    python listening_test/app.py --stimuli-csv listening_test/stimuli.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime
from pathlib import Path
from typing import Any, Dict, List

import gradio as gr


def load_stimuli(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def create_app(stimuli_csv_path: Path, responses_csv_path: Path) -> gr.Blocks:
    base_dir = stimuli_csv_path.parent
    stimuli = load_stimuli(stimuli_csv_path)

    def get_item_view(idx: int):
        if not stimuli:
            return (
                "Chưa có file `stimuli.csv`. Hãy chạy `scripts/build_listening_test.py` trước.",
                None,
                None,
                "",
            )
        idx = max(0, min(idx, len(stimuli) - 1))
        row = stimuli[idx]
        ref_path = base_dir / row["ref_audio_relpath"]
        tts_path = base_dir / row["tts_audio_relpath"]
        header = f"### Mẫu {idx + 1} / {len(stimuli)} — Mã: `{row['stimulus_id']}`"
        text_info = (
            f"**Câu tiếng Việt (TTS đọc):** {row.get('vi_text', '')}\n\n"
            f"**Câu tiếng Anh gốc:** {row.get('en_text', '')}"
        )
        return (
            header,
            str(ref_path) if ref_path.exists() else None,
            str(tts_path) if tts_path.exists() else None,
            text_info,
        )

    def submit_and_next(
        listener_id: str,
        native_vi: str,
        idx: int,
        mos_naturalness: int,
        mos_similarity: int,
        mos_accentedness: int,
        guessed_origin: str,
        notes: str,
    ):
        if not listener_id.strip():
            raise gr.Error("Vui lòng nhập Mã người nghe (Listener ID) trước khi gửi!")
        if not stimuli:
            raise gr.Error("Không có mẫu âm thanh nào trong stimuli.csv.")

        row = stimuli[idx]
        responses_csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not responses_csv_path.exists()
        record = {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "listener_id": listener_id.strip(),
            "listener_type": native_vi,
            "stimulus_index": idx,
            "stimulus_id": row["stimulus_id"],
            "utt_id": row["utt_id"],
            "speaker_id": row["speaker_id"],
            "true_accent": row["accent"],
            "system": row["system"],
            "is_repeat_check": row.get("is_repeat_check", False),
            "mos_naturalness_1to5": mos_naturalness,
            "mos_similarity_1to5": mos_similarity,
            "mos_accentedness_1to5": mos_accentedness,
            "guessed_origin": guessed_origin,
            "notes": notes.strip(),
        }
        with responses_csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(record)

        next_idx = idx + 1
        if next_idx >= len(stimuli):
            return (
                idx,
                "### Đã hoàn thành toàn bộ bài nghe thử! Cảm ơn bạn rất nhiều!",
                None,
                None,
                "Bạn có thể đóng cửa sổ này.",
                "",
            )
        h, r_aud, t_aud, t_info = get_item_view(next_idx)
        return next_idx, h, r_aud, t_aud, t_info, ""

    with gr.Blocks(title="S2ST Voice & Accent Evaluation") as demo:
        gr.Markdown(
            "## Khảo sát nghe thử: Đánh giá độ giữ giọng và accent (Anh -> Việt)\n"
            "Hãy đeo tai nghe ở nơi yên tĩnh. Với mỗi mẫu, bạn nghe **Audio tham chiếu tiếng Anh** "
            "(giọng gốc) và **Audio tiếng Việt tổng hợp**, sau đó chấm 3 thang điểm 1–5."
        )
        with gr.Row():
            listener_id = gr.Textbox(label="Mã người nghe (VD: L01, tên viết tắt)", value="")
            native_vi = gr.Dropdown(
                choices=["Người Việt bản ngữ", "Song ngữ Việt-Anh", "Khác"],
                value="Người Việt bản ngữ",
                label="Nền tảng ngôn ngữ",
            )

        idx_state = gr.State(0)
        header_md = gr.Markdown()
        text_md = gr.Markdown()

        with gr.Row():
            ref_audio = gr.Audio(label="1. Giọng gốc tiếng Anh (Reference Speaker)", interactive=False)
            tts_audio = gr.Audio(label="2. Giọng tiếng Việt được tổng hợp (System Output)", interactive=False)

        mos_nat = gr.Slider(
            1, 5, value=3, step=1,
            label="A. Độ tự nhiên (Naturalness): 1 = Rất máy móc/lỗi nặng ... 5 = Tự nhiên như người thật",
        )
        mos_sim = gr.Slider(
            1, 5, value=3, step=1,
            label="B. Độ giống giọng người nói gốc (Speaker Similarity): 1 = Hoàn toàn khác người ... 5 = Chắc chắn cùng một người nói",
        )
        mos_acc = gr.Slider(
            1, 5, value=3, step=1,
            label="C. Mức độ mang âm hưởng/accent nước ngoài (Foreign Accentedness): 1 = Chuẩn giọng Việt bản ngữ (không có chất ngoại) ... 5 = Nghe rõ chất giọng/ngữ điệu người nước ngoài nói tiếng Việt",
        )
        guessed_origin = gr.Radio(
            choices=["Giọng Việt bản ngữ", "Chất giọng Anh-Mỹ (US)", "Chất giọng Anh-Anh (UK)", "Có chất nước ngoài nhưng không phân biệt được US/UK"],
            value="Có chất nước ngoài nhưng không phân biệt được US/UK",
            label="D. Cảm nhận nguồn gốc giọng nói (Tuỳ chọn)",
        )
        notes = gr.Textbox(label="Ghi chú thêm (lỗi phát âm, sai thanh điệu, ngắt nghỉ lạ...)", lines=1)
        submit_btn = gr.Button("Lưu & Chuyển sang mẫu tiếp theo", variant="primary")

        demo.load(
            fn=get_item_view,
            inputs=[idx_state],
            outputs=[header_md, ref_audio, tts_audio, text_md],
        )
        submit_btn.click(
            fn=submit_and_next,
            inputs=[listener_id, native_vi, idx_state, mos_nat, mos_sim, mos_acc, guessed_origin, notes],
            outputs=[idx_state, header_md, ref_audio, tts_audio, text_md, notes],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli-csv", default="listening_test/stimuli.csv")
    parser.add_argument("--responses-csv", default="listening_test/responses.csv")
    parser.add_argument("--share", action="store_true", default=True)
    args = parser.parse_args()

    app = create_app(Path(args.stimuli_csv), Path(args.responses_csv))
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
