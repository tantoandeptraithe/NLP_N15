# Đánh giá khả năng giữ giọng và accent của pipeline dịch giọng nói sang tiếng Việt dùng zero-shot voice cloning

Dự án nghiên cứu thực nghiệm đánh giá pipeline dịch giọng nói nối tiếp (**Cascade Speech-to-Speech Translation - S2ST**) từ **Tiếng Anh (US/UK accents)** sang **Tiếng Việt** sử dụng kỹ thuật **Zero-shot Voice Cloning**:

$$\text{Audio EN (US/UK)} \longrightarrow \text{STT (Whisper)} \longrightarrow \text{MT (EN}\to\text{VI)} \longrightarrow \text{Zero-shot TTS (VieNeu-TTS / viXTTS)} \longrightarrow \text{Audio VI}$$

---

## 1. Câu hỏi nghiên cứu (Research Questions)
1. **RQ1 (Timbre Preservation):** Pipeline giữ được bao nhiêu âm sắc (speaker similarity) của người nói gốc khi chuyển từ tiếng Anh sang tiếng Việt?
2. **RQ2 (Accent & Prosody Transfer):** Pipeline giữ được bao nhiêu đặc trưng phát âm/ngữ điệu (accentedness & prosody) của người nói gốc so với TTS baseline khác (`viXTTS`) và so với giọng đọc tiếng Việt bản ngữ (preset anchor)?
3. **RQ3 (Reference Sensitivity):** Độ dài (`3s` / `5s` / `8s`) và chất lượng/tiền xử lý khử nhiễu (`clean`, `noisy SNR=10dB`, `denoise=True/False`) của audio tham chiếu ảnh hưởng thế nào đến chất lượng tổng hợp và độ giữ giọng?

---

## 2. Kiến trúc cô lập môi trường (Stage-Isolated Architecture)

Để giải quyết xung đột phụ thuộc thực tế trên Google Colab (ví dụ: `vieneu[cuda]` ghim `torch==2.8.0`, trong khi `unbabel-comet==2.2.7` yêu cầu `numpy<2.0` và `transformers<5.0`), mỗi chặng của pipeline chạy trong một môi trường ảo `uv` độc lập và giao tiếp qua **JSONL Manifests** lưu trên Google Drive:

```
NLP_project/
├── pyproject.toml                 # Package nội bộ `s2st` (cài `pip install -e . --no-deps` ở mọi env)
├── requirements/                  # Danh sách thư viện ghim phiên bản cho từng stage
│   ├── stt.txt                    # faster-whisper==1.2.1, ctranslate2, jiwer, silero-vad
│   ├── mt.txt                     # transformers (cho VinAI, NLLB, Qwen3.5-9B, Hunyuan-MT)
│   ├── tts_vieneu.txt             # torch==2.8.0, transformers==4.57.6, vieneu==3.8.3
│   ├── tts_vixtts.txt             # coqui-tts==0.28.0 / viXTTS dependencies, vinorm, underthesea
│   ├── eval_mt.txt                # unbabel-comet==2.2.7 (numpy<2, transformers<5), sacrebleu
│   └── eval_audio.txt             # speechbrain==1.1.2, UTMOSv2, praat-parselmouth, transformers<5
├── configs/                       # Cấu hình YAML cho đường dẫn, dữ liệu, model và thí nghiệm E1-E4
├── src/s2st/                      # Mã nguồn core của dự án
│   ├── manifest.py                # Đọc/ghi JSONL nguyên tử (atomic write), idempotent resume
│   ├── cache.py                   # Sinh deterministic hash cho cấu hình & kiểm tra kết quả có sẵn
│   ├── runmeta.py                 # Lưu git commit, pip freeze, GPU info cho mỗi lần chạy
│   ├── data/                      # Tải VCTK/FLEURS/GLOBE, cắt reference 3/5/8s, trộn nhiễu SNR
│   ├── stt/                       # Wrapper faster-whisper large-v3 (kèm fallback HF Whisper)
│   ├── mt/                        # Wrapper Seq2Seq (VinAI, NLLB) và LLM (Qwen3.5-9B, Hunyuan-MT)
│   ├── tts/                       # Wrapper VieNeu-TTS v3 Turbo và viXTTS
│   └── eval/                      # WER, COMET/BLEU/chrF++, Speaker Sim, UTMOS, CER, Prosody, Stats
├── scripts/                       # CLI chạy hàng loạt (hỗ trợ resume khi Colab ngắt kết nối)
│   ├── setup_colab_envs.sh        # Khởi tạo các môi trường uv Python 3.11 trên Colab
│   ├── run_stage.py               # Chạy từng stage: prepare | stt | mt | tts
│   ├── run_eval.py                # Chấm điểm tự động: wer | mt | audio | all
│   ├── build_listening_test.py    # Đóng gói mẫu nghe thử chuẩn Latin-square
│   └── analyze.py                 # Tổng hợp bảng kết quả & kiểm định thống kê cho E1-E4
├── notebooks/                     # Notebooks chạy trên Google Colab
│   ├── 00_env_check.ipynb         # Kiểm tra GPU, mount Drive, dựng uv virtualenvs
│   ├── 01_e2e_smoke_test.ipynb    # Chạy thử end-to-end 5-10 mẫu từ đầu đến cuối
│   ├── 10_stt.ipynb               # Chạy STT + đánh giá WER theo accent
│   ├── 20_mt_benchmark.ipynb      # Benchmark 100 câu chọn model MT (E1)
│   ├── 30_tts_vieneu.ipynb        # Chạy VieNeu-TTS cho E2 & E3
│   ├── 31_tts_vixtts.ipynb        # Chạy baseline viXTTS cho E2 & E3
│   ├── 40_eval.ipynb              # Chạy toàn bộ thước đo tự động
│   └── 50_analysis.ipynb          # Phân tích lỗi mất accent (E4) & vẽ biểu đồ
├── listening_test/                # Ứng dụng Gradio thu thập đánh giá MOS / Accentedness
├── docs/                          # Data card, mẫu phiếu đồng ý (consent form), hướng dẫn dịch
└── tests/                         # Unit tests kiểm tra manifest, chuẩn hoá văn bản, trộn nhiễu
```

---

## 3. Hướng dẫn bắt đầu nhanh trên Google Colab

1. Clone repo vào Google Drive hoặc `/content/NLP_project` và mở [`notebooks/00_env_check.ipynb`](notebooks/00_env_check.ipynb) để tạo các môi trường `uv` và kiểm tra kết nối Drive.
2. Mở [`notebooks/01_e2e_smoke_test.ipynb`](notebooks/01_e2e_smoke_test.ipynb) trên runtime **T4 GPU** để chạy thử nghiệm 5–10 mẫu từ bộ `sanchit-gandhi/vctk` (Parquet mirror của CSTR VCTK) qua toàn bộ pipeline.
3. Khi chạy hàng loạt (E1–E4), mọi lệnh thông qua [`scripts/run_stage.py`](scripts/run_stage.py) đều tự động lưu trạng thái vào `MyDrive/s2st_vi/`; nếu Colab ngắt phiên, chỉ cần chạy lại cell để tiếp tục từ mẫu kế tiếp.

---

## 4. Giấy phép & Đạo đức nghiên cứu
- Mã nguồn khung dự án: **MIT License**.
- Các mô hình và bộ dữ liệu sử dụng tuân thủ giấy phép gốc của tác giả (xem chi tiết tại [`docs/data_card.md`](docs/data_card.md)):
  - `VieNeu-TTS-v3-Turbo` (Apache-2.0), `faster-whisper-large-v3` (MIT), `Qwen3.5-9B` (Apache-2.0).
  - Các thành phần **chỉ dành cho nghiên cứu phi thương mại (Non-Commercial)**: `facebook/nllb-200-distilled-1.3B` (CC-BY-NC-4.0), `capleaf/viXTTS` (Coqui Public Model License), `KoelLabs/L2Arctic` (CC-BY-NC-4.0).
  - `vinai/vinai-translate-en2vi-v2` (AGPL-3.0).
- **Nguyên tắc đạo đức Voice Cloning:** Chỉ nhân bản giọng từ bộ dữ liệu học thuật công khai hoặc từ người tình nguyện đã ký [`docs/consent_form.md`](docs/consent_form.md). Không công bố công khai các tệp âm thanh nhân bản của cá nhân khi chưa có sự đồng ý rõ ràng.
