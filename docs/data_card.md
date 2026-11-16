# Data Card & Model Licensing Inventory

Tài liệu này ghi nhận nguồn gốc, giấy phép (license), tần số lấy mẫu và điều khoản sử dụng của toàn bộ các bộ dữ liệu và mô hình trong dự án (đã xác minh trực tiếp qua Hugging Face API / GitHub ngày `2026-11-10`).

---

## 1. Bộ dữ liệu (Datasets)

| Tên bộ dữ liệu | Hugging Face ID | Giấy phép | Tần số mẫu | Vai trò trong dự án | Lưu ý pháp lý & kỹ thuật |
|---|---|---|---|---|---|
| **CSTR VCTK Corpus** | `CSTR-Edinburgh/vctk` (Parquet mirror: `sanchit-gandhi/vctk`) | **CC BY 4.0** | 48 kHz (FLAC) | **Tập đánh giá chính (E2, E3, E4):** 12 speaker American + 12 speaker English, cùng điều kiện phòng thu. | Điều khoản VCTK yêu cầu không cố gắng xác định danh tính đời thực của người hiến giọng. |
| **GOOGLE FLEURS (`en_us`)** | `google/fleurs` | **CC BY 4.0** | 16 kHz | **Tập A của Benchmark MT (E1)** (60 câu) + kiểm tra STT. | Không có nhãn vùng miền chi tiết cho từng speaker nên không dùng cho kết luận US vs UK. |
| **FLORES+ (`eng_Latn`, `vie_Latn`)** | `openlanguagedata/flores_plus` | **CC BY-SA 4.0** (`gated=auto`) | Văn bản | Bản dịch tham chiếu tiếng Việt chuẩn cho 60 câu FLEURS. | Cần đăng nhập HF và bấm đồng ý điều khoản truy cập (`gated=auto`); yêu cầu ShareAlike nếu phân phối lại bản dịch. |
| **GLOBE_V2** | `MushanW/GLOBE_V2` | **CC0-1.0** | 44.1 kHz (supersampled) | Tập kiểm tra độ bền (robustness) trên dữ liệu crowdsourced (Common Voice). | Nhãn accent do người dùng tự khai báo; âm thanh được supersample lên 44.1 kHz. |
| **L2-ARCTIC** *(Dự phòng G3)* | `KoelLabs/L2Arctic` | **CC BY-NC 4.0** (`gated=auto`) | 44.1 kHz | Tập dự phòng có accent L2 đậm (Hindi, Mandarin, Vietnamese...) nếu US vs UK khó phân biệt ở mốc G3. | **Chỉ dùng cho mục đích nghiên cứu phi thương mại (Non-Commercial).** |
| **Bản thu bạn bè (Friends Set)** | Lưu nội bộ trên Drive | **Informed Consent** | 48 kHz | Đo trần Speaker Similarity xuyên ngôn ngữ thật (cùng 1 người nói cả câu tiếng Anh và tiếng Việt). | Bắt buộc có phiếu đồng ý (`docs/consent_form.md`); không công khai file audio nếu chưa được phép. |

---

## 2. Mô hình (Models)

| Thành phần | Hugging Face / GitHub ID | Giấy phép | Ghi chú |
|---|---|---|---|
| **STT Chính** | `Systran/faster-whisper-large-v3` | **MIT** | Chạy qua `faster-whisper==1.2.1`. |
| **MT Ứng viên A** | `vinai/vinai-translate-en2vi-v2` | **GNU AGPL-3.0** | 448M params; mBART fine-tuned chuyên cặp EN→VI. |
| **MT Ứng viên B** | `facebook/nllb-200-distilled-1.3B` | **CC-BY-NC-4.0** | **Phi thương mại.** Mã ngôn ngữ: `eng_Latn` → `vie_Latn`. |
| **MT Ứng viên C** | `Qwen/Qwen3.5-9B` | **Apache-2.0** | 9.65B params; bắt buộc tắt thinking mode (`enable_thinking: False`). |
| **MT Ứng viên D (Tuỳ chọn)** | `tencent/Hunyuan-MT-7B` | ⚠️ **Tencent Hunyuan Community License** | Cần kiểm tra kỹ giới hạn lãnh thổ/thương mại trong file LICENSE của repo gốc. |
| **TTS Chính** | `pnnbao-ump/VieNeu-TTS-v3-Turbo` | **Apache-2.0** | 48 kHz, SDK `vieneu==3.8.3`. Tự động cắt reference ≤ 8 giây. (Bản v4 là đóng mã nguồn, không sử dụng). |
| **TTS Baseline** | `capleaf/viXTTS` | **Coqui Public Model License (CPML)** | **Phi thương mại.** 24 kHz, fine-tune từ `coqui/XTTS-v2`. |
| **Speaker Similarity 1** | `speechbrain/spkrec-ecapa-voxceleb` | **Apache-2.0** | ECAPA-TDNN (16 kHz). |
| **Speaker Similarity 2** | `microsoft/wavlm-base-plus-sv` | ⚠️ Không gắn thẻ license trên HF | Kiểm tra license tại repo Microsoft UniSpeech trước khi phân phối sản phẩm. |
| **Automated MOS** | `sarulab-speech/UTMOSv2` | **MIT** | Dự phòng nhẹ: `tarepan/SpeechMOS` (MIT). |
| **Vietnamese Back-ASR** | `vinai/PhoWhisper-large` | **BSD-3-Clause** | Dùng để đo độ rõ (CER / Syllable WER) của giọng đọc TTS tiếng Việt. |
| **Accent Probe** | `Jzuluaga/accent-id-commonaccent_ecapa` | **MIT** | Chỉ dùng như phép đo thăm dò phụ trợ (exploratory). |
