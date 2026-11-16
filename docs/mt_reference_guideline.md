# Hướng dẫn tạo bản dịch tham chiếu tiếng Việt cho Tập B (40 câu In-Domain VCTK)

Để tránh thiên vị (bias) cho bất kỳ mô hình dịch máy nào trong Thí nghiệm E1, **40 câu tiếng Anh thuộc Tập B (trích từ VCTK)** cần được dịch thủ công sang tiếng Việt **từ đầu (from scratch)** thay vì sửa lại (post-edit) kết quả của một mô hình MT.

---

## Quy tắc dịch nhất quán (6 nguyên tắc)

1. **Văn phong nói tự nhiên (Spoken Vietnamese):**
   - Dịch sao cho khi đọc thành tiếng nghe tự nhiên, trôi chảy, đúng ngữ pháp tiếng Việt hiện đại.
   - Tránh dịch word-by-word kiểu văn dịch cứng nhắc (ví dụ: hạn chế lạm dụng *"bị/được"* khi không cần thiết).
2. **Đại từ nhân xưng mặc định:**
   - Khi câu đơn lẻ không có ngữ cảnh quan hệ xã hội cụ thể:
     - `I` $\rightarrow$ **tôi**
     - `you` $\rightarrow$ **bạn**
     - `we` $\rightarrow$ **chúng tôi** (hoặc **chúng ta** nếu mang tính kêu gọi chung)
     - `he / she` $\rightarrow$ **anh ấy / cô ấy** (hoặc **ông ấy / bà ấy** nếu ngữ cảnh trang trọng như chính trị, thời sự trong báo *The Herald* của VCTK).
3. **Giữ nguyên tên riêng (Proper Nouns):**
   - Tên người, địa danh, tổ chức (`Stella`, `Snow White`, `Scotland`, `Glasgow`, `Labour Party`...): **giữ nguyên chữ viết gốc** đối với tên người và tên địa danh riêng (ví dụ: `Stella`, `Glasgow`), chỉ dịch phần danh từ chung đi kèm nếu phổ biến (ví dụ: `Công đảng` hoặc giữ `Đảng Lao động`).
4. **Con số và đơn vị đo:**
   - Để phục vụ đánh giá cả COMET lẫn khả năng đọc của TTS, hãy viết con số thống nhất ở dạng chữ số chuẩn (ví dụ: `6`, `1998`, `50%`) như trong câu gốc; bộ tiền xử lý của TTS (`sea-g2p` hoặc `vinorm`) sẽ đảm nhiệm bước đọc số thành chữ.
5. **Độ dài câu (Lưu ý cho Baseline `viXTTS`):**
   - Đảm bảo câu dịch hoàn chỉnh có dấu chấm câu rõ ràng ở cuối câu (`.` / `?` / `!`).
6. **Quy trình kiểm tra chéo (Review):**
   - Sau khi bạn dịch xong 40 câu vào trường `"ref_vi_text"` trong file `mt_benchmark_100.jsonl`, hãy nhờ 1 người bạn đọc soát lại lỗi chính tả và dấu thanh trước khi chạy đánh giá COMET/BLEU.
