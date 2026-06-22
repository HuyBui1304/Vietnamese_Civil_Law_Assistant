# HƯỚNG DẪN XÂY DỰNG ỨNG DỤNG RAG TỔNG HỢP
# Chủ đề: Chatbot Tra Cứu Bộ Luật Dân Sự Việt Nam 2015

---

## TỔNG QUAN DỰ ÁN

**Mục tiêu**: Xây dựng MỘT ứng dụng RAG duy nhất, tích hợp đầy đủ nội dung 6 bài học RAG.

**Dữ liệu**:
- File PDF scan Bộ Luật Dân Sự 2015 (172 trang, dạng ảnh scan không copy được text)
- 3-5 file audio giải thích luật dân sự (tải từ YouTube/podcast pháp luật)

**Giao diện truy vấn đa phương thức** (query-time):
- **Text**: Gõ câu hỏi trực tiếp
- **Audio (mic)**: Nói vào mic → Whisper transcribe → RAG
- **Ảnh chụp**: Ảnh chụp đoạn văn bản hoặc câu hỏi viết tay → OCR → RAG
- **Output**: Luôn trả text + tùy chọn trả lời bằng audio (TTS tiếng Việt)

**6 bài cần tích hợp**:

| Bài | Nội dung | Thể hiện qua |
|-----|----------|--------------|
| 0 - RAG cơ bản | Pipeline ingestion + query, FAISS, chatbot | Toàn bộ pipeline cơ bản của dự án |
| 1 - Fine-tuning LLM tiếng Việt | LoRA, dataset pháp luật | Tạo dataset Q&A pháp luật, fine-tune model hiểu domain luật |
| 2 - Multi-modal RAG | Xử lý nhiều loại media | Text (OCR từ PDF scan) + Audio (Whisper transcribe) |
| 3 - Agent-based Systems | LangChain/LangGraph, tools, memory | Agent pháp luật có tools tra cứu luật, tính toán, web search |
| 4 - Evaluation & Monitoring | Metrics đánh giá, LLM-as-judge | Đánh giá Recall@K, MRR, Faithfulness + bảng số liệu |
| 5 - Vector DB Optimization | FAISS → Chroma → Qdrant, hybrid search | **BẮT BUỘC có bảng so sánh trước/sau optimization** |

---

## BƯỚC 1: CHUẨN BỊ MÔI TRƯỜNG VÀ DỮ LIỆU

### Yêu cầu:
- Cài Python 3.10+, tạo virtual environment
- Cài tất cả thư viện cần thiết cho 6 bài (OCR, Whisper, FAISS, Chroma, Qdrant, LangChain, LangGraph, fine-tuning, evaluation...)
- Tạo file `.env` chứa GEMINI_API_KEY
- Đặt file PDF Bộ Luật Dân Sự vào thư mục `data/pdf/`
- Tải 3-5 file audio giải thích luật dân sự (mp3/wav) vào `data/audio/`
- Tạo cấu trúc thư mục project theo chuẩn

### Kết quả bước này:
- Project folder sẵn sàng, tất cả thư viện đã cài
- Dữ liệu PDF + audio đã có trong thư mục tương ứng

---

## BƯỚC 2: OCR TRÍCH XUẤT TEXT TỪ PDF SCAN (Bài 0 + Bài 2)

### Yêu cầu:
- File PDF là dạng **scan** (ảnh), không thể dùng pypdf extract text trực tiếp
- Cần convert từng trang PDF thành ảnh (dùng pdf2image)
- Dùng pytesseract với ngôn ngữ tiếng Việt (`lang='vie'`) để OCR từng trang
- Ghép toàn bộ text lại, lưu ra file `data/extracted_text/full_text.txt`
- In progress trong quá trình OCR (đang xử lý trang X/172)

### Tại sao bước này cover Bài 2:
- Bản chất PDF scan = image → OCR chính là **image processing** (1 modality)
- Bước sau sẽ thêm audio processing (modality thứ 2)

### Kết quả bước này:
- File `full_text.txt` chứa toàn bộ nội dung Bộ Luật Dân Sự đã OCR

---

## BƯỚC 3: TRANSCRIBE AUDIO BẰNG WHISPER (Bài 2)

### Yêu cầu:
- Dùng thư viện Whisper (model "base" hoặc "small")
- Transcribe tất cả file audio trong `data/audio/` với `language="vi"`
- Lưu transcript của mỗi file kèm metadata (tên file nguồn, loại media = "audio")
- Lưu tất cả transcript ra file JSON

### Kết quả bước này:
- Có transcript text từ tất cả audio files
- Kết hợp với bước 2, đã có đủ **2 modality**: text (từ OCR) + audio (từ Whisper) → cover Bài 2

---

## BƯỚC 4: CHUNKING VĂN BẢN (Bài 0 + Bài 5)

### Yêu cầu:
- Implement **2 phương pháp chunking** khác nhau (để sau so sánh hiệu quả):

**Phương pháp 1 — Fixed-size chunking (baseline)**:
- Cắt text thành chunks cố định 500 ký tự, overlap 100
- Đây là cách đơn giản nhất, dùng làm baseline

**Phương pháp 2 — Chunking theo Điều luật (semantic chunking)**:
- Dùng regex tìm pattern "Điều X." trong text
- Mỗi Điều luật = 1 chunk
- Kèm metadata: số điều, nguồn (PDF hay audio), loại media

- Chunking riêng cho audio transcript: cắt transcript thành chunks, gắn metadata `type: "audio"` và `source: tên_file_audio`

### Tại sao cần 2 phương pháp:
- Để **so sánh hiệu quả** ở bước evaluation (Bài 4 + Bài 5)
- Chunking theo Điều luật sẽ cho kết quả tốt hơn fixed-size → đây là điểm hay để trình bày

### Kết quả bước này:
- 2 bộ chunks: 1 bộ fixed-size, 1 bộ theo Điều luật
- Mỗi chunk đều có metadata (source, type, article_number nếu có)

---

## BƯỚC 5: EMBEDDING (Bài 0)

### Yêu cầu:
- Dùng SentenceTransformers model `all-MiniLM-L6-v2` (384 chiều)
- Embed tất cả chunks thành vectors
- Embed cả 2 bộ chunks (fixed-size và theo Điều luật)

### Kết quả bước này:
- 2 bộ embeddings tương ứng 2 bộ chunks

---

## BƯỚC 6: XÂY DỰNG 3 VECTOR DATABASE (Bài 0 + Bài 5)

### Yêu cầu:
Xây dựng **3 vector store** từ cùng dữ liệu, để sau so sánh hiệu quả:

**6a. FAISS Store (baseline — Bài 0)**:
- Dùng IndexFlatL2 (brute-force search)
- Lưu chunks mapping riêng bằng pickle (vì FAISS không lưu metadata)
- Đây là baseline đơn giản nhất

**6b. Chroma Store (development — Bài 5)**:
- Dùng PersistentClient (lưu disk)
- Lưu document kèm metadata (source, type, article_number)
- Hỗ trợ metadata filtering (ví dụ: chỉ tìm trong audio, chỉ tìm Điều 100-200...)

**6c. Qdrant Store (production — Bài 5)**:
- Dùng QdrantClient (in-memory hoặc Docker)
- Hỗ trợ metadata filtering nâng cao
- Batch insert (mỗi batch 100 points)
- Thử nghiệm HNSW parameters và quantization nếu có thể

### Kết quả bước này:
- 3 vector store chứa cùng dữ liệu, sẵn sàng để benchmark so sánh

---

## BƯỚC 7: BENCHMARK VECTOR DB + BẢNG SO SÁNH (Bài 5 — RẤT QUAN TRỌNG)

### Yêu cầu:

**7a. Tạo test set benchmark**:
- Tạo 30-50 câu hỏi pháp luật kèm ground truth (điều luật đúng cần tìm ra)
- Ví dụ: "Quyền dân sự bị hạn chế khi nào?" → ground truth: Điều 2
- Có thể dùng Gemini để tự sinh câu hỏi từ text luật

**7b. Chạy benchmark trên cả 3 vector store**:
- Cùng bộ câu hỏi, hỏi cả 3 store, đo:
  - **Recall@3**: trong top 3 kết quả có chứa điều luật đúng không?
  - **Recall@5**: trong top 5?
  - **MRR (Mean Reciprocal Rank)**: vị trí trung bình của kết quả đúng
  - **Avg Latency (ms)**: thời gian search trung bình
  - **Faithfulness**: dùng LLM-as-judge đánh giá câu trả lời có đúng theo điều luật không

**7c. Tạo BẢNG SO SÁNH 1 — Vector DB**:

| Metric | FAISS (Baseline) | Chroma | Qdrant |
|--------|-----------------|--------|--------|
| Recall@3 | ? | ? | ? |
| Recall@5 | ? | ? | ? |
| MRR | ? | ? | ? |
| Avg Latency (ms) | ? | ? | ? |
| Faithfulness | ? | ? | ? |
| Metadata Filter | Không hỗ trợ | Có | Có |
| Storage (MB) | ? | ? | ? |

**7d. Tạo BẢNG SO SÁNH 2 — Chunking Strategy** (cùng 1 vector store, so sánh 2 cách chunk):

| Metric | Fixed 500 chars | Theo Điều luật |
|--------|----------------|----------------|
| Recall@3 | ? | ? |
| Recall@5 | ? | ? |
| MRR | ? | ? |
| Faithfulness | ? | ? |

**7e. Viết nhận xét**:
- Giải thích tại sao Qdrant/Chroma tốt hơn FAISS (metadata filter, advanced indexing)
- Giải thích tại sao chunking theo Điều luật tốt hơn fixed-size (semantic boundary)
- Ghi rõ trade-off: latency vs accuracy, memory vs disk

### Kết quả bước này:
- 2 bảng so sánh số liệu cụ thể (đây là thứ giảng viên yêu cầu)
- File CSV hoặc markdown chứa kết quả benchmark

---

## BƯỚC 8: GIAO DIỆN TRUY VẤN ĐA PHƯƠNG THỨC (Bài 0 + Bài 2)

### Yêu cầu:
Xây dựng một hàm `query()` thống nhất nhận 3 loại input và pipeline xử lý tương ứng:

**8a. Audio input (mic)**:
- Ghi âm từ microphone bằng `sounddevice` (N giây, user tự chọn)
- Lưu ra file WAV tạm → Whisper transcribe → text → RAG
- Xóa file WAV tạm sau khi transcribe

**8b. Image input (ảnh chụp)**:
- Nhận ảnh chụp đoạn văn bản hoặc câu hỏi viết tay
- Dùng lại pipeline OCR của Bước 2 (`pytesseract`, `lang='vie'`) cho ảnh đơn lẻ
- text sau OCR → RAG

**8c. Text input**:
- Nhận trực tiếp string từ user → RAG

**8d. RAG pipeline** (chung cho cả 3 input):
- Embed câu hỏi (SentenceTransformers) → retrieve top-5 chunks từ Qdrant
- Ghép context → Gemini generate câu trả lời tiếng Việt

**8e. TTS output (tùy chọn)**:
- Nếu user yêu cầu: chuyển câu trả lời thành audio bằng `gTTS` (`lang='vi'`)
- Lưu ra `outputs/response_<timestamp>.mp3`

**8f. Interactive CLI loop**:
- Menu chọn mode: `t` (text) / `m` (mic) / `i` (ảnh) / `q` (thoát)
- Sau mỗi câu hỏi hỏi thêm: "Trả lời bằng audio? [y/N]"

### Công nghệ:
- Mic recording: `sounddevice` + `scipy.io.wavfile`
- Query image OCR: `pytesseract` (tái sử dụng Bước 2)
- Whisper: tái sử dụng Bước 3
- TTS: `gTTS`

### Kết quả bước này:
- Hàm `query()` chấp nhận text / audio_path / image_path / use_mic=True
- Interactive CLI chatbot đa phương thức hoạt động
- Tùy chọn output audio cho mọi câu trả lời

---

## BƯỚC 9: AGENT-BASED SYSTEM (Bài 3)

### Yêu cầu:

**9a. Định nghĩa Tools**:
- **Tool 1 — RAG Search**: Tìm điều luật trong vector DB (Qdrant hoặc Chroma)
- **Tool 2 — Calculator**: Tính toán thời hiệu khởi kiện, lãi suất chậm trả, phân chia tài sản
- **Tool 3 — Web Search**: Dùng DuckDuckGo search tìm án lệ, bình luận pháp luật trên internet

**9b. ReAct Agent với LangChain**:
- Tạo agent dùng AgentType.ZERO_SHOT_REACT_DESCRIPTION
- Agent tự quyết định dùng tool nào dựa trên câu hỏi
- Bật verbose=True để thấy quá trình reasoning
- Ví dụ test: "Thời hiệu khởi kiện hợp đồng dân sự là bao lâu? Nếu hợp đồng ký ngày 01/01/2020 thì hết hạn khởi kiện ngày nào?"
  → Agent sẽ: RAG Search (tìm điều luật về thời hiệu) → Calculator (tính ngày hết hạn) → trả lời

**9c. LangGraph Agent** (nâng cao):
- Tạo state graph với các node: Plan → Research → Synthesize → Critique
- Có conditional edge: nếu quality_score < 7 → quay lại Research
- Ví dụ: câu hỏi phức tạp cần tra nhiều điều luật khác nhau

**9d. Memory**:
- Thêm ConversationBufferMemory để agent nhớ lịch sử hội thoại
- User hỏi tiếp "Điều luật đó có ngoại lệ gì không?" → agent hiểu "đó" là điều luật vừa trả lời

### Kết quả bước này:
- Agent pháp luật có khả năng: tra cứu luật + tính toán + tìm web + nhớ context hội thoại

---

## BƯỚC 10: FINE-TUNING LLM CHO DOMAIN PHÁP LUẬT (Bài 1)

### Yêu cầu:

**10a. Tạo dataset Q&A pháp luật**:
- Dùng Gemini để tự sinh cặp câu hỏi-trả lời từ text Bộ Luật Dân Sự
- Mỗi điều luật → sinh 2-3 câu hỏi
- Mục tiêu: 500-1000 cặp Q&A
- Format instruction-following: {"instruction": "...", "input": "câu hỏi", "output": "câu trả lời"}
- Lưu ra file `data/dataset/qa_dataset.json`

**10b. Fine-tune với LoRA trên Google Colab** (vì cần GPU):
- Chọn model base: google/gemma-2-2b hoặc model tiếng Việt phù hợp
- Cấu hình LoRA: r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"]
- Train 3 epochs, learning_rate=2e-4
- Lưu LoRA adapter

**10c. Kết hợp RAG + Fine-tuned model**:
- Thay LLM trong pipeline RAG bằng fine-tuned model
- So sánh chất lượng trả lời: base model vs fine-tuned model (cùng context)

### Lưu ý:
- Bước này chạy trên **Google Colab** (cần GPU), không chạy local
- Nếu không đủ resource, có thể dùng model nhỏ hơn hoặc giảm dataset

### Kết quả bước này:
- Dataset Q&A pháp luật
- LoRA adapter đã train
- So sánh chất lượng base vs fine-tuned

---

## BƯỚC 11: EVALUATION & MONITORING ĐẦY ĐỦ (Bài 4)

### Yêu cầu:

**11a. Tạo test dataset**:
- Tách 50 câu hỏi từ dataset (không trùng với training set)
- Mỗi câu có: question, ground_truth_answer, relevant_article (số điều luật đúng)

**11b. Retrieval Metrics**:
- **Recall@K**: với K=3 và K=5, đếm bao nhiêu câu hỏi mà điều luật đúng nằm trong top-K
- **MRR**: tính 1/rank trung bình của kết quả đúng đầu tiên

**11c. Generation Metrics (LLM-as-judge)**:
- **Faithfulness**: Dùng Gemini đánh giá — câu trả lời có chứa thông tin KHÔNG có trong context không? → "Faithful" hoặc "Hallucinated"
- **Answer Relevance**: Câu trả lời có đúng với câu hỏi không? → điểm 1-5

**11d. Tạo BẢNG TỔNG HỢP EVALUATION**:

| Metric | FAISS + Fixed Chunk | Chroma + Điều Luật | Qdrant + Điều Luật |
|--------|--------------------|--------------------|---------------------|
| Recall@3 | ? | ? | ? |
| Recall@5 | ? | ? | ? |
| MRR | ? | ? | ? |
| Faithfulness | ? | ? | ? |
| Answer Relevance | ? | ? | ? |
| Avg Latency (ms) | ? | ? | ? |

### Kết quả bước này:
- Bảng evaluation đầy đủ, chứng minh hệ thống sau optimization tốt hơn baseline

---

## BƯỚC 12: DEPLOY API (Bài 3)

### Yêu cầu:
- Tạo FastAPI server với endpoint `/ask` nhận câu hỏi, trả về câu trả lời
- Có session management (mỗi user có conversation history riêng)
- Error handling: timeout, retry, fallback
- Endpoint `/health` để kiểm tra hệ thống

### Kết quả bước này:
- API server chạy được, có thể gọi từ Postman hoặc frontend

---

## BƯỚC 13: VIẾT BÁO CÁO TỔNG HỢP

### Yêu cầu:
- Tổng hợp tất cả kết quả vào 1 báo cáo
- **BẮT BUỘC** có các bảng so sánh số liệu:
  - Bảng 1: So sánh 3 Vector DB (FAISS vs Chroma vs Qdrant)
  - Bảng 2: So sánh 2 Chunking Strategy (Fixed vs Theo Điều luật)
  - Bảng 3: Evaluation tổng hợp (trước/sau optimization)
  - Bảng 4: So sánh base model vs fine-tuned model (nếu có)
- Nhận xét, giải thích kết quả
- Kiến trúc tổng quan hệ thống (sơ đồ pipeline)
- Hướng phát triển tiếp theo

---

## TÓM TẮT: MỖI BƯỚC COVER BÀI NÀO

| Bước | Nội dung | Cover bài |
|------|----------|-----------|
| 1 | Chuẩn bị môi trường + dữ liệu | Nền tảng |
| 2 | OCR từ PDF scan | Bài 0 + Bài 2 (image modality) |
| 3 | Whisper transcribe audio | Bài 2 (audio modality) |
| 4 | Chunking (2 phương pháp) | Bài 0 + Bài 5 |
| 5 | Embedding | Bài 0 |
| 6 | Xây dựng 3 Vector DB | Bài 0 + Bài 5 |
| 7 | Benchmark + bảng so sánh | **Bài 5 (trọng tâm)** + Bài 4 |
| 8 | Giao diện truy vấn đa phương thức (text/mic/ảnh + TTS) | Bài 0 + Bài 2 |
| 9 | Agent + Tools + Memory | Bài 3 |
| 10 | Fine-tuning LoRA | Bài 1 |
| 11 | Evaluation đầy đủ | Bài 4 |
| 12 | Deploy FastAPI | Bài 3 |
| 13 | Báo cáo + bảng số liệu | Tổng hợp tất cả |
