# Trợ lý RAG Bộ luật Dân sự Việt Nam 2015

Ứng dụng hỏi đáp pháp luật bằng tiếng Việt sử dụng Retrieval-Augmented Generation (RAG), Google Gemini và Qdrant. Dự án hỗ trợ câu hỏi dạng văn bản, âm thanh, hình ảnh và có pipeline benchmark, đánh giá retrieval, cùng thử nghiệm fine-tuning LoRA.

> **Lưu ý:** Kết quả do mô hình sinh ra chỉ phục vụ học tập và tham khảo, không thay thế tư vấn pháp lý chuyên nghiệp.

## Tính năng

- Truy xuất điều luật bằng embedding `all-MiniLM-L6-v2` và Qdrant.
- So sánh ba vector store: FAISS, Chroma và Qdrant.
- Router chọn RAG, tìm kiếm web hoặc công cụ tính toán theo câu hỏi.
- Nhận dạng giọng nói bằng Whisper và OCR bằng Tesseract.
- API FastAPI, giao diện web tĩnh và chatbot trên terminal.
- Benchmark Recall@K, MRR, latency và đánh giá bằng LLM-as-a-judge.
- Notebook tạo tập Q&A và fine-tuning LoRA trên Google Colab.

## Kiến trúc

```text
PDF / audio / image
        |
        v
OCR / Whisper -> chunking -> embedding -> vector store
                                             |
Question -> router -> RAG / web / calculator -> Gemini -> response
```

## Cấu trúc chính

```text
.
├── api.py                  # FastAPI và các endpoint đa phương thức
├── run_chatbot.py          # Chatbot trên terminal
├── chat_ui.html            # Giao diện web tĩnh
├── main.ipynb              # Notebook chạy toàn bộ pipeline
├── ocr.py                  # Trích xuất văn bản từ PDF/ảnh
├── transcribe.py           # Chuyển giọng nói thành văn bản
├── chunking.py             # Fixed-size và article-aware chunking
├── embedding.py            # Sinh sentence embeddings
├── vectordb.py             # FAISS, Chroma và Qdrant
├── query.py                # Pipeline truy vấn RAG
├── agent.py                # ReAct agent và LangGraph workflow
├── benchmark.py            # Benchmark retrieval
├── evaluation.py           # Đánh giá chất lượng
├── finetune.py             # Tạo dữ liệu và fine-tuning LoRA
├── models/train_colab.ipynb
└── data/script_question_audio.txt
```

Các file đầu vào lớn, embedding, vector database, model adapter và kết quả chạy được giữ cục bộ, không commit vào Git. Chúng được tạo lại bằng `main.ipynb`.

## Yêu cầu

- Python 3.11
- Tesseract OCR và gói ngôn ngữ tiếng Việt
- Poppler
- FFmpeg (cho Whisper)

macOS:

```bash
brew install tesseract tesseract-lang poppler ffmpeg
```

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-vie poppler-utils ffmpeg
```

## Cài đặt

```bash
git clone <repository-url>
cd BCCK_MXH
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Điền khóa vào `.env`:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
```

Không commit `.env` hoặc bất kỳ API key nào. Nếu một khóa từng được chia sẻ hoặc commit, hãy thu hồi khóa đó và tạo khóa mới.

## Chuẩn bị dữ liệu

Đặt dữ liệu đầu vào tại:

```text
data/pdf/          # PDF Bộ luật Dân sự
data/audio/        # MP3/WAV tùy chọn
data/test_image/   # Ảnh OCR tùy chọn
```

Sau đó mở `main.ipynb` và chạy các cell theo thứ tự để tạo văn bản OCR, chunks, embeddings và vector stores. Có thể kiểm tra môi trường trước bằng:

```bash
python setup.py
```

## Chạy ứng dụng

Khởi động API:

```bash
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Mở `chat_ui.html` trong trình duyệt, hoặc chạy chatbot terminal:

```bash
python run_chatbot.py
```

API mặc định có tài liệu tương tác tại `http://127.0.0.1:8000/docs`.

## API

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/health` | Kiểm tra trạng thái dịch vụ |
| `POST` | `/ask` | Hỏi đáp bằng văn bản |
| `POST` | `/ask/multimodal` | Nhận văn bản, audio hoặc ảnh |
| `GET` | `/audio/{filename}` | Phục vụ audio liên quan |
| `POST` | `/tts` | Chuyển văn bản thành tiếng nói |

Ví dụ:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Quyền dân sự được xác lập dựa trên những căn cứ nào?"}'
```

## Tái tạo pipeline

1. `setup.py`: kiểm tra môi trường và dữ liệu.
2. `ocr.py`, `transcribe.py`: trích xuất nội dung đầu vào.
3. `chunking.py`, `embedding.py`: tạo chunks và vectors.
4. `vectordb.py`: xây dựng FAISS, Chroma và Qdrant.
5. `benchmark.py`, `evaluation.py`: benchmark và đánh giá.
6. `api.py` hoặc `run_chatbot.py`: phục vụ ứng dụng.

## Bảo mật và dữ liệu

- `.env`, model, vector store, dữ liệu đầu vào và artifact sinh ra đều được `.gitignore` loại trừ.
- API hiện dành cho môi trường phát triển. Hãy giới hạn CORS, thêm xác thực và rate limiting trước khi triển khai công khai.
- Kiểm tra quyền sử dụng của PDF, audio và model weights trước khi phân phối.

## Đóng góp

Tạo branch riêng, giữ thay đổi tập trung và bảo đảm lệnh sau chạy thành công trước khi mở pull request:

```bash
python -m compileall -q .
```
