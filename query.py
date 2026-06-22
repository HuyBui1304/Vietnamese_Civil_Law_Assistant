"""
BƯỚC 8: Giao diện truy vấn đa phương thức  (Bài 0 + Bài 2)
=============================================================
Nhận 3 loại input tại QUERY TIME:

  [Text]        ──────────────────────────────────────────────┐
  [Audio/Mic]   → Whisper transcribe → text ─────────────────┤→ RAG → Gemini → Answer
  [Ảnh chụp]   → pytesseract OCR    → text ─────────────────┘
                                                              ↓
                                          [Optional: gTTS → audio response]

Cách dùng nhanh:
    from query import query, interactive_loop

    # Text
    result = query(text="Quyền dân sự là gì?")

    # Mic (ghi âm 5 giây)
    result = query(use_mic=True, mic_duration=5)

    # Ảnh chụp câu hỏi
    result = query(image_path="cauhoi.jpg")

    # Bất kỳ + trả lời bằng audio
    result = query(text="...", respond_with_audio=True)

    # CLI loop tương tác
    interactive_loop(vector_store=qdrant_store, gemini_model=model)
"""

import time
import tempfile
from pathlib import Path


# ================================================================ #
#  INPUT: Audio (mic hoặc file)                                    #
# ================================================================ #

def record_from_mic(duration: int = 5, samplerate: int = 16_000) -> str:
    """Ghi âm từ microphone, lưu WAV tạm, trả đường dẫn file.

    Args:
        duration:   Số giây ghi âm.
        samplerate: Sample rate (Hz). Whisper dùng 16000.

    Returns:
        Đường dẫn file WAV tạm (cần xóa sau khi dùng).
    """
    try:
        import sounddevice as sd
        from scipy.io.wavfile import write as wav_write
    except ImportError:
        raise ImportError("pip install sounddevice scipy")

    print(f"  🎙  Ghi âm {duration} giây... Hãy nói câu hỏi của bạn!")
    audio_data = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    print("  ✓ Ghi xong!")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_write(tmp.name, samplerate, audio_data)
    return tmp.name


def transcribe_audio(audio_path: str, model_size: str = "base") -> str:
    """Whisper transcribe file audio → text tiếng Việt.

    Args:
        audio_path: Đường dẫn file audio (wav/mp3/...).
        model_size: Kích thước model Whisper ('tiny'/'base'/'small').
                    'base' là đủ tốt và nhanh cho tiếng Việt.

    Returns:
        Text đã transcribe.
    """
    try:
        import whisper
    except ImportError:
        raise ImportError("pip install openai-whisper")

    print(f"  Đang transcribe bằng Whisper ({model_size})...")
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language="vi")
    text = result["text"].strip()
    print(f"  ✓ Transcribe xong: {text!r}")
    return text


# ================================================================ #
#  INPUT: Ảnh chụp (OCR)                                          #
# ================================================================ #

def ocr_image(image_path: str | Path, lang: str = "vie") -> str:
    """OCR ảnh chụp câu hỏi → text.

    Tái sử dụng pytesseract từ Bước 2 (ocr.py) nhưng nhận
    ảnh đơn lẻ thay vì PDF.

    Args:
        image_path: Đường dẫn ảnh (jpg/png/...).
        lang:       Ngôn ngữ Tesseract (mặc định 'vie').

    Returns:
        Text trích xuất từ ảnh.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        raise ImportError("pip install pytesseract Pillow")

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    print(f"  Đang OCR ảnh: {image_path.name}...")
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=lang).strip()

    if not text:
        print("  ⚠  OCR không nhận được text. Kiểm tra:")
        print("     - Ảnh đủ rõ nét?")
        print("     - Tesseract gói 'vie' đã cài?  (brew install tesseract-lang)")
    else:
        print(f"  ✓ OCR xong: {len(text)} ký tự — {text[:60]!r}...")

    return text


# ================================================================ #
#  OUTPUT: TTS (text → audio)                                      #
# ================================================================ #

def text_to_speech(
    text: str,
    output_path: str | Path | None = None,
    lang: str = "vi",
) -> str:
    """Chuyển câu trả lời thành audio MP3 bằng gTTS.

    Args:
        text:        Câu trả lời cần đọc.
        output_path: Nơi lưu MP3. Mặc định: outputs/response_<ts>.mp3
        lang:        Ngôn ngữ gTTS ('vi' = tiếng Việt).

    Returns:
        Đường dẫn file MP3 đã lưu.
    """
    try:
        from gtts import gTTS
    except ImportError:
        raise ImportError("pip install gTTS")

    if output_path is None:
        ts = int(time.time())
        output_path = Path("outputs") / f"response_{ts}.mp3"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(str(output_path))
    print(f"  ✓ Audio response: {output_path}")
    return str(output_path)


# ================================================================ #
#  RAG Pipeline (gọi sau khi đã có text câu hỏi)                  #
# ================================================================ #

def _rag_generate(question: str, vector_store, gemini_model) -> str:
    """Retrieve + generate câu trả lời từ RAG pipeline.

    Args:
        question:     Câu hỏi đã được chuẩn hoá thành text.
        vector_store: Đối tượng vector store (Qdrant/Chroma/FAISS).
                      Phải có method .search(query, top_k) trả list[dict{"text":...}]
        gemini_model: google.generativeai.GenerativeModel instance.

    Returns:
        Câu trả lời từ Gemini.
    """
    # Retrieve
    results = vector_store.search(question, top_k=5)
    context = "\n\n---\n\n".join(r["text"] for r in results)

    # Generate
    prompt = (
        "Bạn là trợ lý pháp luật chuyên về Bộ Luật Dân Sự Việt Nam 2015.\n"
        "Hãy trả lời câu hỏi dựa VÀO NGỮ CẢNH được cung cấp, bằng tiếng Việt.\n"
        "Nếu ngữ cảnh không đủ thông tin, hãy nói rõ thay vì bịa đặt.\n\n"
        f"NGỮ CẢNH:\n{context}\n\n"
        f"CÂU HỎI: {question}\n\n"
        "TRẢ LỜI:"
    )
    response = gemini_model.generate_content(prompt)
    return response.text.strip()


# ================================================================ #
#  Hàm truy vấn thống nhất                                        #
# ================================================================ #

def query(
    text: str | None = None,
    audio_path: str | None = None,
    image_path: str | None = None,
    use_mic: bool = False,
    mic_duration: int = 5,
    whisper_model: str = "base",
    respond_with_audio: bool = False,
    vector_store=None,
    gemini_model=None,
) -> dict:
    """Giao diện truy vấn thống nhất — nhận text / audio / ảnh.

    Ưu tiên: use_mic > audio_path > image_path > text

    Args:
        text:               Câu hỏi dạng string.
        audio_path:         Đường dẫn file audio có sẵn (mp3/wav).
        image_path:         Đường dẫn ảnh chụp câu hỏi.
        use_mic:            Ghi âm từ mic (bỏ qua audio_path).
        mic_duration:       Thời gian ghi âm (giây).
        whisper_model:      Kích thước model Whisper ('base'/'small').
        respond_with_audio: Nếu True, xuất câu trả lời dưới dạng MP3.
        vector_store:       Vector store để retrieve (truyền từ notebook).
        gemini_model:       Gemini GenerativeModel instance.

    Returns:
        dict:
            question      (str)  — câu hỏi đã chuẩn hoá
            input_mode    (str)  — 'text' | 'mic' | 'audio_file' | 'image'
            answer        (str)  — câu trả lời
            audio_response(str|None) — đường dẫn MP3 nếu respond_with_audio
    """
    question_text: str | None = None
    input_mode: str = "text"
    tmp_audio: str | None = None  # file tạm cần xóa

    print("\n" + "─" * 50)

    # ── Bước A: Nhận và chuẩn hoá input ──────────────────────────
    if use_mic:
        input_mode = "mic"
        print("[INPUT] Chế độ: Microphone")
        tmp_audio = record_from_mic(duration=mic_duration)
        question_text = transcribe_audio(tmp_audio, model_size=whisper_model)

    elif audio_path:
        input_mode = "audio_file"
        print(f"[INPUT] Chế độ: Audio file → {audio_path}")
        question_text = transcribe_audio(audio_path, model_size=whisper_model)

    elif image_path:
        input_mode = "image"
        print(f"[INPUT] Chế độ: Ảnh → {image_path}")
        question_text = ocr_image(image_path)

    elif text:
        input_mode = "text"
        print("[INPUT] Chế độ: Text")
        question_text = text.strip()

    else:
        raise ValueError(
            "Phải cung cấp ít nhất một trong:\n"
            "  text=, audio_path=, image_path=, hoặc use_mic=True"
        )

    # Xóa file WAV tạm ngay sau transcribe
    if tmp_audio:
        Path(tmp_audio).unlink(missing_ok=True)

    if not question_text:
        print("  ⚠  Không nhận được câu hỏi (input rỗng hoặc OCR thất bại)")
        return {"question": "", "input_mode": input_mode, "answer": "", "audio_response": None}

    print(f"\n  Câu hỏi: {question_text}")

    # ── Bước B: RAG retrieve + generate ───────────────────────────
    if vector_store is None or gemini_model is None:
        answer = (
            f"[STUB — chưa kết nối vector store / LLM]\n"
            f"Câu hỏi nhận được: {question_text}\n"
            "Hoàn thành Bước 6 rồi truyền vector_store và gemini_model vào."
        )
    else:
        answer = _rag_generate(question_text, vector_store, gemini_model)

    print(f"\n  Câu trả lời:\n{answer}")

    # ── Bước C: TTS output (tùy chọn) ─────────────────────────────
    audio_out: str | None = None
    if respond_with_audio:
        print("\n  Đang tạo audio response...")
        audio_out = text_to_speech(answer)

    print("─" * 50)

    return {
        "question": question_text,
        "input_mode": input_mode,
        "answer": answer,
        "audio_response": audio_out,
    }


# ================================================================ #
#  Interactive CLI loop                                            #
# ================================================================ #

def interactive_loop(
    vector_store=None,
    gemini_model=None,
    whisper_model: str = "base",
):
    """CLI chatbot đa phương thức — chạy từ terminal hoặc notebook cell.

    Args:
        vector_store:  Vector store (từ Bước 6).
        gemini_model:  Gemini model (từ Bước 8).
        whisper_model: Kích thước Whisper model.
    """
    print("\n" + "=" * 55)
    print("  CHATBOT BỘ LUẬT DÂN SỰ 2015 — ĐA PHƯƠNG THỨC")
    print("=" * 55)
    print("  Chọn loại input:")
    print("    t  →  Nhập text")
    print("    m  →  Nói vào mic")
    print("    i  →  Ảnh chụp câu hỏi")
    print("    q  →  Thoát")
    print()

    while True:
        mode = input("Input [t/m/i/q]: ").strip().lower()

        if mode == "q":
            print("Tạm biệt!")
            break

        if mode not in ("t", "m", "i"):
            print("  Hãy chọn t / m / i / q")
            continue

        want_audio = input("Trả lời bằng audio? [y/N]: ").strip().lower() == "y"

        try:
            if mode == "t":
                text = input("Câu hỏi: ").strip()
                if not text:
                    continue
                query(
                    text=text,
                    respond_with_audio=want_audio,
                    vector_store=vector_store,
                    gemini_model=gemini_model,
                    whisper_model=whisper_model,
                )

            elif mode == "m":
                raw = input("Thời gian ghi âm (giây, Enter=5): ").strip()
                duration = int(raw) if raw.isdigit() else 5
                query(
                    use_mic=True,
                    mic_duration=duration,
                    respond_with_audio=want_audio,
                    vector_store=vector_store,
                    gemini_model=gemini_model,
                    whisper_model=whisper_model,
                )

            elif mode == "i":
                img_path = input("Đường dẫn file ảnh: ").strip()
                if not Path(img_path).exists():
                    print(f"  ⚠  Không tìm thấy: {img_path}")
                    continue
                query(
                    image_path=img_path,
                    respond_with_audio=want_audio,
                    vector_store=vector_store,
                    gemini_model=gemini_model,
                    whisper_model=whisper_model,
                )

        except KeyboardInterrupt:
            print("\n  (Interrupted)")

        print()


# ================================================================ #
if __name__ == "__main__":
    # Chạy demo không có vector store (stub mode)
    interactive_loop()
