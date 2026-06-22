"""
BƯỚC 1: Chuẩn bị môi trường và dữ liệu
========================================
- Tạo toàn bộ cấu trúc thư mục project
- Kiểm tra file .env và GEMINI_API_KEY
- Kiểm tra dữ liệu PDF + audio đã có chưa
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Tất cả thư mục cần tạo cho toàn bộ 13 bước
PROJECT_DIRS = [
    "data/pdf",                # PDF Bộ Luật Dân Sự (input)
    "data/audio",              # Audio giải thích luật (input)
    "data/extracted_text",     # Kết quả OCR (bước 2)
    "data/transcripts",        # Kết quả Whisper (bước 3)
    "data/chunks",             # Chunks sau khi chunking (bước 4)
    "data/embeddings",         # Embeddings (bước 5)
    "data/dataset",            # Dataset Q&A cho fine-tuning (bước 10)
    "vector_stores/faiss",     # FAISS index (bước 6a)
    "vector_stores/chroma",    # Chroma persistent (bước 6b)
    "vector_stores/qdrant",    # Qdrant snapshot (bước 6c)
    "outputs/benchmark",       # Kết quả benchmark (bước 7)
    "outputs/evaluation",      # Kết quả evaluation (bước 11)
    "models/lora_adapter",     # LoRA adapter sau fine-tune (bước 10)
]


def create_project_structure():
    """Tạo toàn bộ cấu trúc thư mục cần thiết cho project."""
    print("=" * 55)
    print("  TẠO CẤU TRÚC THƯ MỤC PROJECT")
    print("=" * 55)
    created = 0
    for d in PROJECT_DIRS:
        path = Path(d)
        already_exists = path.exists()
        path.mkdir(parents=True, exist_ok=True)
        if already_exists:
            print(f"  ○ Đã có:   {d}/")
        else:
            print(f"  ✓ Tạo mới: {d}/")
            created += 1
    print(f"\nKết quả: {created} thư mục mới, {len(PROJECT_DIRS) - created} đã tồn tại")
    print()


def check_env():
    """Kiểm tra file .env và GEMINI_API_KEY.

    Returns:
        bool: True nếu API key hợp lệ
    """
    print("=" * 55)
    print("  KIỂM TRA BIẾN MÔI TRƯỜNG (.env)")
    print("=" * 55)

    env_file = Path(".env")
    if not env_file.exists():
        print("  ⚠  Chưa có file .env")
        print("  →  Chạy lệnh sau rồi điền API key:")
        print("     cp .env.example .env")
        print()
        return False

    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY", "")

    if not api_key:
        print("  ⚠  GEMINI_API_KEY chưa được điền trong .env")
        print("  →  Lấy key tại: https://aistudio.google.com/app/apikey")
        print()
        return False

    masked = api_key[:8] + "..." + api_key[-4:]
    print(f"  ✓ GEMINI_API_KEY: {masked}")
    print()
    return True


def check_data_files():
    """Kiểm tra dữ liệu PDF và audio đã có chưa.

    Returns:
        tuple[bool, bool]: (có PDF, có audio)
    """
    print("=" * 55)
    print("  KIỂM TRA DỮ LIỆU ĐẦU VÀO")
    print("=" * 55)

    # PDF files
    pdf_files = sorted(Path("data/pdf").glob("*.pdf"))
    print(f"  PDF files ({len(pdf_files)} file):")
    if pdf_files:
        for f in pdf_files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"    ✓ {f.name}  ({size_mb:.1f} MB)")
    else:
        print("    ⚠  Chưa có! Đặt file PDF Bộ Luật Dân Sự vào data/pdf/")

    # Audio files
    audio_extensions = ["*.mp3", "*.wav", "*.m4a", "*.ogg", "*.flac"]
    audio_files = sorted(
        f for ext in audio_extensions for f in Path("data/audio").glob(ext)
    )
    print(f"\n  Audio files ({len(audio_files)} file):")
    if audio_files:
        for f in audio_files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"    ✓ {f.name}  ({size_mb:.1f} MB)")
        if len(audio_files) < 3:
            print("    ⚠  Nên có 3-5 file audio (hiện chỉ có "
                  f"{len(audio_files)})")
    else:
        print("    ⚠  Chưa có! Đặt 3-5 file mp3/wav vào data/audio/")

    print()
    return len(pdf_files) > 0, len(audio_files) > 0


def run_setup():
    """Chạy toàn bộ Bước 1."""
    print("\n" + "=" * 55)
    print("  BƯỚC 1: CHUẨN BỊ MÔI TRƯỜNG VÀ DỮ LIỆU")
    print("=" * 55 + "\n")

    create_project_structure()
    env_ok = check_env()
    has_pdf, has_audio = check_data_files()

    # Tóm tắt
    print("=" * 55)
    print("  TÓM TẮT BƯỚC 1")
    print("=" * 55)
    print(f"  Thư mục project: ✓ Sẵn sàng")
    print(f"  .env / API key:  {'✓ OK' if env_ok else '⚠  Cần cấu hình'}")
    print(f"  PDF data:        {'✓ OK' if has_pdf else '⚠  Cần thêm file'}")
    print(f"  Audio data:      {'✓ OK' if has_audio else '⚠  Cần thêm file'}")

    if env_ok and has_pdf:
        print("\n  → Sẵn sàng chạy Bước 2 (OCR)!")
    else:
        print("\n  → Hãy hoàn thành các mục ⚠ trước khi tiếp tục.")
    print()

    return {"env_ok": env_ok, "has_pdf": has_pdf, "has_audio": has_audio}


if __name__ == "__main__":
    run_setup()
