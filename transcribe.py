"""
BƯỚC 3: Transcribe audio bằng Whisper  (Bài 2 — audio modality)
=================================================================
Transcribe tất cả file audio trong data/audio/ → text tiếng Việt
Lưu kết quả ra data/transcripts/transcripts.json
"""

import json
from pathlib import Path

try:
    import whisper
    from tqdm import tqdm
except ImportError:
    raise ImportError("pip install openai-whisper tqdm")

AUDIO_DIR = Path("data/audio")
OUTPUT_FILE = Path("data/transcripts/transcripts.json")
SUPPORTED = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4"}


def transcribe_all(model_size: str = "base", force_rerun: bool = False) -> list[dict]:
    """Transcribe tất cả file audio trong data/audio/.

    Args:
        model_size:  Kích thước Whisper model ('tiny'/'base'/'small').
                     'base' đủ tốt cho tiếng Việt, 'small' chính xác hơn.
        force_rerun: Nếu True, transcribe lại dù đã có kết quả.

    Returns:
        List[dict] — mỗi phần tử gồm: source, type, text, duration_s
    """
    # Kiểm tra đã có kết quả chưa
    if OUTPUT_FILE.exists() and not force_rerun:
        print(f"  ○ Đã có kết quả: {OUTPUT_FILE}")
        print("    Dùng load_transcripts() để đọc, hoặc set force_rerun=True")
        return load_transcripts()

    audio_files = sorted(f for f in AUDIO_DIR.iterdir() if f.suffix.lower() in SUPPORTED)

    if not audio_files:
        print(f"  ⚠  Không tìm thấy file audio trong {AUDIO_DIR}/")
        print(f"     Hỗ trợ: {', '.join(SUPPORTED)}")
        return []

    print(f"\n{'='*55}")
    print(f"  BƯỚC 3: TRANSCRIBE AUDIO BẰNG WHISPER")
    print(f"{'='*55}")
    print(f"  Model  : {model_size}")
    print(f"  Files  : {len(audio_files)}")
    for f in audio_files:
        print(f"    - {f.name}  ({f.stat().st_size / 1024 / 1024:.1f} MB)")
    print()
    print("  Đang load Whisper model (lần đầu sẽ tải về)...")

    model = whisper.load_model(model_size)
    print(f"  ✓ Model '{model_size}' sẵn sàng\n")

    results = []
    for audio_file in tqdm(audio_files, desc="  Transcribe", unit="file"):
        print(f"  Đang xử lý: {audio_file.name}...")
        result = model.transcribe(str(audio_file), language="vi")
        text = result["text"].strip()
        duration = result.get("segments", [{}])[-1].get("end", 0) if result.get("segments") else 0

        entry = {
            "source": audio_file.name,
            "type": "audio",
            "text": text,
            "duration_s": round(duration, 1),
            "char_count": len(text),
        }
        results.append(entry)
        print(f"    ✓ {len(text):,} ký tự  ({duration:.0f}s)")

    # Lưu kết quả
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total_chars = sum(r["char_count"] for r in results)
    print(f"\n  {'='*51}")
    print(f"  KẾT QUẢ TRANSCRIBE")
    print(f"  {'='*51}")
    print(f"  ✓ {len(results)} file transcribed")
    print(f"  ✓ Tổng ký tự: {total_chars:,}")
    print(f"  ✓ Lưu: {OUTPUT_FILE}")
    print()

    return results


def load_transcripts(path: str | Path = OUTPUT_FILE) -> list[dict]:
    """Load transcripts đã lưu từ file JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Chưa có {path}. Chạy transcribe_all() trước.")
    return json.loads(path.read_text(encoding="utf-8"))


def run_transcribe(model_size: str = "base", force_rerun: bool = False) -> list[dict]:
    return transcribe_all(model_size=model_size, force_rerun=force_rerun)


if __name__ == "__main__":
    run_transcribe()
