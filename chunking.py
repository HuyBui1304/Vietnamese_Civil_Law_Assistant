"""
BƯỚC 4: Chunking văn bản  (Bài 0 + Bài 5)
==========================================
2 phương pháp để sau so sánh hiệu quả:
  - Fixed-size: 500 ký tự, overlap 100 (baseline)
  - Theo Điều luật: mỗi Điều = 1 chunk (semantic)
"""

import json
import re
from pathlib import Path

CHUNKS_DIR = Path("data/chunks")
OCR_TEXT   = Path("data/extracted_text/full_text.txt")
TRANSCRIPTS = Path("data/transcripts/transcripts.json")


# ── Phương pháp 1: Fixed-size ──────────────────────────────────────
def chunk_fixed(text: str, size: int = 500, overlap: int = 100,
                source: str = "data1.pdf") -> list[dict]:
    chunks = []
    i = 0
    idx = 0
    while i < len(text):
        chunk_text = text[i:i + size].strip()
        if chunk_text:
            chunks.append({
                "id": f"fixed_{idx}",
                "text": chunk_text,
                "source": source,
                "type": "pdf",
                "method": "fixed",
            })
            idx += 1
        i += size - overlap
    return chunks


# ── Phương pháp 2: Theo Điều luật ─────────────────────────────────
def chunk_by_article(text: str, source: str = "data1.pdf") -> list[dict]:
    # Tìm tất cả vị trí "Điều X."
    pattern = re.compile(r'(Điều\s+\d+\.)', re.UNICODE)
    splits = list(pattern.finditer(text))

    chunks = []
    for i, match in enumerate(splits):
        start = match.start()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        chunk_text = text[start:end].strip()

        article_num = re.search(r'\d+', match.group())
        article_num = int(article_num.group()) if article_num else None

        if chunk_text:
            chunks.append({
                "id": f"article_{article_num}_{i}",
                "text": chunk_text,
                "source": source,
                "type": "pdf",
                "method": "article",
                "article_number": article_num,
            })
    return chunks


# ── Chunking audio transcripts ─────────────────────────────────────
def chunk_audio(transcripts: list[dict], size: int = 500, overlap: int = 100) -> list[dict]:
    chunks = []
    for t in transcripts:
        text = t["text"]
        source = t["source"]
        i = 0
        idx = 0
        while i < len(text):
            chunk_text = text[i:i + size].strip()
            if chunk_text:
                chunks.append({
                    "id": f"audio_{source}_{idx}",
                    "text": chunk_text,
                    "source": source,
                    "type": "audio",
                    "method": "fixed",
                })
                idx += 1
            i += size - overlap
    return chunks


# ── Run ────────────────────────────────────────────────────────────
def run_chunking(full_text: str | None = None,
              transcripts: list[dict] | None = None,
              force_rerun: bool = False):
    """
    Returns:
        chunks_fixed   (list[dict]) — fixed-size chunks (PDF + audio)
        chunks_article (list[dict]) — article-based chunks (PDF) + fixed audio
    """
    fixed_path   = CHUNKS_DIR / "chunks_fixed.json"
    article_path = CHUNKS_DIR / "chunks_article.json"

    if fixed_path.exists() and article_path.exists() and not force_rerun:
        print("  ○ Đã có chunks. Dùng load_chunks() hoặc set force_rerun=True")
        return load_chunks()

    # Load text nếu chưa truyền vào
    if full_text is None:
        full_text = Path(OCR_TEXT).read_text(encoding="utf-8")
    if transcripts is None and TRANSCRIPTS.exists():
        transcripts = json.loads(TRANSCRIPTS.read_text(encoding="utf-8"))
    if transcripts is None:
        transcripts = []

    print(f"\n{'='*55}")
    print(f"  BƯỚC 4: CHUNKING VĂN BẢN")
    print(f"{'='*55}")

    # PDF chunks
    c_fixed_pdf   = chunk_fixed(full_text)
    c_article_pdf = chunk_by_article(full_text)

    # Audio chunks
    c_audio = chunk_audio(transcripts) if transcripts else []

    # Gộp
    chunks_fixed   = c_fixed_pdf + c_audio
    chunks_article = c_article_pdf + c_audio

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    fixed_path.write_text(
        json.dumps(chunks_fixed, ensure_ascii=False, indent=2), encoding="utf-8")
    article_path.write_text(
        json.dumps(chunks_article, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  Fixed-size chunks  : {len(chunks_fixed):,}  (PDF: {len(c_fixed_pdf)}, audio: {len(c_audio)})")
    print(f"  Theo Điều luật     : {len(chunks_article):,}  (PDF: {len(c_article_pdf)}, audio: {len(c_audio)})")
    print(f"  ✓ Lưu: {fixed_path}")
    print(f"  ✓ Lưu: {article_path}\n")

    return chunks_fixed, chunks_article


def load_chunks():
    fixed   = json.loads((CHUNKS_DIR / "chunks_fixed.json").read_text(encoding="utf-8"))
    article = json.loads((CHUNKS_DIR / "chunks_article.json").read_text(encoding="utf-8"))
    return fixed, article
