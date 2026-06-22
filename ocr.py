"""
BƯỚC 2: OCR trích xuất text từ PDF scan  (Bài 0 + Bài 2)
==========================================================
File PDF Bộ Luật Dân Sự là dạng SCAN (ảnh) → không extract text trực tiếp.
Quy trình:
  1. pdf2image  →  convert từng trang PDF thành ảnh PIL
  2. pytesseract →  OCR từng ảnh với ngôn ngữ tiếng Việt ('vie')
  3. Ghép toàn bộ text, lưu ra data/extracted_text/full_text.txt
  4. Lưu metadata từng trang ra data/extracted_text/ocr_metadata.json

Tại sao cover Bài 2 (Multi-modal RAG):
  PDF scan  =  dữ liệu IMAGE  →  OCR chính là image processing (modality 1)
  Bước 3 sẽ thêm AUDIO processing (modality 2) để hoàn thiện multi-modal.

Yêu cầu hệ thống:
  - Tesseract OCR đã cài (https://github.com/UB-Mannheim/tesseract/wiki)
  - Gói ngôn ngữ tiếng Việt của Tesseract (tessdata/vie.traineddata)
  - Poppler (cần cho pdf2image trên Windows/Mac):
      macOS:   brew install poppler
      Ubuntu:  sudo apt install poppler-utils
"""

import json
import time
from pathlib import Path

try:
    from pdf2image import convert_from_path
    import pytesseract
    from tqdm import tqdm
except ImportError as e:
    raise ImportError(
        f"Thiếu thư viện: {e}\n"
        "Chạy: pip install pdf2image pytesseract tqdm"
    ) from e

# ------------------------------------------------------------------ #
# Đường dẫn mặc định
# ------------------------------------------------------------------ #
OUTPUT_DIR = Path("data/extracted_text")
OUTPUT_TEXT = OUTPUT_DIR / "full_text.txt"
OUTPUT_META = OUTPUT_DIR / "ocr_metadata.json"


# ------------------------------------------------------------------ #
# Hàm chính
# ------------------------------------------------------------------ #
def ocr_pdf(
    pdf_path: str | Path,
    lang: str = "vie",
    dpi: int = 300,
    output_text: str | Path = OUTPUT_TEXT,
    output_meta: str | Path = OUTPUT_META,
    force_rerun: bool = False,
) -> str:
    """OCR toàn bộ file PDF scan, trả về full text.

    Args:
        pdf_path:    Đường dẫn file PDF.
        lang:        Ngôn ngữ Tesseract ('vie' = tiếng Việt).
        dpi:         Độ phân giải khi render PDF → ảnh (cao hơn = chính xác
                     hơn nhưng chậm hơn; 300 là khuyến nghị).
        output_text: File lưu toàn bộ text OCR.
        output_meta: File JSON lưu metadata từng trang.
        force_rerun: Nếu True, chạy lại dù đã có file output.

    Returns:
        Full text đã OCR (string).
    """
    pdf_path = Path(pdf_path)
    output_text = Path(output_text)
    output_meta = Path(output_meta)

    # ── Kiểm tra đã OCR chưa ────────────────────────────────────────
    if output_text.exists() and not force_rerun:
        size_kb = output_text.stat().st_size / 1024
        print(f"  ○ Đã có kết quả OCR: {output_text}  ({size_kb:.0f} KB)")
        print("    Dùng load_extracted_text() để đọc, hoặc set force_rerun=True")
        return load_extracted_text(output_text)

    # ── Kiểm tra file PDF ────────────────────────────────────────────
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file PDF: {pdf_path}\n"
            "Đặt file vào data/pdf/ rồi chạy lại."
        )

    size_mb = pdf_path.stat().st_size / 1024 / 1024
    print(f"\n{'='*55}")
    print(f"  BƯỚC 2: OCR TRÍCH XUẤT TEXT TỪ PDF SCAN")
    print(f"{'='*55}")
    print(f"  File   : {pdf_path.name}  ({size_mb:.1f} MB)")
    print(f"  Lang   : {lang}  |  DPI: {dpi}")
    print()

    # ── Bước 2a: Convert PDF → ảnh ──────────────────────────────────
    print("  [2a] Convert PDF → ảnh (có thể mất vài phút)...")
    t0 = time.time()
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    total_pages = len(pages)
    elapsed = time.time() - t0
    print(f"       Xong! {total_pages} trang  ({elapsed:.1f}s)\n")

    # ── Bước 2b: OCR từng trang ──────────────────────────────────────
    print("  [2b] OCR từng trang bằng Tesseract...")
    page_records = []
    t1 = time.time()

    for i, page_img in enumerate(tqdm(pages, desc="  OCR", unit="trang")):
        print(f"       Đang xử lý trang {i + 1}/{total_pages}...", end="\r")
        text = pytesseract.image_to_string(page_img, lang=lang).strip()
        page_records.append({
            "page": i + 1,
            "text": text,
            "char_count": len(text),
        })

    print()  # newline sau \r
    elapsed_ocr = time.time() - t1
    print(f"       Xong! ({elapsed_ocr:.1f}s)\n")

    # ── Bước 2c: Ghép text ──────────────────────────────────────────
    separator = "\n" + "=" * 60 + "\n"
    full_text = separator.join(
        f"TRANG {r['page']}/{total_pages}\n{separator.strip()}\n{r['text']}"
        for r in page_records
    )

    # ── Bước 2d: Lưu kết quả ────────────────────────────────────────
    output_text.parent.mkdir(parents=True, exist_ok=True)
    output_text.write_text(full_text, encoding="utf-8")

    metadata = {
        "source_file": str(pdf_path),
        "total_pages": total_pages,
        "lang": lang,
        "dpi": dpi,
        "total_chars": len(full_text),
        "ocr_time_seconds": round(elapsed_ocr, 1),
        "pages": page_records,
    }
    output_meta.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── Tóm tắt ─────────────────────────────────────────────────────
    total_chars = sum(r["char_count"] for r in page_records)
    avg_per_page = total_chars // total_pages if total_pages else 0
    empty_pages = sum(1 for r in page_records if r["char_count"] < 50)

    print(f"  {'='*51}")
    print(f"  KẾT QUẢ OCR")
    print(f"  {'='*51}")
    print(f"  ✓ Tổng trang     : {total_pages}")
    print(f"  ✓ Tổng ký tự     : {total_chars:,}")
    print(f"  ✓ Trung bình/trang: {avg_per_page:,} ký tự")
    if empty_pages:
        print(f"  ⚠  Trang ít text (<50 ký tự): {empty_pages} trang")
    print(f"  ✓ Lưu text : {output_text}")
    print(f"  ✓ Lưu meta : {output_meta}")
    print()

    return full_text


# ------------------------------------------------------------------ #
# Tiện ích
# ------------------------------------------------------------------ #
def load_extracted_text(path: str | Path = OUTPUT_TEXT) -> str:
    """Load text đã OCR từ file (để tránh phải OCR lại)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có file {path}.\n"
            "Hãy chạy ocr_pdf() trước."
        )
    return path.read_text(encoding="utf-8")


def preview_ocr_result(n_chars: int = 800, path: str | Path = OUTPUT_TEXT):
    """In preview kết quả OCR ra màn hình."""
    text = load_extracted_text(path)
    print(f"Tổng ký tự: {len(text):,}")
    print(f"\n{'='*55}")
    print(f"PREVIEW ({n_chars} ký tự đầu):")
    print("=" * 55)
    print(text[:n_chars])
    print("...")


def ocr_image_query(image_path: str | Path, lang: str = "vie") -> str:
    """OCR một ảnh chụp đơn lẻ tại QUERY TIME (không phải ingestion).

    Dùng cho input ảnh chụp câu hỏi viết tay / đoạn văn bản in.
    Tái sử dụng pytesseract từ Bước 2, không cần pdf2image.

    Args:
        image_path: Đường dẫn ảnh (jpg, png, ...).
        lang:       Ngôn ngữ Tesseract (mặc định 'vie').

    Returns:
        Text trích xuất từ ảnh.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("pip install Pillow")

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=lang).strip()

    if not text:
        print(f"  ⚠ OCR không nhận được text từ {image_path.name}")
        print("    Thử tăng độ phân giải ảnh hoặc kiểm tra gói ngôn ngữ 'vie'")
    else:
        print(f"  ✓ OCR ảnh xong: {len(text)} ký tự")

    return text


def run_ocr(force_rerun: bool = False) -> str:
    """Chạy toàn bộ Bước 2: tự động tìm PDF trong data/pdf/ và OCR.

    Args:
        force_rerun: Nếu True, OCR lại dù đã có file output.

    Returns:
        Full text OCR.
    """
    pdf_files = sorted(Path("data/pdf").glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            "Không tìm thấy file PDF trong data/pdf/\n"
            "Đặt file Bộ Luật Dân Sự vào thư mục đó rồi chạy lại."
        )

    if len(pdf_files) > 1:
        print(f"  Tìm thấy {len(pdf_files)} file PDF, dùng file đầu tiên:")
        for f in pdf_files:
            print(f"    - {f.name}")
        print()

    return ocr_pdf(pdf_files[0], force_rerun=force_rerun)


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    text = run_ocr()
    preview_ocr_result()
