"""
FastAPI — RAG Chatbot Bộ Luật Dân Sự Việt Nam 2015

Tích hợp: Embedding → Qdrant → Agent (RAG/Web/Calc) → Gemini
Multimodal: Whisper (audio), pytesseract (OCR), gTTS (text-to-speech)

Chạy: uvicorn api:app --reload --port 8000
"""

import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

GEMINI_MODEL = "gemini-3-flash-preview"

app = FastAPI(
    title="RAG Bộ Luật Dân Sự 2015",
    description="API tra cứu Bộ Luật Dân Sự Việt Nam 2015 — Tích hợp RAG + Agent",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================ #
#  Khởi tạo lazy (chỉ load khi request đầu tiên đến)              #
# ================================================================ #
_resources: dict = {}


def get_resources() -> dict:
    """Lazy-load toàn bộ resources: vector store, embedding, Gemini, audio index."""
    if _resources:
        return _resources

    from embedding import embed_texts          # Bước 5
    from vectordb import search_qdrant         # Bước 6
    from qdrant_client import QdrantClient
    from google import genai

    # Vector DB
    qdrant_cl = QdrantClient(path="vector_stores/qdrant", force_disable_check_same_thread=True)

    # Embedding (cached)
    _resources["embed_texts"]   = embed_texts

    # Gemini client
    api_key = os.getenv("GEMINI_API_KEY")
    gemini_client = genai.Client(api_key=api_key)

    _resources["qdrant_cl"]     = qdrant_cl
    _resources["search_qdrant"] = search_qdrant
    _resources["gemini"]        = gemini_client

    # Audio index
    _resources["audio_index"] = _build_audio_index()

    return _resources


# ================================================================ #
#  Audio Index (matching câu hỏi → file audio liên quan)           #
# ================================================================ #

def _build_audio_index() -> list[dict]:
    """Đọc script_question_audio.txt và tạo embedding index cho audio files."""
    import numpy as np
    from embedding import embed_texts

    script_path = Path("data/script_question_audio.txt")
    audio_dir = Path("data/audio")
    if not script_path.exists():
        print("  ⚠ Không tìm thấy script_question_audio.txt")
        return []

    entries = []
    for line in script_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        num, text = line.split(":", 1)
        num = num.strip()
        text = text.strip()
        if not num or not text:
            continue

        candidates = list(audio_dir.glob(f"IMG_{num}*"))
        if not candidates:
            continue

        entries.append({
            "file_num": num,
            "description": text,
            "audio_path": str(candidates[0]),
            "audio_filename": candidates[0].name,
        })

    if not entries:
        return []

    texts = [e["description"] for e in entries]
    embeddings = embed_texts(texts)
    for i, e in enumerate(entries):
        e["embedding"] = embeddings[i]

    print(f"  ✓ Audio index: {len(entries)} files")
    return entries


def _find_relevant_audio(question: str, threshold: float = 0.35) -> dict | None:
    """Tìm file audio liên quan nhất với câu hỏi (cosine similarity)."""
    import numpy as np

    r = get_resources()
    audio_index = r.get("audio_index", [])
    if not audio_index:
        return None

    q_vec = r["embed_texts"]([question])[0]

    best_score = -1
    best_entry = None
    for entry in audio_index:
        a, b = q_vec, entry["embedding"]
        score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_score >= threshold and best_entry:
        return {**best_entry, "score": round(best_score, 4)}
    return None


# ================================================================ #
#  Bước 9: Agent — Router + Tools (RAG / Web / Direct / Calc)      #
# ================================================================ #

def _classify_route(question: str) -> str:
    """Phân loại câu hỏi → 'rag' / 'web' / 'direct' / 'calc'.

    Tương đương router_node trong agent.py (LangGraph workflow):
      START → router → rag_node / web_node / direct_node → END
    """
    q = question.lower()

    # Calculator: có biểu thức tính toán
    calc_keywords = ["tính", "bao nhiêu tiền", "lãi suất", "ngày", "tháng sau", "cộng", "trừ", "nhân", "chia"]
    if any(kw in q for kw in calc_keywords) and any(c.isdigit() for c in q):
        return "calc"

    # RAG: câu hỏi về luật dân sự
    law_keywords = [
        "điều", "luật dân sự", "dân sự", "quyền", "nghĩa vụ",
        "hợp đồng", "tài sản", "thừa kế", "bồi thường",
        "giao dịch", "pháp nhân", "cá nhân", "năng lực hành vi",
        "đại diện", "thời hạn", "thời hiệu", "sở hữu",
        "nhân thân", "tặng cho", "thuê", "mượn", "vay",
        "di chúc", "thế chấp", "cầm cố", "bảo lãnh",
        "bộ luật", "pháp luật", "luật", "chế tài",
    ]
    if any(kw in q for kw in law_keywords):
        return "rag"

    # Web: mặc định cho câu hỏi ngoài phạm vi
    return "web"


def _calc_answer(question: str) -> dict:
    """Calculator tool — tính toán thời hạn, số tiền, lãi suất.

    Tương đương make_calculator_tool() trong agent.py.
    """
    r = get_resources()
    prompt = (
        "Bạn là trợ lý pháp luật. Câu hỏi có liên quan đến tính toán.\n"
        "Hãy tính toán và trả lời bằng tiếng Việt.\n"
        "Nếu liên quan đến luật dân sự, hãy trích dẫn điều khoản.\n\n"
        f"CÂU HỎI: {question}\n\nTRẢ LỜI:"
    )
    response = r["gemini"].models.generate_content(
        model=GEMINI_MODEL, contents=prompt
    )
    return {
        "answer": response.text.strip(),
        "sources": [{"article_number": None, "score": 0, "preview": "Công cụ: Calculator"}],
        "audio": None,
        "route": "calc",
    }


def _web_answer(question: str) -> dict:
    """Web search — tìm trên internet cho câu hỏi ngoài phạm vi.

    Tương đương web_node() trong agent.py (LangGraph workflow)
    + make_web_search_tool() (DuckDuckGo).
    """
    r = get_resources()
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(question, region="vn-vi", max_results=5))
        web_context = "\n\n".join(
            f"- {item['title']}: {item['body']}" for item in results
        ) if results else "(Không tìm được kết quả)"
    except Exception:
        web_context = "(Không thể tìm kiếm web)"

    prompt = (
        "Bạn là trợ lý pháp luật thông minh. "
        "Câu hỏi này NGOÀI phạm vi Bộ Luật Dân Sự 2015, "
        "nên bạn đã tìm kiếm trên internet.\n"
        "Dựa vào kết quả tìm kiếm bên dưới, trả lời bằng tiếng Việt.\n"
        "Ghi rõ nguồn thông tin từ internet.\n\n"
        f"KẾT QUẢ TÌM KIẾM:\n{web_context}\n\n"
        f"CÂU HỎI: {question}\n\nTRẢ LỜI:"
    )
    response = r["gemini"].models.generate_content(
        model=GEMINI_MODEL, contents=prompt
    )
    return {
        "answer": response.text.strip(),
        "sources": [{"article_number": None, "score": 0, "preview": "Nguồn: Internet (DuckDuckGo)"}],
        "audio": None,
        "route": "web",
    }


def _rag_answer(question: str, top_k: int = 5) -> dict:
    """RAG: Retrieve + Generate cho câu hỏi luật dân sự.

    Tương đương rag_node() trong agent.py (LangGraph workflow).
    Kết hợp:
      - Bước 5: embed_texts (embedding)
      - Bước 6: search_qdrant (retrieval)
      - Gemini (generation)
    """
    r = get_resources()
    q_vec = r["embed_texts"]([question])[0]
    results = r["search_qdrant"](q_vec, r["qdrant_cl"], top_k=top_k)

    context = "\n\n---\n\n".join(res["text"] for res in results)
    sources = [
        {
            "article_number": res["metadata"].get("article_number"),
            "score": round(float(res["score"]), 4),
            "preview": res["text"][:200],
        }
        for res in results
    ]

    prompt = (
        "Bạn là trợ lý pháp luật chuyên về Bộ Luật Dân Sự Việt Nam 2015.\n"
        "Trả lời câu hỏi dựa VÀO NGỮ CẢNH bên dưới, bằng tiếng Việt.\n"
        "Nếu không đủ thông tin, hãy nói rõ.\n\n"
        f"NGỮ CẢNH:\n{context}\n\n"
        f"CÂU HỎI: {question}\n\nTRẢ LỜI:"
    )
    response = r["gemini"].models.generate_content(
        model=GEMINI_MODEL, contents=prompt
    )

    # Tìm audio liên quan
    audio_match = _find_relevant_audio(question)
    audio_info = None
    if audio_match:
        audio_info = {
            "filename": audio_match["audio_filename"],
            "description": audio_match["description"],
            "score": audio_match["score"],
            "url": f"/audio/{audio_match['audio_filename']}",
        }

    return {
        "answer": response.text.strip(),
        "sources": sources,
        "audio": audio_info,
        "route": "rag",
    }


def _smart_answer(question: str, top_k: int = 5) -> dict:
    """Agent thông minh: Router → RAG / Web / Calculator.

    Tương đương build_langgraph_workflow() từ agent.py:
      START → router → rag_node / web_node / calc_node → END
    """
    route = _classify_route(question)
    if route == "rag":
        return _rag_answer(question, top_k=top_k)
    elif route == "calc":
        return _calc_answer(question)
    else:  # web
        return _web_answer(question)


# ================================================================ #
#  Schemas                                                         #
# ================================================================ #

class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    audio: dict | None = None
    route: str | None = None  # Bước 9: cho biết Agent chọn route nào


# ================================================================ #
#  Endpoints                                                       #
# ================================================================ #

@app.get("/health")
def health():
    """Kiểm tra server đang chạy + trạng thái các module."""
    r = get_resources() if _resources else {}
    return {
        "status": "ok",
        "service": "RAG Bộ Luật Dân Sự 2015",
        "modules": {
            "qdrant": "qdrant_cl" in r,
            "embedding": "embed_texts" in r,
            "gemini": "gemini" in r,
            "audio_index": len(r.get("audio_index", [])),
        },
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Smart Agent query — tự phân loại câu hỏi → RAG / Web / Calculator.

    Tích hợp: Bước 5 (embed) + 6 (Qdrant) + 9 (Agent routing).

    Example:
        curl -X POST http://localhost:8000/ask \\
             -H "Content-Type: application/json" \\
             -d '{"question": "Quyền dân sự là gì?", "top_k": 5}'
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question không được rỗng")
    try:
        result = _smart_answer(req.question, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return AskResponse(question=req.question, **result)


@app.post("/ask/multimodal", response_model=AskResponse)
async def ask_multimodal(
    text: str = Form(default=""),
    audio: UploadFile = File(default=None),
    image: UploadFile = File(default=None),
    top_k: int = Form(default=5),
):
    """Smart Agent query với multi-modal input (text / audio / image).

    Tích hợp:
      - Audio input: Whisper medium (Bước 8 multimodal)
      - Image input: pytesseract OCR (Bước 8 multimodal)
      - Routing: Agent (Bước 9) → RAG / Web / Calculator

    Example (curl):
        curl -X POST http://localhost:8000/ask/multimodal \\
             -F "text=Quyền dân sự là gì?"

        curl -X POST http://localhost:8000/ask/multimodal \\
             -F "audio=@cauhoi.wav"

        curl -X POST http://localhost:8000/ask/multimodal \\
             -F "image=@cauhoi.jpg"
    """
    question = ""
    tmp_files = []

    try:
        if audio and audio.filename:
            # Whisper transcribe (medium cho tiếng Việt chính xác hơn)
            suffix = Path(audio.filename).suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(await audio.read())
                tmp_path = f.name
            tmp_files.append(tmp_path)

            import whisper
            model = whisper.load_model("medium")
            result = model.transcribe(
                tmp_path,
                language="vi",
                initial_prompt="Đây là câu hỏi pháp luật về Bộ Luật Dân Sự Việt Nam 2015. "
                               "Các từ khóa thường gặp: quyền dân sự, hợp đồng, thừa kế, "
                               "tài sản, nghĩa vụ, bồi thường, giao dịch dân sự, pháp nhân.",
            )
            question = result["text"].strip()

        elif image and image.filename:
            # pytesseract OCR
            suffix = Path(image.filename).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(await image.read())
                tmp_path = f.name
            tmp_files.append(tmp_path)

            import pytesseract
            from PIL import Image as PILImage
            img = PILImage.open(tmp_path)
            question = pytesseract.image_to_string(img, lang="vie").strip()

        elif text:
            question = text.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Phải cung cấp text, audio, hoặc image"
            )

        result = _smart_answer(question, top_k=top_k)
        return AskResponse(question=question, **result)

    finally:
        for p in tmp_files:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass


@app.get("/audio/{filename}")
def serve_audio(filename: str):
    """Phục vụ file audio từ data/audio/."""
    audio_path = Path("data/audio") / filename
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy file audio")
    return FileResponse(str(audio_path), media_type="audio/mpeg", filename=filename)


@app.post("/tts")
async def text_to_speech(req: AskRequest):
    """Chuyển văn bản thành giọng nói tiếng Việt (gTTS).

    Example:
        curl -X POST http://localhost:8000/tts \\
             -H "Content-Type: application/json" \\
             -d '{"question": "Xin chào"}' --output output.mp3
    """
    text = req.question.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Không có văn bản để đọc")

    try:
        from gtts import gTTS
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tts = gTTS(text=text, lang="vi")
            tts.save(f.name)
            return FileResponse(
                f.name,
                media_type="audio/mpeg",
                filename="response.mp3",
                background=None,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS lỗi: {e}")


# ================================================================ #
#  Chạy trực tiếp (dev mode)                                      #
# ================================================================ #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
