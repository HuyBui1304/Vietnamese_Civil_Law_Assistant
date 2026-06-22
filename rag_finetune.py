"""
FastAPI — RAG Chatbot Bộ Luật Dân Sự Việt Nam 2015 (Fine-tuned Model)

Thay thế hoàn toàn Gemini API bằng SeaLLMs-v3-7B-Chat + LoRA adapter.
Giữ nguyên: Embedding (SentenceTransformer), Qdrant, Agent routing.

Chạy: uvicorn rag_finetune:app --reload --port 8000
"""

import os
import tempfile
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ================================================================ #
#  Config                                                          #
# ================================================================ #

BASE_MODEL_ID = "SeaLLMs/SeaLLMs-v3-7B-Chat"
LORA_ADAPTER_PATH = "models/lora_adapter"
MAX_NEW_TOKENS = 1024

app = FastAPI(
    title="RAG Bộ Luật Dân Sự 2015 (Fine-tuned)",
    description="API tra cứu Bộ Luật Dân Sự Việt Nam 2015 — SeaLLMs + LoRA, không dùng Gemini",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================ #
#  Load Fine-tuned Model                                           #
# ================================================================ #

_model_cache: dict = {}


def _detect_device() -> tuple[str, str | None]:
    """Trả về (device_str, torch_dtype_str).

    - CUDA  : device_map="auto", float16  (có thể dùng 4-bit)
    - MPS   : device_map=None,   float16  (Mac Apple Silicon)
    - CPU   : device_map=None,   float32
    """
    if torch.cuda.is_available():
        return "cuda", "auto"
    if torch.backends.mps.is_available():
        return "mps", None      # device_map không hỗ trợ MPS
    return "cpu", None


def get_model_and_tokenizer():
    """Load base model + LoRA adapter, merge và cache.

    - CUDA + bitsandbytes : 4-bit quantization, device_map="auto"
    - CUDA không bnb      : float16, device_map="auto"
    - MPS (Mac)           : float16, load thẳng lên mps (không disk offload)
    - CPU                 : float32

    Tránh disk-offload (meta device) để LoRA weights được load đúng.
    """
    if _model_cache:
        return _model_cache["model"], _model_cache["tokenizer"]

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    hw, _ = _detect_device()

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- CUDA: ưu tiên 4-bit để tiết kiệm VRAM ---
    if hw == "cuda":
        use_4bit = False
        try:
            import bitsandbytes  # noqa: F401
            use_4bit = True
        except ImportError:
            pass

        if use_4bit:
            print("  Loading base model (CUDA, 4-bit NF4)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            print("  Loading base model (CUDA, float16)...")
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

    # --- MPS (Mac Apple Silicon): load thẳng, không disk offload ---
    elif hw == "mps":
        use_4bit = False
        print("  Loading base model (MPS / Apple Silicon, float16)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        base_model = base_model.to("mps")

    # --- CPU fallback ---
    else:
        use_4bit = False
        print("  Loading base model (CPU, float32) — may be slow...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    # Load LoRA adapter — model đã fully loaded, không còn meta tensors
    lora_path = Path(LORA_ADAPTER_PATH)
    if lora_path.exists() and (lora_path / "adapter_config.json").exists():
        print(f"  Loading LoRA adapter from {LORA_ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(
            base_model,
            LORA_ADAPTER_PATH,
            is_trainable=False,
        )
        # Merge weights (không được merge khi dùng 4-bit)
        if not use_4bit:
            print("  Merging LoRA weights into base model...")
            model = model.merge_and_unload()
    else:
        print(f"  LoRA adapter not found at {LORA_ADAPTER_PATH}, using base model only.")
        model = base_model

    model.eval()
    print(f"  Model loaded on: {model.device if hasattr(model, 'device') else 'auto'}")

    _model_cache["model"] = model
    _model_cache["tokenizer"] = tokenizer
    return model, tokenizer


def generate_response(prompt: str) -> str:
    """Generate text từ fine-tuned model với prompt đã format."""
    model, tokenizer = get_model_and_tokenizer()

    # SeaLLMs chat format
    messages = [
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode chỉ phần generated (bỏ prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer


# ================================================================ #
#  Lazy-load Resources (Qdrant, Embedding, Audio Index)            #
# ================================================================ #

_resources: dict = {}


def get_resources() -> dict:
    """Lazy-load: vector store, embedding, audio index."""
    if _resources:
        return _resources

    from embedding import embed_texts
    from qdrant_client import QdrantClient
    from vectordb import search_qdrant

    qdrant_cl = QdrantClient(path="vector_stores/qdrant", force_disable_check_same_thread=True)

    _resources["embed_texts"] = embed_texts
    _resources["qdrant_cl"] = qdrant_cl
    _resources["search_qdrant"] = search_qdrant
    _resources["audio_index"] = _build_audio_index()

    return _resources


# ================================================================ #
#  Audio Index                                                     #
# ================================================================ #

def _build_audio_index() -> list[dict]:
    """Đọc script_question_audio.txt và tạo embedding index cho audio files."""
    import numpy as np
    from embedding import embed_texts

    script_path = Path("data/script_question_audio.txt")
    audio_dir = Path("data/audio")
    if not script_path.exists():
        print("  Khong tim thay script_question_audio.txt")
        return []

    entries = []
    for line in script_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        num, text = line.split(":", 1)
        num, text = num.strip(), text.strip()
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

    print(f"  Audio index: {len(entries)} files")
    return entries


def _find_relevant_audio(question: str, threshold: float = 0.35) -> dict | None:
    """Tìm file audio liên quan nhất (cosine similarity)."""
    import numpy as np

    r = get_resources()
    audio_index = r.get("audio_index", [])
    if not audio_index:
        return None

    q_vec = r["embed_texts"]([question])[0]
    best_score, best_entry = -1, None
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
#  Agent Router + Tools (RAG / Web / Calc) — No Gemini             #
# ================================================================ #

def _classify_route(question: str) -> str:
    """Phân loại câu hỏi -> 'rag' / 'web' / 'calc'."""
    q = question.lower()

    calc_keywords = [
        "tính", "bao nhiêu tiền", "lãi suất", "ngày", "tháng sau",
        "cộng", "trừ", "nhân", "chia",
    ]
    if any(kw in q for kw in calc_keywords) and any(c.isdigit() for c in q):
        return "calc"

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

    return "web"


def _calc_answer(question: str) -> dict:
    """Calculator — dùng fine-tuned model thay Gemini."""
    prompt = (
        "Bạn là trợ lý pháp luật. Câu hỏi có liên quan đến tính toán.\n"
        "Hãy tính toán và trả lời bằng tiếng Việt.\n"
        "Nếu liên quan đến luật dân sự, hãy trích dẫn điều khoản.\n\n"
        f"CÂU HỎI: {question}\n\nTRẢ LỜI:"
    )
    answer = generate_response(prompt)
    return {
        "answer": answer,
        "sources": [{"article_number": None, "score": 0, "preview": "Công cụ: Calculator"}],
        "audio": None,
        "route": "calc",
    }


def _web_answer(question: str) -> dict:
    """Web search — DuckDuckGo + fine-tuned model tổng hợp."""
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
    answer = generate_response(prompt)
    return {
        "answer": answer,
        "sources": [{"article_number": None, "score": 0, "preview": "Nguồn: Internet (DuckDuckGo)"}],
        "audio": None,
        "route": "web",
    }


def _rag_answer(question: str, top_k: int = 5) -> dict:
    """RAG: Retrieve (Qdrant) + Generate (fine-tuned model)."""
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
        "CHỈ trả lời dựa trên NGỮ CẢNH được cung cấp bên dưới.\n"
        "Nếu ngữ cảnh không chứa đủ thông tin, hãy nói rõ 'Không đủ thông tin trong ngữ cảnh'.\n"
        "KHÔNG được tự bịa thêm thông tin ngoài ngữ cảnh.\n"
        "Trả lời bằng tiếng Việt, trích dẫn số điều khoản nếu có.\n\n"
        f"NGỮ CẢNH:\n{context}\n\n"
        f"CÂU HỎI: {question}\n\nTRẢ LỜI:"
    )
    answer = generate_response(prompt)

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
        "answer": answer,
        "sources": sources,
        "audio": audio_info,
        "route": "rag",
    }


def _smart_answer(question: str, top_k: int = 5) -> dict:
    """Agent: Router -> RAG / Web / Calculator."""
    route = _classify_route(question)
    if route == "rag":
        return _rag_answer(question, top_k=top_k)
    elif route == "calc":
        return _calc_answer(question)
    else:
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
    route: str | None = None


# ================================================================ #
#  Endpoints                                                       #
# ================================================================ #

@app.get("/health")
def health():
    """Kiểm tra server + trạng thái modules."""
    r = get_resources() if _resources else {}
    model_loaded = bool(_model_cache)
    return {
        "status": "ok",
        "service": "RAG Bộ Luật Dân Sự 2015 (Fine-tuned)",
        "model": BASE_MODEL_ID,
        "lora_adapter": LORA_ADAPTER_PATH,
        "modules": {
            "qdrant": "qdrant_cl" in r,
            "embedding": "embed_texts" in r,
            "fine_tuned_model": model_loaded,
            "audio_index": len(r.get("audio_index", [])),
        },
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Smart Agent query — RAG / Web / Calculator (fine-tuned model).

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
    """Multimodal input (text / audio / image) -> Agent -> fine-tuned model.

    Example:
        curl -X POST http://localhost:8000/ask/multimodal -F "text=Quyền dân sự là gì?"
        curl -X POST http://localhost:8000/ask/multimodal -F "audio=@cauhoi.wav"
        curl -X POST http://localhost:8000/ask/multimodal -F "image=@cauhoi.jpg"
    """
    question = ""
    tmp_files = []

    try:
        if audio and audio.filename:
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
                initial_prompt=(
                    "Đây là câu hỏi pháp luật về Bộ Luật Dân Sự Việt Nam 2015. "
                    "Các từ khóa thường gặp: quyền dân sự, hợp đồng, thừa kế, "
                    "tài sản, nghĩa vụ, bồi thường, giao dịch dân sự, pháp nhân."
                ),
            )
            question = result["text"].strip()

        elif image and image.filename:
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
                detail="Phải cung cấp text, audio, hoặc image",
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
    """Text-to-speech tiếng Việt (gTTS).

    Example:
        curl -X POST http://localhost:8000/tts \\
             -H "Content-Type: application/json" \\
             -d '{"question": "Xin chào"}' --output output.mp3
    """
    text_input = req.question.strip()
    if not text_input:
        raise HTTPException(status_code=400, detail="Không có văn bản để đọc")

    try:
        from gtts import gTTS
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tts = gTTS(text=text_input, lang="vi")
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
#  Startup: Pre-load model khi server khởi động                    #
# ================================================================ #

@app.on_event("startup")
async def startup_load_model():
    """Pre-load fine-tuned model và resources khi server start."""
    print("\n" + "=" * 55)
    print("  RAG Bộ Luật Dân Sự 2015 — Fine-tuned Server")
    print("  Model: SeaLLMs-v3-7B-Chat + LoRA")
    print("=" * 55 + "\n")

    get_model_and_tokenizer()
    get_resources()

    print("\n  Server ready!\n")


# ================================================================ #
#  Dev mode                                                        #
# ================================================================ #

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_finetune:app", host="0.0.0.0", port=8000, reload=True)
