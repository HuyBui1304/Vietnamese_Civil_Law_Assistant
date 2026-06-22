"""
BƯỚC 7: Benchmark Vector DB + Bảng So Sánh  (Bài 5 — quan trọng)
==================================================================
7a. Tạo test set 30 câu hỏi pháp luật bằng Gemini
7b. Chạy benchmark trên 3 vector store + 2 chunking strategy
7c. Đo: Recall@3, Recall@5, MRR, Avg Latency (ms)
7d. Xuất 2 bảng so sánh ra CSV + in ra màn hình
"""

import json
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OUTPUTS_DIR = Path("outputs/benchmark")
TEST_SET_FILE = OUTPUTS_DIR / "test_set.json"
RESULTS_FILE  = OUTPUTS_DIR / "benchmark_results.csv"


# ================================================================ #
#  7a. Tạo test set bằng Gemini                                    #
# ================================================================ #
def generate_test_set(full_text: str, n: int = 50,
                      force_rerun: bool = False) -> list[dict]:
    """Dùng Gemini sinh n câu hỏi pháp luật kèm ground truth (số Điều)."""
    if TEST_SET_FILE.exists() and not force_rerun:
        print(f"  ○ Đã có test set: {TEST_SET_FILE}")
        return json.loads(TEST_SET_FILE.read_text(encoding="utf-8"))

    from google import genai
    client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    prompt = f"""Dựa vào Bộ Luật Dân Sự Việt Nam 2015 dưới đây, hãy tạo ra {n} câu hỏi pháp luật.

Yêu cầu:
- Mỗi câu hỏi phải trả lời được từ một Điều cụ thể trong đoạn trích
- Câu hỏi ngắn gọn, rõ ràng, tiếng Việt
- Trả về JSON array, mỗi phần tử gồm: "question" và "article_number" (số nguyên)

Văn bản:
{full_text}

Chỉ trả về JSON array, không giải thích thêm. Ví dụ:
[{{"question": "Phạm vi điều chỉnh của Bộ luật dân sự là gì?", "article_number": 1}}]"""

    print(f"  Đang sinh {n} câu hỏi bằng Gemini...")
    response = client_gemini.models.generate_content(
        model="gemini-3-flash-preview", contents=prompt
    )
    raw = response.text.strip()

    # Parse JSON
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    test_set = json.loads(raw.strip())

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    TEST_SET_FILE.write_text(
        json.dumps(test_set, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  ✓ Sinh xong {len(test_set)} câu hỏi → {TEST_SET_FILE}")
    return test_set


# ================================================================ #
#  7b. Embed câu hỏi                                               #
# ================================================================ #
def embed_questions(questions: list[str]) -> np.ndarray:
    from embedding import embed_texts
    return embed_texts(questions)


# ================================================================ #
#  7c. Tính metrics                                                #
# ================================================================ #
def recall_at_k(retrieved: list[dict], ground_truth_article: int, k: int) -> int:
    """1 nếu ground truth nằm trong top-k, 0 nếu không."""
    for r in retrieved[:k]:
        meta = r.get("metadata", {})
        if meta.get("article_number") == ground_truth_article:
            return 1
    return 0


def reciprocal_rank(retrieved: list[dict], ground_truth_article: int) -> float:
    """1/rank của kết quả đúng đầu tiên."""
    for i, r in enumerate(retrieved):
        meta = r.get("metadata", {})
        if meta.get("article_number") == ground_truth_article:
            return 1.0 / (i + 1)
    return 0.0


# ================================================================ #
#  7d. Chạy benchmark                                              #
# ================================================================ #
def benchmark_store(name: str, search_fn, q_vecs: np.ndarray,
                    test_set: list[dict], top_k: int = 5) -> dict:
    """Chạy benchmark trên 1 vector store."""
    recall3_list, recall5_list, mrr_list, latency_list = [], [], [], []

    for i, item in enumerate(test_set):
        q_vec = q_vecs[i]
        gt    = item.get("article_number")

        t0 = time.perf_counter()
        results = search_fn(q_vec, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000

        latency_list.append(latency_ms)

        if gt is not None:
            recall3_list.append(recall_at_k(results, gt, 3))
            recall5_list.append(recall_at_k(results, gt, 5))
            mrr_list.append(reciprocal_rank(results, gt))

    return {
        "Store": name,
        "Recall@3": round(np.mean(recall3_list), 3) if recall3_list else 0,
        "Recall@5": round(np.mean(recall5_list), 3) if recall5_list else 0,
        "MRR":      round(np.mean(mrr_list), 3)     if mrr_list     else 0,
        "Avg Latency (ms)": round(np.mean(latency_list), 1),
    }


def run_benchmark(faiss_index=None, faiss_chunks=None,
              chroma_col=None, qdrant_cl=None,
              chunks_fixed=None, embeddings_fixed=None,
              full_text: str | None = None,
              force_rerun: bool = False):
    """
    Returns:
        df_vectordb  (pd.DataFrame) — bảng so sánh 3 vector DB
        df_chunking  (pd.DataFrame) — bảng so sánh 2 chunking strategy
    """
    if RESULTS_FILE.exists() and not force_rerun:
        print(f"  ○ Đã có kết quả benchmark: {RESULTS_FILE}")
        df = pd.read_csv(RESULTS_FILE)
        print(df.to_string(index=False))
        return df, df

    # Load dữ liệu nếu chưa truyền vào
    if full_text is None:
        full_text = Path("data/extracted_text/full_text.txt").read_text(encoding="utf-8")
    if faiss_index is None:
        from vectordb import load_faiss, load_chroma, load_qdrant
        faiss_index, faiss_chunks = load_faiss()
        chroma_col = load_chroma()
        qdrant_cl  = load_qdrant()
    if chunks_fixed is None:
        from chunking import load_chunks
        from embedding import load_embeddings
        chunks_fixed, _ = load_chunks()
        embeddings_fixed, _ = load_embeddings()

    print(f"\n{'='*55}")
    print(f"  BƯỚC 7: BENCHMARK VECTOR DB")
    print(f"{'='*55}\n")

    # Tạo test set
    test_set = generate_test_set(full_text, n=50)
    questions = [item["question"] for item in test_set]

    print(f"\n  Embedding {len(questions)} câu hỏi...")
    q_vecs = embed_questions(questions)

    # Search functions
    from vectordb import search_faiss, search_chroma, search_qdrant

    def faiss_search(q_vec, top_k=5):
        return search_faiss(q_vec, faiss_index, faiss_chunks, top_k)

    def chroma_search(q_vec, top_k=5):
        return search_chroma(q_vec, chroma_col, top_k)

    def qdrant_search(q_vec, top_k=5):
        return search_qdrant(q_vec, qdrant_cl, top_k)

    # ── Bảng 1: So sánh 3 Vector DB ──────────────────────────────
    print("\n  Đang benchmark 3 Vector DB...")
    rows_db = []
    for name, fn in [("FAISS", faiss_search),
                 ("Chroma", chroma_search),
                 ("Qdrant", qdrant_search)]:
        print(f"    {name}...", end=" ", flush=True)
        row = benchmark_store(name, fn, q_vecs, test_set)
        rows_db.append(row)
        print(f"Recall@3={row['Recall@3']:.3f}  MRR={row['MRR']:.3f}  "
              f"Latency={row['Avg Latency (ms)']}ms")

    df_vectordb = pd.DataFrame(rows_db)

    # ── Bảng 2: So sánh 2 Chunking Strategy (dùng Qdrant) ────────
    print("\n  Đang benchmark 2 Chunking Strategy (Qdrant)...")

    # Build Qdrant tạm cho fixed chunks
    from vectordb import build_qdrant, search_qdrant
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    tmp_path = Path("vector_stores/qdrant_fixed_tmp")
    tmp_client = QdrantClient(path=str(tmp_path))
    dim = embeddings_fixed.shape[1]
    try:
        tmp_client.delete_collection("luat_fixed")
    except Exception:
        pass
    tmp_client.create_collection(
        collection_name="luat_fixed",
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    batch_size = 100
    for i in range(0, len(chunks_fixed), batch_size):
        b_chunks = chunks_fixed[i:i + batch_size]
        b_embs   = embeddings_fixed[i:i + batch_size]
        points = [
            PointStruct(id=i + j, vector=b_embs[j].tolist(),
                        payload={k: v for k, v in b_chunks[j].items() if v is not None})
            for j in range(len(b_chunks))
        ]
        tmp_client.upsert(collection_name="luat_fixed", points=points)

    def fixed_search(q_vec, top_k=5):
        results = tmp_client.query_points(
            collection_name="luat_fixed",
            query=q_vec.tolist(),
            limit=top_k,
        ).points
        return [{"text": r.payload.get("text", ""), "metadata": r.payload,
                 "score": r.score} for r in results]

    rows_chunk = []
    for name, fn in [("Fixed 500 chars", fixed_search),
                     ("Theo Điều luật", qdrant_search)]:
        print(f"    {name}...", end=" ", flush=True)
        row = benchmark_store(name, fn, q_vecs, test_set)
        rows_chunk.append(row)
        print(f"Recall@3={row['Recall@3']:.3f}  MRR={row['MRR']:.3f}")

    df_chunking = pd.DataFrame(rows_chunk)

    # Xóa DB tạm
    import shutil
    shutil.rmtree(tmp_path, ignore_errors=True)

    # ── In kết quả ────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  BẢNG 1: SO SÁNH 3 VECTOR DB")
    print(f"{'='*55}")
    print(df_vectordb.to_string(index=False))

    print(f"\n{'='*55}")
    print("  BẢNG 2: SO SÁNH 2 CHUNKING STRATEGY")
    print(f"{'='*55}")
    print(df_chunking.to_string(index=False))

    # Lưu CSV
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df_vectordb.to_csv(OUTPUTS_DIR / "benchmark_vectordb.csv", index=False)
    df_chunking.to_csv(OUTPUTS_DIR / "benchmark_chunking.csv", index=False)
    print(f"\n  ✓ Lưu: {OUTPUTS_DIR}/benchmark_vectordb.csv")
    print(f"  ✓ Lưu: {OUTPUTS_DIR}/benchmark_chunking.csv\n")

    return df_vectordb, df_chunking
