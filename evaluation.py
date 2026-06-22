"""
BƯỚC 11: Evaluation toàn diện  (Bài 5 — mở rộng từ Bước 7)
============================================================
11a. Retrieval metrics: Recall@K, MRR, NDCG (từ test set Bước 7)
11b. Generation metrics: Faithfulness, Answer Relevance (LLM-as-judge)
11c. Tổng hợp bảng đánh giá + xuất CSV + vẽ biểu đồ
"""

import json
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

EVAL_DIR     = Path("outputs/evaluation")
REPORT_FILE  = EVAL_DIR / "eval_report.csv"
TEST_SET_FILE = Path("outputs/benchmark/test_set.json")


# ================================================================ #
#  11a. Retrieval metrics (mở rộng từ Bước 7)                     #
# ================================================================ #

def ndcg_at_k(retrieved: list[dict], ground_truth_article: int, k: int) -> float:
    """Normalized Discounted Cumulative Gain tại k."""
    import math
    dcg = 0.0
    for i, r in enumerate(retrieved[:k]):
        if r.get("metadata", {}).get("article_number") == ground_truth_article:
            dcg += 1.0 / math.log2(i + 2)
    ideal_dcg = 1.0 / math.log2(2)  # best case: rank 1
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_retrieval(search_fn, q_vecs: np.ndarray,
                       test_set: list[dict], store_name: str) -> dict:
    """Đánh giá retrieval cho một vector store.

    Args:
        search_fn:  Callable(q_vec, top_k) → list[dict]
        q_vecs:     Embedded questions.
        test_set:   List {"question":..., "article_number":...}
        store_name: Tên hiển thị.

    Returns:
        Dict với các metrics.
    """
    r3, r5, mrr, ndcg3, lat = [], [], [], [], []

    for i, item in enumerate(test_set):
        gt = item.get("article_number")
        t0 = time.perf_counter()
        results = search_fn(q_vecs[i], top_k=5)
        lat.append((time.perf_counter() - t0) * 1000)

        if gt is not None:
            r3.append(int(any(
                r.get("metadata", {}).get("article_number") == gt
                for r in results[:3]
            )))
            r5.append(int(any(
                r.get("metadata", {}).get("article_number") == gt
                for r in results[:5]
            )))
            mrr_val = next(
                (1.0 / (j + 1) for j, r in enumerate(results)
                 if r.get("metadata", {}).get("article_number") == gt),
                0.0,
            )
            mrr.append(mrr_val)
            ndcg3.append(ndcg_at_k(results, gt, 3))

    return {
        "Store": store_name,
        "Recall@3": round(np.mean(r3), 3) if r3 else 0,
        "Recall@5": round(np.mean(r5), 3) if r5 else 0,
        "MRR":      round(np.mean(mrr), 3) if mrr else 0,
        "NDCG@3":   round(np.mean(ndcg3), 3) if ndcg3 else 0,
        "Latency(ms)": round(np.mean(lat), 1),
    }


# ================================================================ #
#  11b. Generation metrics (LLM-as-judge)                         #
# ================================================================ #

def llm_judge(question: str, context: str, answer: str,
              judge_client) -> dict:
    """Dùng Gemini đánh giá chất lượng câu trả lời RAG.

    Args:
        question:     Câu hỏi.
        context:      Đoạn văn bản retrieve được.
        answer:       Câu trả lời của RAG.
        judge_client: google.genai.Client instance.

    Returns:
        {"faithfulness": float, "relevance": float} (0-1)
    """
    prompt = f"""Đánh giá chất lượng câu trả lời RAG theo 2 tiêu chí.

CÂU HỎI: {question}

NGỮ CẢNH (từ vector DB):
{context[:800]}

CÂU TRẢ LỜI:
{answer[:500]}

Hãy chấm điểm theo thang 0.0 đến 1.0:
1. faithfulness: Câu trả lời có trung thực với ngữ cảnh không? (1.0 = hoàn toàn dựa trên ngữ cảnh)
2. relevance: Câu trả lời có liên quan đến câu hỏi không? (1.0 = trả lời đúng trọng tâm)

Chỉ trả về JSON: {{"faithfulness": 0.0, "relevance": 0.0}}"""

    try:
        response = judge_client.models.generate_content(
            model="gemini-3-flash-preview", contents=prompt
        )
        raw = response.text.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        scores = json.loads(raw.strip())
        return {
            "faithfulness": float(scores.get("faithfulness", 0)),
            "relevance":    float(scores.get("relevance", 0)),
        }
    except Exception:
        return {"faithfulness": 0.0, "relevance": 0.0}


def evaluate_generation(search_fn, rag_fn, q_vecs: np.ndarray,
                        test_set: list[dict], store_name: str,
                        n_samples: int = 10) -> dict:
    """Đánh giá chất lượng generation với LLM-as-judge.

    Args:
        search_fn:   Callable(q_vec, top_k) → list[dict]
        rag_fn:      Callable(question) → str  (gọi Gemini sinh câu trả lời)
        q_vecs:      Embedded questions.
        test_set:    Test set.
        store_name:  Tên vector store.
        n_samples:   Số câu hỏi đánh giá (giữ nhỏ để tiết kiệm API).

    Returns:
        Dict {"Store": ..., "Faithfulness": ..., "Answer Relevance": ...}
    """
    from google import genai
    judge_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    samples = test_set[:n_samples]
    faith_scores, rel_scores = [], []

    for i, item in enumerate(samples):
        q = item["question"]
        q_vec = q_vecs[i]
        results = search_fn(q_vec, top_k=3)
        context = "\n\n---\n\n".join(r["text"] for r in results)
        answer = rag_fn(q)
        scores = llm_judge(q, context, answer, judge_client)
        faith_scores.append(scores["faithfulness"])
        rel_scores.append(scores["relevance"])

    return {
        "Store":            store_name,
        "Faithfulness":     round(np.mean(faith_scores), 3),
        "Answer Relevance": round(np.mean(rel_scores), 3),
        "Samples":          len(samples),
    }


# ================================================================ #
#  11c. Biểu đồ                                                   #
# ================================================================ #

def plot_evaluation(df_retrieval: pd.DataFrame, df_generation: pd.DataFrame):
    """Vẽ biểu đồ so sánh và lưu PNG.

    Args:
        df_retrieval:  DataFrame retrieval metrics.
        df_generation: DataFrame generation metrics.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("  ⚠ matplotlib chưa cài — bỏ qua biểu đồ")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Evaluation Report — Bộ Luật Dân Sự RAG", fontsize=13)

    # Retrieval metrics
    metrics_r = ["Recall@3", "Recall@5", "MRR", "NDCG@3"]
    available_r = [m for m in metrics_r if m in df_retrieval.columns]
    df_retrieval.set_index("Store")[available_r].plot(
        kind="bar", ax=axes[0], rot=0, colormap="tab10"
    )
    axes[0].set_title("Retrieval Metrics")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("Score")

    # Generation metrics
    if not df_generation.empty:
        metrics_g = ["Faithfulness", "Answer Relevance"]
        available_g = [m for m in metrics_g if m in df_generation.columns]
        df_generation.set_index("Store")[available_g].plot(
            kind="bar", ax=axes[1], rot=0, colormap="Set2"
        )
        axes[1].set_title("Generation Metrics (LLM-as-judge)")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(loc="lower right", fontsize=8)
        axes[1].set_ylabel("Score")
    else:
        axes[1].text(0.5, 0.5, "No generation data",
                     ha="center", va="center", transform=axes[1].transAxes)

    plt.tight_layout()
    chart_path = EVAL_DIR / "eval_chart.png"
    plt.savefig(chart_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Biểu đồ: {chart_path}")


# ================================================================ #
#  Run                                                             #
# ================================================================ #

def run_evaluation(faiss_index=None, faiss_chunks=None,
               chroma_col=None, qdrant_cl=None,
               gemini_api_key: str | None = None,
               run_generation_eval: bool = True,
               force_rerun: bool = False):
    """
    Returns:
        df_retrieval  (pd.DataFrame) — retrieval metrics
        df_generation (pd.DataFrame) — generation metrics (LLM-as-judge)
    """
    if REPORT_FILE.exists() and not force_rerun:
        print(f"  ○ Đã có báo cáo: {REPORT_FILE}")
        df = pd.read_csv(REPORT_FILE)
        print(df.to_string(index=False))
        return df, pd.DataFrame()

    print(f"\n{'='*55}")
    print(f"  BƯỚC 11: EVALUATION TOÀN DIỆN")
    print(f"{'='*55}\n")

    # Load resources
    if faiss_index is None:
        from vectordb import load_faiss, load_chroma, load_qdrant
        faiss_index, faiss_chunks = load_faiss()
        chroma_col = load_chroma()
        qdrant_cl  = load_qdrant()

    if not TEST_SET_FILE.exists():
        raise FileNotFoundError(
            f"Chưa có test set. Chạy Bước 7 trước: {TEST_SET_FILE}"
        )

    test_set  = json.loads(TEST_SET_FILE.read_text(encoding="utf-8"))
    questions = [item["question"] for item in test_set]

    from embedding import embed_texts
    
    q_vecs = embed_texts(questions)

    from vectordb import search_faiss, search_chroma, search_qdrant

    def faiss_search(q_vec, top_k=5):
        return search_faiss(q_vec, faiss_index, faiss_chunks, top_k)

    def chroma_search(q_vec, top_k=5):
        return search_chroma(q_vec, chroma_col, top_k)

    def qdrant_search(q_vec, top_k=5):
        return search_qdrant(q_vec, qdrant_cl, top_k)

    # ── Retrieval Evaluation ────────────────────────────────────
    print("  Đánh giá Retrieval...")
    rows_r = []
    for name, fn in [
                 ("FAISS", faiss_search),
                 ("Chroma", chroma_search),
                 ("Qdrant", qdrant_search)]:
        print(f"    {name}...", end=" ", flush=True)
        row = evaluate_retrieval(fn, q_vecs, test_set, name)
        rows_r.append(row)
        print(f"Recall@3={row['Recall@3']:.3f}  MRR={row['MRR']:.3f}  NDCG@3={row['NDCG@3']:.3f}")
    df_retrieval = pd.DataFrame(rows_r)

    # ── Generation Evaluation ───────────────────────────────────
    df_generation = pd.DataFrame()
    if run_generation_eval:
        print("\n  Đánh giá Generation (LLM-as-judge, 10 câu)...")
        from google import genai
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        gemini_client = genai.Client(api_key=api_key)

        def make_rag_fn(search_fn_inner):
            def rag_fn(question):
                from embedding import embed_texts as _embed_texts
                
                qv = _embed_texts([question])[0]
                results = search_fn_inner(qv, top_k=3)
                context = "\n\n---\n\n".join(r["text"] for r in results)
                prompt = (
                    "Trả lời câu hỏi dựa vào ngữ cảnh sau, bằng tiếng Việt:\n\n"
                    f"NGỮ CẢNH:\n{context}\n\nCÂU HỎI: {question}\n\nTRẢ LỜI:"
                )
                response = gemini_client.models.generate_content(
                    model="gemini-3-flash-preview", contents=prompt
                )
                return response.text.strip()
            return rag_fn

        rows_g = []
        for name, search_fn in [
                          ("FAISS", faiss_search),
                          ("Chroma", chroma_search),
                          ("Qdrant", qdrant_search)]:
            print(f"    {name}...", end=" ", flush=True)
            row = evaluate_generation(
                search_fn, make_rag_fn(search_fn),
                q_vecs, test_set, name, n_samples=10,
            )
            rows_g.append(row)
            print(f"Faithfulness={row['Faithfulness']:.3f}  Relevance={row['Answer Relevance']:.3f}")
        df_generation = pd.DataFrame(rows_g)

    # ── In kết quả ────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  BẢNG RETRIEVAL METRICS")
    print(f"{'='*55}")
    print(df_retrieval.to_string(index=False))

    if not df_generation.empty:
        print(f"\n{'='*55}")
        print("  BẢNG GENERATION METRICS (LLM-as-judge)")
        print(f"{'='*55}")
        print(df_generation.to_string(index=False))

    # Lưu CSV + biểu đồ
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    df_retrieval.to_csv(REPORT_FILE, index=False)
    df_retrieval.to_csv(EVAL_DIR / "eval_retrieval.csv", index=False)
    if not df_generation.empty:
        df_generation.to_csv(EVAL_DIR / "eval_generation.csv", index=False)

    plot_evaluation(df_retrieval, df_generation)

    print(f"\n  ✓ Lưu: {EVAL_DIR}/")
    print(f"  ✓ Bước 11 hoàn tất!\n")
    return df_retrieval, df_generation
