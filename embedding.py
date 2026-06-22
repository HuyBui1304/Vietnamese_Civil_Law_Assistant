"""
BƯỚC 5: Embedding  (Bài 0)
===========================
Dùng SentenceTransformers all-MiniLM-L6-v2 (384 chiều)
Embed cả 2 bộ chunks → lưu ra data/embeddings/
"""

import json
import numpy as np
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm
except ImportError:
    raise ImportError("pip install sentence-transformers tqdm")

EMBEDDINGS_DIR = Path("data/embeddings")
MODEL_NAME     = "all-MiniLM-L6-v2"
BATCH_SIZE     = 64


def embed_chunks(chunks: list[dict], model: SentenceTransformer,
                 desc: str = "Embedding") -> np.ndarray:
    texts = [c["text"] for c in chunks]
    vectors = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"  {desc}"):
        batch = texts[i:i + BATCH_SIZE]
        vectors.append(model.encode(batch, show_progress_bar=False))
    return np.vstack(vectors).astype("float32")


_cached_model = None

def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed danh sách text tùy ý (dùng ở benchmark, evaluation...)."""
    global _cached_model
    if _cached_model is None:
        _cached_model = SentenceTransformer(MODEL_NAME)
    return _cached_model.encode(texts, show_progress_bar=False).astype("float32")


def run_embedding(chunks_fixed: list[dict] | None = None,
              chunks_article: list[dict] | None = None,
              force_rerun: bool = False):
    """
    Returns:
        embeddings_fixed   (np.ndarray)  shape: (N, 384)
        embeddings_article (np.ndarray)  shape: (M, 384)
    """
    fixed_path   = EMBEDDINGS_DIR / "embeddings_fixed.npy"
    article_path = EMBEDDINGS_DIR / "embeddings_article.npy"

    if fixed_path.exists() and article_path.exists() and not force_rerun:
        print("  ○ Đã có embeddings. Dùng load_embeddings() hoặc set force_rerun=True")
        return load_embeddings()

    if chunks_fixed is None or chunks_article is None:
        from chunking import load_chunks
        chunks_fixed, chunks_article = load_chunks()

    print(f"\n{'='*55}")
    print(f"  BƯỚC 5: EMBEDDING")
    print(f"{'='*55}")
    print(f"  Model : {MODEL_NAME}")
    print(f"  Fixed chunks  : {len(chunks_fixed):,}")
    print(f"  Article chunks: {len(chunks_article):,}")
    print()
    print("  Đang load model (lần đầu sẽ tải về ~90MB)...")

    model = SentenceTransformer(MODEL_NAME)
    print(f"  ✓ Model sẵn sàng  (dim={model.get_sentence_embedding_dimension()})\n")

    emb_fixed   = embed_chunks(chunks_fixed,   model, desc="Fixed-size")
    emb_article = embed_chunks(chunks_article, model, desc="Theo Điều luật")

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(fixed_path,   emb_fixed)
    np.save(article_path, emb_article)

    print(f"\n  ✓ embeddings_fixed  : {emb_fixed.shape}  → {fixed_path}")
    print(f"  ✓ embeddings_article: {emb_article.shape}  → {article_path}\n")

    return emb_fixed, emb_article


def load_embeddings():
    fixed   = np.load(EMBEDDINGS_DIR / "embeddings_fixed.npy")
    article = np.load(EMBEDDINGS_DIR / "embeddings_article.npy")
    print(f"  ✓ Loaded fixed   : {fixed.shape}")
    print(f"  ✓ Loaded article : {article.shape}")
    return fixed, article
