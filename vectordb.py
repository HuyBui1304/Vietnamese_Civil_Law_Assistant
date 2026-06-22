"""
BƯỚC 6: Xây dựng 3 Vector Database  (Bài 0 + Bài 5)
=====================================================
6a. FAISS   — baseline, brute-force, không có metadata
6b. Chroma  — persistent, metadata filtering
6c. Qdrant  — production, batch insert, metadata filtering nâng cao
"""

import json
import pickle
import numpy as np
from pathlib import Path

CHUNKS_DIR     = Path("data/chunks")
EMBEDDINGS_DIR = Path("data/embeddings")
VS_FAISS       = Path("vector_stores/faiss")
VS_CHROMA      = Path("vector_stores/chroma")
VS_QDRANT      = Path("vector_stores/qdrant")


# ================================================================ #
#  6a. FAISS                                                        #
# ================================================================ #
def build_faiss(chunks: list[dict], embeddings: np.ndarray):
    try:
        import faiss
    except ImportError:
        raise ImportError("pip install faiss-cpu")

    VS_FAISS.mkdir(parents=True, exist_ok=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(VS_FAISS / "index.faiss"))
    with open(VS_FAISS / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"  ✓ FAISS : {index.ntotal:,} vectors  → {VS_FAISS}/")
    return index, chunks


def load_faiss():
    import faiss
    index = faiss.read_index(str(VS_FAISS / "index.faiss"))
    with open(VS_FAISS / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def search_faiss(query_vec: np.ndarray, index, chunks: list[dict],
                 top_k: int = 5) -> list[dict]:
    q = np.ascontiguousarray(query_vec.reshape(1, -1).astype("float32"))
    D, I = index.search(q, top_k)
    return [{"text": chunks[i]["text"], "metadata": chunks[i], "score": float(D[0][j])}
            for j, i in enumerate(I[0]) if i < len(chunks)]


# ================================================================ #
#  6b. Chroma                                                       #
# ================================================================ #
def build_chroma(chunks: list[dict], embeddings: np.ndarray):
    try:
        import chromadb
    except ImportError:
        raise ImportError("pip install chromadb")

    VS_CHROMA.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VS_CHROMA))

    # Xóa collection cũ nếu có
    try:
        client.delete_collection("luat_dan_su")
    except Exception:
        pass

    collection = client.create_collection(
        name="luat_dan_su",
        metadata={"hnsw:space": "cosine"},
    )

    # Insert theo batch 500
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embs   = embeddings[i:i + batch_size]
        collection.add(
            ids        = [f"{c['id']}_{i + j}" for j, c in enumerate(batch_chunks)],
            embeddings = batch_embs.tolist(),
            documents  = [c["text"] for c in batch_chunks],
            metadatas  = [{k: v for k, v in c.items()
                           if k != "text" and v is not None} for c in batch_chunks],
        )

    print(f"  ✓ Chroma: {collection.count():,} vectors  → {VS_CHROMA}/")
    return collection


def load_chroma():
    import chromadb
    client = chromadb.PersistentClient(path=str(VS_CHROMA))
    return client.get_collection("luat_dan_su")


def search_chroma(query_vec: np.ndarray, collection, top_k: int = 5,
                  filter_type: str | None = None) -> list[dict]:
    where = {"type": filter_type} if filter_type else None
    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=top_k,
        where=where,
    )
    return [{"text": results["documents"][0][i],
             "metadata": results["metadatas"][0][i],
             "score": results["distances"][0][i]}
            for i in range(len(results["documents"][0]))]


# ================================================================ #
#  6c. Qdrant                                                       #
# ================================================================ #
def build_qdrant(chunks: list[dict], embeddings: np.ndarray):
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
    except ImportError:
        raise ImportError("pip install qdrant-client")

    VS_QDRANT.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(VS_QDRANT))
    dim = embeddings.shape[1]

    # Tạo lại collection
    try:
        client.delete_collection("luat_dan_su")
    except Exception:
        pass
    client.create_collection(
        collection_name="luat_dan_su",
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    # Batch insert 100 points
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embs   = embeddings[i:i + batch_size]
        points = [
            PointStruct(
                id=i + j,
                vector=batch_embs[j].tolist(),
                payload={k: v for k, v in batch_chunks[j].items()
                         if v is not None},
            )
            for j in range(len(batch_chunks))
        ]
        client.upsert(collection_name="luat_dan_su", points=points)

    info = client.get_collection("luat_dan_su")
    print(f"  ✓ Qdrant: {info.points_count:,} vectors  → {VS_QDRANT}/")
    return client


def load_qdrant():
    from qdrant_client import QdrantClient
    return QdrantClient(path=str(VS_QDRANT))


def search_qdrant(query_vec: np.ndarray, client, top_k: int = 5,
                  filter_type: str | None = None) -> list[dict]:
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    query_filter = None
    if filter_type:
        query_filter = Filter(
            must=[FieldCondition(key="type", match=MatchValue(value=filter_type))]
        )
    results = client.query_points(
        collection_name="luat_dan_su",
        query=query_vec.tolist(),
        limit=top_k,
        query_filter=query_filter,
    ).points
    return [{"text": r.payload.get("text", ""), "metadata": r.payload, "score": r.score}
            for r in results]


# ================================================================ #
#  Run                                                              #
# ================================================================ #
def run_vectordb(chunks_article: list[dict] | None = None,
              embeddings_article: np.ndarray | None = None,
              force_rerun: bool = False):
    """
    Returns:
        faiss_index, faiss_chunks  — FAISS index + chunk list
        chroma_collection          — Chroma collection
        qdrant_client              — Qdrant client
    """
    faiss_ready  = (VS_FAISS  / "index.faiss").exists()
    chroma_ready = (VS_CHROMA / "chroma.sqlite3").exists()
    qdrant_ready = any(VS_QDRANT.iterdir()) if VS_QDRANT.exists() else False

    if faiss_ready and chroma_ready and qdrant_ready and not force_rerun:
        print("  ○ Đã có vector stores. Load lại...")
        faiss_index, faiss_chunks = load_faiss()
        chroma_col = load_chroma()
        qdrant_cl  = load_qdrant()
        print(f"  ✓ FAISS  : {faiss_index.ntotal:,} vectors")
        print(f"  ✓ Chroma : {chroma_col.count():,} vectors")
        print(f"  ✓ Qdrant : {qdrant_cl.get_collection('luat_dan_su').points_count:,} vectors")
        return (faiss_index, faiss_chunks), chroma_col, qdrant_cl

    # Load nếu chưa truyền vào
    if chunks_article is None or embeddings_article is None:
        from chunking import load_chunks
        from embedding import load_embeddings
        _, chunks_article = load_chunks()
        _, embeddings_article = load_embeddings()

    print(f"\n{'='*55}")
    print(f"  BƯỚC 6: XÂY DỰNG 3 VECTOR DATABASE")
    print(f"{'='*55}")
    print(f"  Chunks: {len(chunks_article):,}  |  Dim: {embeddings_article.shape[1]}\n")

    faiss_index, faiss_chunks = build_faiss(chunks_article, embeddings_article)
    chroma_col  = build_chroma(chunks_article, embeddings_article)
    qdrant_cl   = build_qdrant(chunks_article, embeddings_article)

    print(f"\n  ✓ Bước 6 hoàn tất!\n")
    return (faiss_index, faiss_chunks), chroma_col, qdrant_cl
