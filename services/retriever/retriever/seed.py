import os
import glob
import uuid
import yaml

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
REG_DIR = "/app/data/regulations"
POLICY_PATH = "/app/data/policy.yaml"


def get_embed():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def _collection_exists(client: QdrantClient, name: str) -> bool:
    cols = client.get_collections()
    return any(c.name == name for c in cols.collections)


def _existing_dim(client: QdrantClient, name: str) -> int | None:
    """
    Returns current vector size for the collection, handling both single and named vectors.
    """
    try:
        info = client.get_collection(name)
        # Newer clients: vector config is under config.params.vectors
        vectors_cfg = info.config.params.vectors  # could be VectorParams or dict[str, VectorParams]
        if isinstance(vectors_cfg, VectorParams):
            return vectors_cfg.size
        if isinstance(vectors_cfg, dict):
            # pick the first named vector
            vp = next(iter(vectors_cfg.values()), None)
            return vp.size if isinstance(vp, VectorParams) else None
    except Exception:
        pass
    return None

def _ensure_collection(client, name, dim):
    force = os.getenv("FORCE_RECREATE") == "1"
    if force or not _collection_exists(client, name):
        client.recreate_collection(collection_name=name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        return
    cur = _existing_dim(client, name)
    if cur is None or int(cur) != int(dim):
        client.recreate_collection(collection_name=name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

def main():
    with open(POLICY_PATH, "r", encoding="utf-8") as f:
        policy = yaml.safe_load(f) or {}
    collection = (policy.get("retrieval") or {}).get("collection", "regs")

    client = QdrantClient(url=QDRANT_URL)
    embed = get_embed()
    dim = len(embed.embed_query("dimension_probe"))

    _ensure_collection(client, collection, dim)

    files = sorted(glob.glob(os.path.join(REG_DIR, "*.*")))
    if not files:
        print(f"No regulation files found under {REG_DIR}")
        return

    points: list[PointStruct] = []
    for fp in files:
        reg_id = os.path.splitext(os.path.basename(fp))[0]  # file name becomes reg_id
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read().strip()
        vec = embed.embed_query(text)
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"text": text, "metadata": {"reg_id": reg_id}},
            )
        )

    if points:
        client.upsert(collection_name=collection, points=points)
        print(f"Upserted {len(points)} regulation docs into {collection}")


if __name__ == "__main__":
    main()