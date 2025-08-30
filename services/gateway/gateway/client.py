# gateway/client.py
import os
import httpx

CLASSIFIER_URL = os.getenv("CLASSIFIER_URL", "http://classifier:8000")
RETRIEVER_URL = os.getenv("RETRIEVER_URL", "http://retriever:8000")


def _mk_timeout(prefix: str, *, read_default: float, connect_default: float = 10.0, write_default: float = 60.0) -> httpx.Timeout:
    """Build an httpx.Timeout from env vars like:
       {PREFIX}_TIMEOUT (read & pool), {PREFIX}_CONNECT_TIMEOUT, {PREFIX}_WRITE_TIMEOUT
    """
    read = float(os.getenv(f"{prefix}_TIMEOUT", str(read_default)))
    connect = float(os.getenv(f"{prefix}_CONNECT_TIMEOUT", str(connect_default)))
    write = float(os.getenv(f"{prefix}_WRITE_TIMEOUT", str(write_default)))
    # pool timeout usually matches read
    pool = float(os.getenv(f"{prefix}_POOL_TIMEOUT", str(read)))
    return httpx.Timeout(connect=connect, read=read, write=write, pool=pool)


# Defaults: classifier calls can take a while on first Ollama run; retriever /index can be even longer.
CLASSIFIER_TIMEOUT = _mk_timeout("CLASSIFIER", read_default=240.0, connect_default=10.0, write_default=60.0)
RETRIEVER_TIMEOUT = _mk_timeout("RETRIEVER", read_default=600.0, connect_default=10.0, write_default=120.0)


async def infer(payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=CLASSIFIER_TIMEOUT) as client:
        r = await client.post(f"{CLASSIFIER_URL}/infer", json=payload)
        r.raise_for_status()
        return r.json()


async def seed_retriever() -> dict:
    async with httpx.AsyncClient(timeout=RETRIEVER_TIMEOUT) as client:
        r = await client.post(f"{RETRIEVER_URL}/index")
        r.raise_for_status()
        return r.json()