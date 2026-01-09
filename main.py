import os
import asyncio
import random
from datetime import datetime, timezone
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

app = FastAPI(title="Graphiti Ingestion Service")

API_KEY = os.getenv("GRAPHITI_API_KEY")
RUN_SCHEMA = os.getenv("RUN_SCHEMA", "false").lower() == "true"
MAX_INFLIGHT = int(os.getenv("MAX_INFLIGHT_EPISODES", "3"))
EPISODE_TIMEOUT = max(int(os.getenv("EPISODE_TIMEOUT_SECONDS", "90")), 10)
MAX_CHUNKS_PER_REQUEST = int(os.getenv("MAX_CHUNKS_PER_REQUEST", "25"))

def must(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

graphiti = Graphiti(
    must("NEO4J_URI"),
    must("NEO4J_USER"),
    must("NEO4J_PASSWORD"),
)

GLOBAL_SEM = asyncio.Semaphore(MAX_INFLIGHT)

@app.on_event("startup")
async def startup():
    if RUN_SCHEMA:
        await graphiti.build_indices_and_constraints()

@app.on_event("shutdown")
async def shutdown():
    await graphiti.close()

def require_key(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

class Chunk(BaseModel):
    chunk_id: str
    text: str
    source_file_id: str | None = None
    source_file_name: str | None = None

class IngestRequest(BaseModel):
    doc: dict
    chunks: list[Chunk]

@app.get("/")
async def root():
    return {"status": "ok"}

async def add_episode_once(c: Chunk):
    await graphiti.add_episode(
        name=c.chunk_id,
        episode_body=c.text,
        source=EpisodeType.text,
        source_description=c.source_file_name or "openwebui",
        reference_time=datetime.now(timezone.utc),
    )

async def add_episode_with_timeout_and_retry(c: Chunk, attempts: int = 3):
    for i in range(attempts):
        try:
            await asyncio.wait_for(add_episode_once(c), timeout=EPISODE_TIMEOUT)
            return {"chunk_id": c.chunk_id, "status": "ok"}
        except asyncio.TimeoutError:
            if i < attempts - 1:
                await asyncio.sleep(0.5 + random.random())
                continue
            return {"chunk_id": c.chunk_id, "status": "timeout"}
        except Exception as e:
            msg = str(e).lower()
            transient = (
                "failed to obtain a connection from the pool" in msg
                or "rate limit" in msg
                or "temporarily unavailable" in msg
                or "invalid duplicate_facts" in msg
                or "timeout" in msg
            )
            if transient and i < attempts - 1:
                await asyncio.sleep(0.5 + random.random())
                continue
            return {"chunk_id": c.chunk_id, "status": "error", "error": str(e)[:500]}

@app.post("/ingest-chunks")
async def ingest_chunks(req: IngestRequest, x_api_key: str | None = Header(default=None)):
    require_key(x_api_key)

    if len(req.chunks) > MAX_CHUNKS_PER_REQUEST:
        raise HTTPException(413, f"Too many chunks; max {MAX_CHUNKS_PER_REQUEST}")

    async def ingest_one(c: Chunk):
        async with GLOBAL_SEM:
            return await add_episode_with_timeout_and_retry(c)

    results = await asyncio.gather(*(ingest_one(c) for c in req.chunks))
    ok = sum(1 for r in results if r.get("status") == "ok")
    timeouts = sum(1 for r in results if r.get("status") == "timeout")
    errors = [r for r in results if r.get("status") == "error"]

    return {
        "status": "ok",
        "chunks_received": len(req.chunks),
        "chunks_ingested_ok": ok,
        "chunks_timeout": timeouts,
        "chunks_error": len(errors),
        "errors_sample": errors[:5],
    }

@app.post("/finalize")
async def finalize(x_api_key: str | None = Header(default=None)):
    require_key(x_api_key)
    return {"status": "finalized"}
