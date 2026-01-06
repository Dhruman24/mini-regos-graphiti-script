import os
from datetime import datetime, timezone
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

app = FastAPI(title="Graphiti Ingestion Service")

API_KEY = os.getenv("GRAPHITI_API_KEY")  # ✅ correct

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

@app.on_event("startup")
async def startup():
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

@app.post("/ingest-chunks")
async def ingest_chunks(req: IngestRequest, x_api_key: str | None = Header(default=None)):
    require_key(x_api_key)
    for c in req.chunks:
        await graphiti.add_episode(
            name=c.chunk_id,
            episode_body=c.text,
            source=EpisodeType.text,
            source_description=c.source_file_name or "openwebui",
            reference_time=datetime.now(timezone.utc),
        )
    return {"status": "ok", "chunks_ingested": len(req.chunks)}

@app.post("/finalize")
async def finalize(x_api_key: str | None = Header(default=None)):
    require_key(x_api_key)
    return {"status": "finalized"}
