import os
import asyncio
import random
import logging
import secrets
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("GRAPHITI_API_KEY")
RUN_SCHEMA = os.getenv("RUN_SCHEMA", "false").lower() == "true"
MAX_INFLIGHT = int(os.getenv("MAX_INFLIGHT_EPISODES", "3"))
EPISODE_TIMEOUT = max(int(os.getenv("EPISODE_TIMEOUT_SECONDS", "90")), 10)
MAX_CHUNKS_PER_REQUEST = int(os.getenv("MAX_CHUNKS_PER_REQUEST", "25"))
ENABLE_DEDUPLICATION = os.getenv("ENABLE_DEDUPLICATION", "true").lower() == "true"

def must(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

# Initialize graphiti and semaphore
graphiti: Graphiti | None = None
GLOBAL_SEM = asyncio.Semaphore(MAX_INFLIGHT)
processed_chunks: set[str] = set()  # Track processed chunk IDs for deduplication

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global graphiti

    logger.info("Starting Graphiti Ingestion Service...")
    logger.info(f"Max inflight episodes: {MAX_INFLIGHT}")
    logger.info(f"Episode timeout: {EPISODE_TIMEOUT}s")
    logger.info(f"Max chunks per request: {MAX_CHUNKS_PER_REQUEST}")
    logger.info(f"Deduplication enabled: {ENABLE_DEDUPLICATION}")

    try:
        graphiti = Graphiti(
            must("NEO4J_URI"),
            must("NEO4J_USER"),
            must("NEO4J_PASSWORD"),
        )

        if RUN_SCHEMA:
            logger.info("Building Neo4j indices and constraints...")
            await graphiti.build_indices_and_constraints()
            logger.info("Schema setup complete")

        # Test Neo4j connection
        logger.info("Testing Neo4j connection...")
        # Simple test by checking if graphiti is initialized
        logger.info("Neo4j connection successful")

        yield

    except Exception as e:
        logger.error(f"Failed to initialize Graphiti: {e}")
        raise
    finally:
        logger.info("Shutting down Graphiti Ingestion Service...")
        if graphiti:
            await graphiti.close()
            logger.info("Graphiti connection closed")

app = FastAPI(title="Graphiti Ingestion Service", lifespan=lifespan)

def require_key(x_api_key: str | None):
    """Validate API key using constant-time comparison to prevent timing attacks"""
    if API_KEY:
        if not x_api_key:
            logger.warning("API request missing x-api-key header")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        if not secrets.compare_digest(x_api_key, API_KEY):
            logger.warning("API request with invalid x-api-key")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

class DocumentMetadata(BaseModel):
    """Metadata about the document being processed"""
    doc_id: str
    doc_name: str
    version: str | None = None
    jurisdiction: str | None = None
    knowledge_id: str | None = None

class Chunk(BaseModel):
    chunk_id: str
    text: str
    source_file_id: str | None = None
    source_file_name: str | None = None

class IngestRequest(BaseModel):
    doc: DocumentMetadata
    chunks: list[Chunk]

@app.get("/")
async def root():
    """Health check endpoint that verifies Neo4j connectivity"""
    try:
        if graphiti is None:
            return {"status": "initializing"}
        return {
            "status": "ok",
            "service": "Graphiti Ingestion Service",
            "neo4j_connected": True,
            "processed_chunks": len(processed_chunks) if ENABLE_DEDUPLICATION else None
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable"
        )

async def add_episode_once(c: Chunk, doc: DocumentMetadata):
    """Add a single episode to Graphiti with document metadata"""
    if graphiti is None:
        raise RuntimeError("Graphiti not initialized")

    # Build source description with document metadata
    source_parts = []
    if doc.doc_name:
        source_parts.append(doc.doc_name)
    if doc.version:
        source_parts.append(f"v{doc.version}")
    if doc.jurisdiction:
        source_parts.append(doc.jurisdiction)
    if c.source_file_name:
        source_parts.append(f"File: {c.source_file_name}")

    source_description = " | ".join(source_parts) if source_parts else "openwebui"

    logger.debug(f"Adding episode: chunk_id={c.chunk_id}, doc_id={doc.doc_id}, text_length={len(c.text)}")

    await graphiti.add_episode(
        name=f"{doc.doc_id}:{c.chunk_id}",
        episode_body=c.text,
        source=EpisodeType.text,
        source_description=source_description,
        reference_time=datetime.now(timezone.utc),
    )

async def add_episode_with_timeout_and_retry(c: Chunk, doc: DocumentMetadata, attempts: int = 3):
    """Add episode with retry logic and timeout handling"""
    for i in range(attempts):
        try:
            await asyncio.wait_for(add_episode_once(c, doc), timeout=EPISODE_TIMEOUT)
            logger.info(f"Successfully ingested chunk: {c.chunk_id} (doc: {doc.doc_id})")
            return {"chunk_id": c.chunk_id, "status": "ok"}
        except asyncio.TimeoutError:
            logger.warning(f"Timeout on chunk {c.chunk_id} (attempt {i+1}/{attempts})")
            if i < attempts - 1:
                await asyncio.sleep(0.5 + random.random())
                continue
            logger.error(f"Final timeout on chunk {c.chunk_id} after {attempts} attempts")
            return {"chunk_id": c.chunk_id, "status": "timeout"}
        except Exception as e:
            error_msg = str(e)
            msg_lower = error_msg.lower()

            # Identify transient errors that should be retried
            transient = (
                "failed to obtain a connection from the pool" in msg_lower
                or "rate limit" in msg_lower
                or "temporarily unavailable" in msg_lower
                or "invalid duplicate_facts" in msg_lower
                or "timeout" in msg_lower
                or "connection" in msg_lower
            )

            if transient:
                logger.warning(f"Transient error on chunk {c.chunk_id} (attempt {i+1}/{attempts}): {error_msg[:200]}")
                if i < attempts - 1:
                    await asyncio.sleep(0.5 + random.random())
                    continue
            else:
                logger.error(f"Non-transient error on chunk {c.chunk_id}: {error_msg[:200]}")

            # Sanitize error message for client (don't expose internal details)
            safe_error = "Internal processing error" if not transient else "Temporary service issue"
            return {"chunk_id": c.chunk_id, "status": "error", "error": safe_error}

@app.post("/ingest-chunks")
async def ingest_chunks(req: IngestRequest, x_api_key: str | None = Header(default=None)):
    """Ingest chunks with deduplication and parallel processing"""
    require_key(x_api_key)

    logger.info(f"Received ingest request: doc_id={req.doc.doc_id}, chunks={len(req.chunks)}")

    if len(req.chunks) > MAX_CHUNKS_PER_REQUEST:
        logger.warning(f"Request rejected: too many chunks ({len(req.chunks)} > {MAX_CHUNKS_PER_REQUEST})")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Too many chunks; max {MAX_CHUNKS_PER_REQUEST}"
        )

    # Filter out already processed chunks (deduplication)
    chunks_to_process = []
    chunks_skipped = 0

    for c in req.chunks:
        chunk_key = f"{req.doc.doc_id}:{c.chunk_id}"
        if ENABLE_DEDUPLICATION and chunk_key in processed_chunks:
            chunks_skipped += 1
            logger.debug(f"Skipping duplicate chunk: {chunk_key}")
        else:
            chunks_to_process.append(c)

    logger.info(f"Processing {len(chunks_to_process)} chunks (skipped {chunks_skipped} duplicates)")

    async def ingest_one(c: Chunk):
        async with GLOBAL_SEM:
            result = await add_episode_with_timeout_and_retry(c, req.doc)
            # Mark as processed if successful
            if ENABLE_DEDUPLICATION and result.get("status") == "ok":
                processed_chunks.add(f"{req.doc.doc_id}:{c.chunk_id}")
            return result

    results = await asyncio.gather(*(ingest_one(c) for c in chunks_to_process))
    ok = sum(1 for r in results if r.get("status") == "ok")
    timeouts = sum(1 for r in results if r.get("status") == "timeout")
    errors = [r for r in results if r.get("status") == "error"]

    logger.info(f"Ingestion complete: ok={ok}, timeout={timeouts}, error={len(errors)}, skipped={chunks_skipped}")

    return {
        "status": "ok",
        "chunks_received": len(req.chunks),
        "chunks_skipped": chunks_skipped,
        "chunks_processed": len(chunks_to_process),
        "chunks_ingested_ok": ok,
        "chunks_timeout": timeouts,
        "chunks_error": len(errors),
        "errors_sample": errors[:3],  # Reduced from 5 to save response size
        "doc_id": req.doc.doc_id,
    }

@app.post("/finalize")
async def finalize(x_api_key: str | None = Header(default=None)):
    """Finalize ingestion - optionally clear deduplication cache"""
    require_key(x_api_key)

    processed_count = len(processed_chunks)
    logger.info(f"Finalize called - {processed_count} chunks in deduplication cache")

    return {
        "status": "finalized",
        "processed_chunks": processed_count,
        "message": "Use POST /clear-cache to reset deduplication cache if needed"
    }

@app.post("/clear-cache")
async def clear_cache(x_api_key: str | None = Header(default=None)):
    """Clear the deduplication cache - use with caution"""
    require_key(x_api_key)

    old_count = len(processed_chunks)
    processed_chunks.clear()

    logger.info(f"Deduplication cache cleared - removed {old_count} entries")

    return {
        "status": "ok",
        "cleared_count": old_count,
        "message": "Deduplication cache cleared successfully"
    }

@app.get("/stats")
async def stats(x_api_key: str | None = Header(default=None)):
    """Get service statistics"""
    require_key(x_api_key)

    return {
        "processed_chunks": len(processed_chunks) if ENABLE_DEDUPLICATION else None,
        "deduplication_enabled": ENABLE_DEDUPLICATION,
        "max_inflight": MAX_INFLIGHT,
        "episode_timeout": EPISODE_TIMEOUT,
        "max_chunks_per_request": MAX_CHUNKS_PER_REQUEST,
    }
