from datetime import datetime
from pydantic import BaseModel


class DocumentSummary(BaseModel):
    doc_id: str
    title: str | None = None
    n_chunks: int
    source_uri: str
    mime_type: str
    created_at: datetime | None = None


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    section_title: str | None = None
    start_char: int | None = None
    end_char: int | None = None


class ChunkHit(Chunk):
    score: float
