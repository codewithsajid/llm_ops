from __future__ import annotations

import torch
import os, re, uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable, List

import mlflow
from sentence_transformers import SentenceTransformer
from llm_ops.weaviate_client import get_client
# Import v4 classes
from weaviate.classes.config import Property, DataType, Configure, VectorDistances

# ────────────────────────────  NEW LOADER STACK  ───────────────────────────────
from tika import detector
from unstructured.partition.auto import partition
import fitz                               # PyMuPDF
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

from collections import OrderedDict
from itertools import islice

BATCH_SIZE = 32          # Weaviate batch + embed batch
MIN_CHARS  = 150


_ocr = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn")

_NOISE_RE = re.compile(
    r"(?i)(lecture\s+\d+\b.*)|(\b\d{1,2}\.\d{1,2}\b)|(^\s*\d+\s*$)|(\bpolicy\s+value\s+function\b)",
    flags=re.M,
)


def _clean(txt: str) -> str:
    txt = _NOISE_RE.sub("", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def _ocr_pdf_page(page) -> str:
    pix = page.get_pixmap(dpi=300)
    docimg = DocumentFile.from_images(pix.samples)
    return _ocr(docimg).export_as_json()["pages"][0]["text"]

def _pdf_to_text(path: Path) -> str:
    with fitz.open(path) as doc:
        pages = []
        for pg in doc:
            text = pg.get_text().strip()
            pages.append(text if text else _ocr_pdf_page(pg))
        return "\n".join(pages)

def load_elements(path: Path) -> List[str]:
    mime = detector.from_file(str(path))
    if mime == "application/pdf":
        try:
            els = partition(filename=str(path), strategy="fast")
            return [_clean(el.text) for el in els if el.category in ("NarrativeText")]
        except Exception:
            return [_clean(_pdf_to_text(path))]
    else:
        els = partition(filename=str(path), strategy="fast")
    narr = [el for el in els if el.category in ("NarrativeText")]
    blocks = coalesce(narr, min_chars=MIN_CHARS)
    # ── deduplicate while preserving order ──
    uniq = list(OrderedDict.fromkeys(blocks))
    return [_clean(t) for t in uniq]

    
    
def coalesce(elements, min_chars=150):
    """
    Merge neighbouring elements until each block has at least `min_chars`.
    Preserves order; joins with space.
    """
    block, for_return = [], []
    for el in elements:
        txt = el.text.strip()
        if not txt:
            continue
        block.append(txt)
        if sum(len(t) for t in block) >= min_chars:
            for_return.append(" ".join(block))
            block = []
    if block:
        for_return.append(" ".join(block))
    return for_return


# ───────────────────────────────  CHUNK + EMBED  ───────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_MODEL = "intfloat/e5-base-v2"
# ─── one global placeholder; each thread builds lazily ───
_embedder = None

def get_embedder() -> SentenceTransformer:
    """
    Lazily initialise the SentenceTransformer **inside the thread**.
    Avoids CUDA re‑initialisation errors.
    """
    global _embedder
    if _embedder is None:                      # first call in this thread
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embedder = SentenceTransformer(EMBED_MODEL, device=device)
    return _embedder

def chunk_text(text: str) -> Iterable[str]:
    words = text.split()
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(words), step):
        yield " ".join(words[i : i + CHUNK_SIZE])

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """GPU‑batch the chunks to speed up."""
    embedder = get_embedder()
    vecs = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        vecs.extend(embedder.encode(batch, show_progress_bar=False))
    return vecs


def upsert(path: Path):
    client = get_client()
    chunks, metas = [], []

    for doc_part in load_elements(path):
        for chunk in chunk_text(doc_part):
            chunks.append(chunk)
            metas.append({
                "doc_id": path.stem,
                "chunk_id": str(uuid.uuid4()),
                "text": chunk,
                "source_uri": str(path),
                "created_at": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            })

    if not chunks:
        return  # nothing to insert

    vectors = embed_chunks(chunks)

    # FIX: Get the collection-specific batch manager to support named vectors
    collection = client.collections.get("DocChunk")
    with collection.batch.dynamic() as batch:
        for i in range(len(chunks)):
            batch.add_object(
                properties=metas[i],
                vector={"content": vectors[i]}  # Change 'vectors' to 'vector'
            )

# ────────────────────────────────  MAIN SCRIPT  ────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

def ensure_schema():
    client = get_client()
    
    # --- FIX: Force recreation of the schema ---
    # To ensure the correct multi-vector schema is applied, we will delete
    # the collection if it exists and then recreate it.
    if client.collections.exists("DocChunk"):
        print("Existing 'DocChunk' collection found. Deleting it to apply new schema...")
        client.collections.delete("DocChunk")
        print("Collection 'DocChunk' deleted.")

    # v4 schema creation with named vector support
    print("Creating 'DocChunk' collection with named vector support enabled.")
    client.collections.create(
        name="DocChunk",
        # FIX: Use a list of named vector configurations.
        # The .none() helper is for when you provide vectors manually.
        vectorizer_config=[
            Configure.NamedVectors.none(
                name="content",
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE,
                    quantizer=Configure.VectorIndex.Quantizer.pq(training_limit=50000)
                )
            )
        ],
        properties=[
            Property(name="doc_id", data_type=DataType.TEXT),
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="text", data_type=DataType.TEXT),
            Property(name="source_uri", data_type=DataType.TEXT),
            Property(name="created_at", data_type=DataType.DATE),
        ]
    )
    print("Collection created successfully.")

def main():
    from multiprocessing import Pool
    pool = Pool(min(4, os.cpu_count() // 2))   # 4 processes default

    try:
        ensure_schema()
        paths = [p for p in DATA_DIR.rglob("*") if p.is_file()]
        with mlflow.start_run(run_name="sprint0_ingest"):
            mlflow.log_param("n_files", len(paths))
            pool.map(upsert, paths)
            # create a fresh client in this (parent) process
            client = get_client()
            # v4 aggregate query
            response = client.collections.get("DocChunk").aggregate.over_all(total_count=True)
            n_chunks = response.total_count
            mlflow.log_metric("n_chunks", n_chunks)
            print("✅ Ingest complete.")
    finally:
        pool.close()
        pool.join()

if __name__ == "__main__":
    import tracemalloc
    tracemalloc.start()

    main()