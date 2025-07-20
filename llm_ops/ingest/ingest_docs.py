from __future__ import annotations

import torch
import os, re, uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable, List

import mlflow
from sentence_transformers import SentenceTransformer
from llm_ops.weaviate_client import get_client

# ────────────────────────────  NEW LOADER STACK  ───────────────────────────────
from tika import detector
from unstructured.partition.auto import partition
import fitz                               # PyMuPDF
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

_ocr = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn")

_NOISE_RE = re.compile(
    r"(lecture\s+\d+\b.*)|(\b\d{1,2}\.\d{1,2}\b)|(^\s*\d+\s*$)",
    flags=re.I | re.M,
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
        return [_clean(t) for t in coalesce(narr)]

    
    
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

def upsert(path: Path):
    client = get_client()
    batch = client.batch

    for doc_part in load_elements(path):
        for chunk in chunk_text(doc_part):
            vec = get_embedder().encode(chunk, show_progress_bar=False)
            meta = {
                "doc_id": path.stem,
                "chunk_id": str(uuid.uuid4()),
                "text": chunk,
                "source_uri": str(path),
                "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            }
            batch.add_data_object(meta, "DocChunk", vector=vec)

    batch.flush()

# ────────────────────────────────  MAIN SCRIPT  ────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

def ensure_schema():
    client = get_client()
    if any(cls["class"] == "DocChunk" for cls in client.schema.get().get("classes", [])):
        return
    client.schema.create_class(
        {
            "class": "DocChunk",
            "vectorizer": "none",
            "properties": [
                {"name": "doc_id",     "dataType": ["text"]},
                {"name": "chunk_id",   "dataType": ["text"]},
                {"name": "text",       "dataType": ["text"]},
                {"name": "source_uri", "dataType": ["text"]},
                {"name": "created_at", "dataType": ["date"]},
            ],
        }
    )

def main():
    from multiprocessing.dummy import Pool
    pool = Pool(min(8, os.cpu_count()))
    try:
        ensure_schema()
        paths = [p for p in DATA_DIR.rglob("*") if p.is_file()]
        with mlflow.start_run(run_name="sprint0_ingest"):
            mlflow.log_param("n_files", len(paths))
            pool.map(upsert, paths)
            # create a fresh client in this (parent) process
            client = get_client()
            n_chunks = (
                client.query.aggregate("DocChunk")
                .with_meta_count()
                .do()["data"]["Aggregate"]["DocChunk"][0]["meta"]["count"]
            )
            mlflow.log_metric("n_chunks", n_chunks)
            print("✅ Ingest complete.")
    finally:
        pool.close()
        pool.join()

if __name__ == "__main__":
    import tracemalloc
    tracemalloc.start()

    main()