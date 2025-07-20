# LLM_OPS – Sprint-0 Docs-MCP

Minimal project that:

1. Ingests Markdown / PDF / TXT into Weaviate with E5-base embeddings  
2. Exposes the chunks through an **MCP-style FastAPI** (`/docs`, `/search`, `/chunk/{id}`)  
3. Logs ingestion runs to **MLflow**

Runbook (after virtual-env & deps):

```bash
docker compose up -d               # start weaviate
python -m llm_ops.ingest.ingest_docs
uvicorn llm_ops.server:app --reload


Now:

    GET http://localhost:8000/docs lists documents

    POST http://localhost:8000/search body {"query_text":"what is RAG","top_k":3}


---

### `llm_ops/__init__.py`

```python
__all__ = ["config", "schemas", "weaviate_client"]