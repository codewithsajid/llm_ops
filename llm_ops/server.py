from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from llm_ops.weaviate_client import get_client

app = FastAPI(title="docs-mcp", version="0.0.1")
wclient = get_client()


class SearchRequest(BaseModel):
    query_text: str = Field(..., examples=["what is retrieval augmented generation"])
    top_k: int = 5


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.get("/docs")
def list_documents():
    q = """
    {
      Aggregate {
        DocChunk {
          groupBy: ["doc_id"]
          meta {
            count
          }
        }
      }
    }
    """
    agg = wclient.query.raw(q)["data"]["Aggregate"]["DocChunk"]
    return [{"doc_id": d["groupedBy"]["value"], "n_chunks": d["meta"]["count"]} for d in agg]


@app.post("/search")
def search(req: SearchRequest):
    res = (
        wclient.query.get("DocChunk", ["doc_id", "chunk_id", "text"])
        .with_near_text({"concepts": [req.query_text]})
        .with_limit(req.top_k)
        .do()
    )
    return res["data"]["Get"]["DocChunk"]


@app.get("/chunk/{chunk_id}")
def get_chunk(chunk_id: str):
    res = (
        wclient.query.get("DocChunk", ["doc_id", "text"])
        .with_where({"path": ["chunk_id"], "operator": "Equal", "valueText": chunk_id})
        .do()
    )["data"]["Get"]["DocChunk"]
    if not res:
        raise HTTPException(404, "chunk_id not found")
    return res[0]
