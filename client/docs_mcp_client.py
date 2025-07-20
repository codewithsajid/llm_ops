import httpx, os, rich

BASE = os.getenv("DOCS_MCP_URL", "http://localhost:8000")


def _get(path: str):
    return httpx.get(f"{BASE}{path}").raise_for_status().json()


def _post(path: str, json):
    return httpx.post(f"{BASE}{path}", json=json).raise_for_status().json()


def list_documents():
    return _get("/docs")


def search(query: str, k: int = 5):
    return _post("/search", {"query_text": query, "top_k": k})


def get_chunk(chunk_id: str):
    return _get(f"/chunk/{chunk_id}")


if __name__ == "__main__":
    rich.print(list_documents())
    hits = search("language model", 2)
    for h in hits:
        rich.print(h["chunk_id"][:8], h["text"][:100], "…")
