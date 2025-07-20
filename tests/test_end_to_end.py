import subprocess, time, httpx, os, pytest


@pytest.fixture(scope="session", autouse=True)
def start_server():
    """Spin up the API once for all tests (expects ingest already run)."""
    proc = subprocess.Popen(
        ["uvicorn", "llm_ops.server:app", "--port", "8111"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.environ["DOCS_MCP_URL"] = "http://localhost:8111"
    # wait for health
    for _ in range(20):
        try:
            httpx.get("http://localhost:8111/healthz").raise_for_status()
            break
        except Exception:
            time.sleep(0.5)
    yield
    proc.terminate()


def test_docs_endpoint():
    res = httpx.get("http://localhost:8111/docs").json()
    assert isinstance(res, list)


def test_search():
    res = httpx.post(
        "http://localhost:8111/search", json={"query_text": "language", "top_k": 1}
    ).json()
    assert res and "chunk_id" in res[0]
