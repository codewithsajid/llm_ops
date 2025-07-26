# LLM-Ops — Agentic RAG CoPilot (Cloud + Local)

LLM-Ops turns your loose folder of PDFs, HTML, and notes into a fully traceable,
search-savvy chatbot.  
It mixes **agentic Retrieval-Augmented Generation (RAG)**, a
**docs-MCP** micro-service, and optional **Vertex AI Search** so answers stay
grounded, fresh, and easy to audit.

---

## ✨ Why another RAG repo?

| Headache                    | How LLM-Ops fixes it                                                |
| --------------------------- | ------------------------------------------------------------------- |
| *“Does my KB already answer this?”* | The **RAGAgent** scores sufficiency before calling the LLM. |
| *Token explosions*          | Deduplication + token budgeting trim context to fit the LLM window. |
| *Out-of-date docs*          | Agent can fire live web / Vertex AI search when confidence is low. |
| *Messy configs*             | All settings live in **Pydantic** models – no magic numbers.        |
| *Observability black hole*  | **MLflow** logs prompts, answers, citations, and latency for every run. |

---

## 🏗️  What’s inside?

| Layer | Tech / Library | What it does |
|-------|----------------|--------------|
| **Ingest** | `unstructured`, Tika, Doctr OCR | Split & clean docs → text blocks |
| **Chunk + Embed** | SentenceTransformer (E5-base-v2) | 800-word chunks → dense vectors |
| **Store** | Weaviate v4 (named vectors + PQ) | Hybrid semantic search |
| **Agent** | `RAGAgent` (typed with `AgentCfg`) | Decides KB vs. web, trims tokens |
| **LLM** | Google **Gemma 7B-it** | Rewrites queries, judges hits, streams answer |
| **Web Search** | DuckDuckGo + LLM query gen | Pulls fresh text snippets |
| **Vertex AI (cloud)** | Gemini 2.5 Flash search tool | Richer results & automatic grounding |
| **docs-MCP** | FastAPI micro-service | List / search / fetch KB chunks |
| **Config + Schemas** | `pydantic` | Type-safe settings & data models |
| **Logs** | MLflow | Everything tracked for repro |

---

## 🔍  Data → Answer flow

```text
PDFs / HTML
     │             ┌─► Weaviate (vectors)
     ▼             │
Ingestion script   │
(split + embed)    │
     │             │
     ▼             │
RAGAgent ──► Sufficiency judge ─┬─► if ✅: use KB only
     │                          │
     │                          └─► if ❌: Web / Vertex search → merge hits
     ▼
Reranker  →  Gemma 7B-it  →  Answer  (+ numbered citations)
````

---

## 🚀  Quick start

```bash
# 1) install
git clone https://github.com/codewithsajid/llm_ops.git
cd llm_ops
pip install -r requirements.txt

# 2) ingest your docs (put files in data/raw/)
python -m llm_ops.ingest.ingest_docs

# 3) ask a question
python -m llm_ops.rag_chatbot \
  --question "Latest advances in reinforcement learning?" \
  --web --creative
```

### Handy flags

| Flag          | What it does                                |
| ------------- | ------------------------------------------- |
| `--web`       | Enable DuckDuckGo fallback                  |
| `--google-ai` | Use experimental Google AI overview scraper |
| `--creative`  | Higher temperature / top-p for LLM          |
| `--debug`     | Print full agent trace & prompts            |

---

## ⚙️  docs-MCP micro-service

```bash
uvicorn llm_ops.server:app --port 8000

# list docs
curl http://localhost:8000/docs

# semantic search
curl -X POST http://localhost:8000/search \
     -d '{"query_text":"policy gradient","top_k":3}'
```

Use it as a drop-in KB API for other apps or dashboards.

---

## 📂  Repo layout

```
llm_ops/
├─ agents/          # RAGAgent + config
├─ ingest/          # document loaders / embed pipeline
├─ llm/             # Gemma wrapper (generate + streaming)
├─ utils/           # web search, Vertex AI, helpers
├─ server.py        # docs-MCP FastAPI app
├─ rag_chatbot.py   # CLI entry-point
└─ tests/
```

---

## 🛣️  Roadmap

* **Multimodal ingestion** (images, video metadata)
* **Iterative agent planning** for multi-step reasoning
* Web UI with live answer streaming & citation inspection
* Plug-and-play hosted LLMs via Vertex AI / OpenAI endpoints

---

## 📜  License & citation

MIT License.
If this repo helps your work, please cite it:

```
@misc{llmops2025,
  author = {Sajid Ansari},
  title  = {LLM-Ops: Agentic RAG Chatbot Playground},
  year   = {2025},
  url    = {https://github.com/codewithsajid/llm_ops}
}
```
