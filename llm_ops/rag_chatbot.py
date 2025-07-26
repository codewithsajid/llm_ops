#!/usr/bin/env python
"""
CLI entry‑point: retrieval‑augmented chatbot driven by Gemma.

Improvements  ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
✓ Uses AgentCfg for tunables (no magic numbers inline)
✓ Embedding & rerankers are cached; retrieval is fully typed
✓ Token‑budget check before LLM call (avoids context over‑flow)
✓ Structured citations → always stable indices
✓ --creative & --debug flags forwarded to LLM generation args
✓ Safe Unicode logging / MLflow
"""

from __future__ import annotations
import argparse, functools, logging, os, rich
from typing import List, Tuple, Dict
import threading

from jinja2 import Template
from sentence_transformers import SentenceTransformer, CrossEncoder

from llm_ops.weaviate_client import get_client
from llm_ops.llm.gemma import generate, generate_stream  # Add generate_stream
from llm_ops.mlflow_utils import log_run
from llm_ops.prompt_templates import QA_TEMPLATE, WEB_FOCUSED_QA_TEMPLATE
from llm_ops.agents.rag_agent import RAGAgent, AgentCfg, Passage
from llm_ops.utils.web_search import WebSearcher
from weaviate.classes.query import MetadataQuery
import torch
import concurrent.futures
from functools import partial

# ---------------------------------------------------------------------------
# Globals / heavy objects (loaded once)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "intfloat/e5-base-v2"
RERANK_MODEL    = "mixedbread-ai/mxbai-rerank-base-v1"

embedder = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
reranker = CrossEncoder(RERANK_MODEL, device="cuda", max_length=512)

# Adjust based on your typical workload and memory constraints
EMBEDDING_CACHE_SIZE = os.getenv("EMBEDDING_CACHE_SIZE", "1024")
QUERY_REWRITE_CACHE_SIZE = os.getenv("QUERY_REWRITE_CACHE_SIZE", "256")

@functools.lru_cache(maxsize=int(EMBEDDING_CACHE_SIZE))
def embed_query(q: str):
    return embedder.encode(q, convert_to_numpy=True)


# ---------------------------------------------------------------------------
# NEW: Optimal Retrieval & Query Rewriting
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=int(QUERY_REWRITE_CACHE_SIZE))
def rewrite_query(question: str) -> List[str]:
    """Uses the LLM to generate multiple perspectives on the user's question."""
    prompt = f"""
Based on the following user question, generate 3 additional search queries that are different perspectives on the original question.
The queries should be suitable for a vector database search.
Return ONLY the queries, one per line. Do not add any other text.

Original Question: "{question}"
"""
    # Use a creative temperature for query generation
    response, _ = generate(prompt, temperature=0.6, max_new_tokens=128, do_sample=True)
    
    # Clean up the response
    queries = response.replace(prompt, "").strip().split('\n')
    # Filter out any empty lines or junk
    valid_queries = [q.strip() for q in queries if q.strip()]
    
    # Always include the original question for a baseline retrieval
    return [question] + valid_queries


def clear_gpu_memory():
    """Clear CUDA cache to prevent memory leaks."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def process_single_query(collection, query, vector, limit):
    """Process a single query and return results."""
    try:
        # Simple vector search with the "content" vector
        res = collection.query.near_vector(
            vector=vector,
            target_vector="content",
            limit=limit,
            return_metadata=MetadataQuery(distance=True)
        )
        return res.objects
    except Exception as e:
        # Fallback to simple hybrid search
        try:
            res = collection.query.hybrid(
                query=query,
                vector=vector,
                alpha=0.5,
                limit=limit
            )
            return res.objects
        except Exception as e2:
            logging.warning(f"Search failed for query '{query}': {e2}")
            return []

def retrieve(client, query: str, top_k: int = 20) -> Tuple[List[Passage], List[float]]:
    """
    Rewrites the query for multiple perspectives and performs multiple queries,
    followed by reranking.
    """
    # 1. Generate multiple query perspectives using the LLM
    rewritten_queries = rewrite_query(query)
    
    # Ensure original query is first in the list for priority processing
    if query != rewritten_queries[0]:
        rewritten_queries.remove(query)
        rewritten_queries.insert(0, query)
    
    # 2. Parallelize embedding generation
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(rewritten_queries))) as executor:
        # Map each query to its embedding in parallel
        future_to_query = {executor.submit(embed_query, q): q for q in rewritten_queries}
        query_vectors = []
        
        for future in concurrent.futures.as_completed(future_to_query):
            q = future_to_query[future]
            try:
                vector = future.result().tolist()
                query_vectors.append((q, vector))
            except Exception as e:
                logging.error(f"Error embedding query '{q}': {e}")
                # Still include original query without embedding
                query_vectors.append((q, None))
    
    # 2. Perform parallel queries for each perspective
    collection = client.collections.get("DocChunk")
    seen_ids = set()
    unique_hits = []
    
    # Use a smaller limit for each query
    per_query_limit = max(5, top_k // len(rewritten_queries))
    
    # Process queries in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(query_vectors))) as executor:
        # Create a partial function with fixed arguments
        processor = partial(process_single_query, collection, limit=per_query_limit)
        
        # Submit all tasks
        futures = [executor.submit(processor, q, vec) for q, vec in query_vectors]
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                results = future.result()
                # Deduplicate on-the-fly
                for hit in results:
                    chunk_id = hit.properties.get("chunk_id", "")
                    if chunk_id and chunk_id not in seen_ids:
                        seen_ids.add(chunk_id)
                        unique_hits.append(hit)
                        
                        # Early stopping if we've reached target
                        if len(unique_hits) >= top_k * 2:
                            break
            except Exception as e:
                logging.error(f"Error processing query results: {e}")
    
    candidate_hits = unique_hits[:top_k]  # Limit to top_k
    if not candidate_hits:
        return [], []

    # 3. Rerank with dynamic batching based on text length
    rerank_pairs = [[query, h.properties["text"]] for h in candidate_hits]
    
    # Estimate text length to optimize batching
    avg_length = sum(len(pair[1]) for pair in rerank_pairs) / len(rerank_pairs) if rerank_pairs else 0
    
    # Adjust batch size based on average length
    if avg_length > 1000:
        BATCH_SIZE = 8  # Smaller batches for very long texts
    elif avg_length > 500:
        BATCH_SIZE = 12  # Medium batches for moderate texts
    else:
        BATCH_SIZE = 24  # Larger batches for shorter texts
    
    # Process in optimized batch sizes
    all_scores = []
    for i in range(0, len(rerank_pairs), BATCH_SIZE):
        batch = rerank_pairs[i:i+BATCH_SIZE]
        batch_scores = reranker.predict(batch)
        all_scores.extend(batch_scores)
    
    # Combine scores with hits and sort
    ranked = sorted(zip(all_scores, candidate_hits), key=lambda t: t[0], reverse=True)
    
    # Clear GPU memory after intensive operations
    clear_gpu_memory()
    
    # 4. Format the final passages
    passages, rerank_scores = [], []
    seen = set()
    for sc, h in ranked:
        txt = h.properties["text"]
        if txt in seen:
            continue
        seen.add(txt)
        passages.append(
            Passage(
                id=h.properties["chunk_id"],
                text=txt,
                meta={
                    "doc_id": h.properties.get("doc_id", ""),
                    "uri":   h.properties.get("source_uri", ""),
                    "source": "kb"
                },
            )
        )
        rerank_scores.append(float(sc))
    return passages, rerank_scores


# ---------------------------------------------------------------------------
# Citation formatting
# ---------------------------------------------------------------------------

def format_citations(ctx: List[Passage]) -> str:
    out = []
    for i, p in enumerate(ctx, 1):
        meta = p.meta
        label = meta.get("title") or meta.get("doc_id") or meta.get("uri", "Local‑KB")
        snippet = p.text[:60].replace("\n", " ")
        out.append(f"[{i}] {label} — {snippet}…")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True, help="User question")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--web", action="store_true", help="Allow web fallback")
    ap.add_argument("--google-ai", action="store_true", help="Use Google AI Mode") # New flag
    ap.add_argument("--creative", action="store_true", help="Higher temperature / nucleus sampling")
    args = ap.parse_args()

    # ---------- RAG agent ----------
    client = get_client()
    agent = RAGAgent(
        retriever=lambda q: retrieve(client, q),
        generator=generate,
        use_web=args.web,
        use_google_ai=args.google_ai,  # Pass new parameter
        cfg=AgentCfg(
            top_k_keep=8,
            token_budget=2000,
            suff_conf_th=0.50,
            log_prompts=args.debug,
        ),
    )

    # ---------- fetch context ----------
    ctx, trace = agent.search_and_reflect(args.question)
    citations = format_citations(ctx)

    # pick the strict template if web mode is on
    tpl = WEB_FOCUSED_QA_TEMPLATE if args.web else QA_TEMPLATE
    prompt = Template(tpl).render(
        context=ctx,
        question=args.question,
        citations=citations,
    )

    gen_kwargs = dict(
        max_new_tokens=768,
        temperature=0.7 if args.creative else 0.0,
        top_p=0.9 if args.creative else 1.0,
        do_sample=args.creative,
    )
    
    # Print answer header before streaming starts
    rich.print("\n[bold yellow]═══ Answer ═══[/]\n")
    rich.print("[bold]Answer:[/] ", end="", flush=True)  # Add this line to provide a lead-in
    
    answer_chunks = []
    stats = None
    
    # Use streaming generation
    for chunk in generate_stream(prompt, **gen_kwargs):
        # The last yielded item will be the stats dictionary
        if isinstance(chunk, dict):
            stats = chunk
        else:
            rich.print(chunk, end="", flush=True)
            answer_chunks.append(chunk)
    
    # Complete answer for logging
    answer = "".join(answer_chunks)

    # ---------- logging ----------
    run_id = None
    try:
        # Single-call logging with all required parameters
        run_id = log_run(
            question=args.question,
            answer=answer,
            stats=stats,
            chunk_ids=[p.id for p in ctx],
            prompt=prompt,
            rerank_scores=[sc for step in trace if step.source == "kb" for sc in step.scores],
            search_steps=[
                {
                    "source": s.source,
                    "confidence": s.confidence,
                    "sufficient": s.sufficient,
                    "query": s.query,
                }
                for s in trace
            ],
            citations=citations,
        )
    except Exception as e:
        rich.print(f"[red]MLflow logging failed:[/] {e}")
        run_id = "logging_failed"

    if citations:
        rich.print("\n[bold cyan]Sources:[/]\n" + citations)

    rich.print(f"\n[magenta]MLflow run:[/] {run_id}")
    rich.print("[blue]Stats:[/]", stats)

    if args.debug:
        rich.print("\n[bold yellow]═══ Trace ═══[/]")
        for s in trace:
            rich.print(
                f"[cyan]{s.source.upper()}[/] sufficient={s.sufficient} "
                f"conf={s.confidence:.0%} passages={len(s.passages)}"
            )

# Preload/warmup models to eliminate first-query latency
def warmup_models():
    """Perform warmup inference to eliminate cold-start latency."""
    logging.info("Warming up models...")
    dummy_text = "This is a warmup query to initialize the models."
    # Warm up embedder
    _ = embedder.encode(dummy_text)
    # Warm up reranker
    _ = reranker.predict([["query", dummy_text]])
    # Warm up LLM if possible
    try:
        _, _ = generate("Hello, world!", max_new_tokens=10)
    except Exception as e:
        logging.warning(f"LLM warmup failed (non-critical): {e}")
    logging.info("Model warmup complete.")

# Call this at import time or in an if __name__ == "__main__" block
if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("RAG_LOG", "WARN"))
    warmup_models()  # Warm up models on startup
    main()
