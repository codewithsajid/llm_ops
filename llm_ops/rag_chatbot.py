#!/usr/bin/env python
import argparse, rich
from jinja2 import Template
from llm_ops.weaviate_client import get_client
from llm_ops.llm.gemma import generate
from llm_ops.mlflow_utils import log_run
from llm_ops.prompt_templates import QA_TEMPLATE, SYSTEM
from sentence_transformers import SentenceTransformer, CrossEncoder
from llm_ops.agents.rag_agent import RAGAgent 
from typing import List


embedder = SentenceTransformer("intfloat/e5-base-v2", device="cuda")
reranker = CrossEncoder("mixedbread-ai/mxbai-rerank-base-v1")

def retrieve(client, query: str, k: int = 12, max_dist: float = 0.75):
    """
    Retrieve relevant chunks using hybrid search and reranking.
    
    Args:
        client: Weaviate client
        query: User question 
        k: Number of initial candidates
        max_dist: Maximum semantic distance threshold (increased from 0.55 to 0.75)
    
    Returns:
        tuple: (valid_hits, rerank_scores) containing filtered chunks and their scores
    """
    try:
        # Get embedding for query
        vec = embedder.encode(query, convert_to_numpy=True)
        
        # Hybrid search combining vector and sparse retrieval with more parameters
        res = (
            client.query.get("DocChunk", [
                "chunk_id", 
                "text",
                "doc_id",
                "_additional {distance}"
            ])
            .with_hybrid(
                query=query,
                vector=vec.tolist(),
                alpha=0.75,  # Increased weight on keyword matching
                properties=["text"]  # Explicitly specify search field
            )
            .with_limit(k)
            .do()
        )

        hits = res["data"]["Get"].get("DocChunk", [])
        if not hits:
            rich.print("[yellow]Warning:[/] No results found in vector search")
            return [], []

        # Rerank results with more generous filtering
        scores = reranker.predict([[query, h["text"]] for h in hits])
        best = sorted(zip(scores, hits), key=lambda t: t[0], reverse=True)
        
        valid_hits = []
        valid_scores = []
        for score, hit in best:
            # More lenient distance check
            distance = hit.get("_additional", {}).get("distance")
            if distance is None or distance < max_dist:
                valid_hits.append(hit)
                valid_scores.append(float(score))
                if len(valid_hits) >= 4:
                    break
        
        if not valid_hits:
            rich.print("[yellow]Warning:[/] No results passed distance threshold")
            # Fall back to top 2 results regardless of distance
            valid_hits = [h for _, h in best[:2]]
            valid_scores = [float(s) for s, _ in best[:2]]
            
        return valid_hits, valid_scores

    except Exception as e:
        rich.print(f"[red]Error in retrieve():[/] {str(e)}")
        return [], []


def format_citations(hits: List[dict]) -> str:
    citations = []
    for i, hit in enumerate(hits, 1):
        if hit.get('source') == 'web':
            citations.append(f"[{i}] {hit.get('title', 'Web Source')} ({hit.get('source_uri', 'N/A')})")
        else:
            citations.append(f"[{i}] Local Knowledge Base: {hit.get('doc_id', 'N/A')}")
    return "\n".join(citations)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--web", action="store_true", help="Force web search")
    args = ap.parse_args()

    client = get_client()
    
    # Initialize agent with web search if enabled
    agent = RAGAgent(
        retriever=lambda q: retrieve(client, q),
        generator=generate,
        use_web=args.web
    )
    
    if args.web:
        rich.print("[cyan]Web search enabled[/]")
    
    # Get context and search steps through agent
    context, search_steps = agent.search_and_reflect(args.question)
    
    # Add citations before template rendering
    citations = format_citations(context)
    
    # Create template with citations included
    template = Template(QA_TEMPLATE)
    prompt = template.render(
        context=context,
        question=args.question,
        steps=search_steps,
        citations=citations
    )
    
    if args.debug:
        rich.print("\n[bold yellow]═══ Search Results ═══[/]")
        for step in search_steps:
            rich.print(f"\n[bold cyan]Source: {step.source}[/]")
            rich.print(f"Confidence: {step.confidence:.2%}")
            if step.source == 'web':
                for hit in step.hits:
                    rich.print(f"• {hit.get('title')}: {hit.get('source_uri')}")
    
    # Generate answer
    answer, stats = generate(prompt, t=0.5, max_new_tokens=512, do_sample=True)
    
    try:
        # Log run with enhanced metadata
        run_id = log_run(
            question=args.question,
            chunk_ids=[h.get("chunk_id") for h in context if "chunk_id" in h],
            answer=answer,
            stats=stats,
            prompt=prompt,
            rerank_scores=[s.scores for s in search_steps if s.source == "local"],
            search_steps=[{
                "query": s.query,
                "reasoning": s.reasoning,
                "source": s.source,
                "confidence": s.confidence
            } for s in search_steps],
            citations=citations
        )
    except Exception as e:
        rich.print(f"[red]Error logging to MLflow:[/] {str(e)}")
        run_id = "logging_failed"
    
    # Print results
    rich.print("\n[bold yellow]═══ Final Answer ═══[/]")
    rich.print(f"\n{answer}")
    
    if citations:
        rich.print("\n[bold cyan]Sources:[/]")
        rich.print(citations)
    
    rich.print(f"\n[magenta]MLflow Run:[/] {run_id}")
    rich.print("\n[blue]Performance:[/]")
    for key, value in stats.items():
        rich.print(f"• {key}: {value}")

if __name__ == "__main__":
    main()
