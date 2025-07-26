# llm_ops/agents/rag_agent.py
"""
Light‑weight *Agentic‑RAG* core used by rag_chatbot.py.

Key improvements vs. the previous version
-----------------------------------------
✔  All tunables live in one Config dataclass – no more magic numbers.  
✔  `Retriever`, `LLM`, `WebSearcher` expressed as *Protocols* → true DI.  
✔  Evaluation prompt returns JSON → parsing is robust.  
✔  Confidence gating: web search only if (sufficient=False ∧ confidence ≤ th).  
✔  Deduplication by hash of text; keeps order: web‑first if judged sufficient.  
✔  Token budget trimming (tiktoken) – avoids blown context windows.  
✔  Thorough logging hooks (use `rich` in CLI, standard logging in production).  
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Protocol, Sequence, Tuple
import concurrent.futures

import tiktoken
from ..utils.web_search import WebSearcher
#from ..utils.google_ai_search import GoogleAIModeSearcher
import rich

enc = tiktoken.get_encoding("cl100k_base")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------#
# Public data‑structures                                                      #
# ---------------------------------------------------------------------------#

@dataclass(frozen=True)
class SearchStep:
    query: str
    hits: List[Passage]
    scores: List[float]
    source: str           # "local" | "web"
    sufficient: bool
    confidence: float

@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    meta: Dict[str, str]   # arbitrary metadata: doc_id / uri / title / source

# ---------------------------------------------------------------------------#
# Strategy protocols                                                          #
# ---------------------------------------------------------------------------#


# ---------------------------------------------------------------------------#
# Config                                                                      #
# ---------------------------------------------------------------------------#

@dataclass
class AgentCfg:
    # retrieval
    top_k_keep: int = 6          # max passages kept in context
    token_budget: int = 1800     # max tokens across passages
    # decision
    suff_conf_th: float = 0.50   # keep web only if confidence ≥ th
    # misc
    log_prompts: bool = False


# ---------------------------------------------------------------------------#
# The agent                                                                   #
# ---------------------------------------------------------------------------#

class RAGAgent:
    def __init__(
        self,
        retriever,
        generator,
        cfg: AgentCfg | None = None,
        *,
        use_web: bool = False,
        use_google_ai: bool = False,  # New parameter
    ):
        self.retriever = retriever
        self.llm = generator
        self.cfg = cfg or AgentCfg()
        self.use_web = use_web  # Store the parameter
        
        # Choose web search strategy
        if use_google_ai:
            self.web_searcher = GoogleAIModeSearcher()
            rich.print("[green]Using Google AI Mode for web search[/]")
        elif use_web:
            self.web_searcher = WebSearcher(generator_fn=generator)
            rich.print("[green]Using DuckDuckGo web search[/]")
        else:
            self.web_searcher = None

    # ------------------------------------------------------------------ public
    def search_and_reflect(
        self, question: str
    ) -> Tuple[List[Passage], List[SearchStep]]:
        """Search knowledge base, web if necessary, and reflect on sufficiency."""
        trace: List[SearchStep] = []
        all_passages = []
        
        # Start web search early if enabled (will run in parallel)
        web_future = None
        if self.use_web:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                web_future = executor.submit(self.web_searcher.search, question)
        
        # Perform knowledge base search
        kb_hits, kb_scores = self.retriever(question)
        kb_sufficient, kb_conf = self._judge(question, kb_hits, "local")
        trace.append(
            SearchStep(
                query=question,
                hits=kb_hits,
                scores=kb_scores,
                source="local",
                sufficient=kb_sufficient,
                confidence=kb_conf,
            )
        )

        # 2) Optional web phase ----------------------------------------------
        if not kb_sufficient and self.web_searcher:
            if web_future:
                try:
                    web_hits_raw = web_future.result()
                    web_hits = [
                        Passage(
                            id = h.get("url", ""),               # or a proper hash
                            text = h["text"],
                            meta = {"uri": h["url"], "title": h.get("title",""), "source": "web"},
                        )
                        for h in web_hits_raw
                    ]
                    web_scores = [1.0] * len(web_hits)
                    web_sufficient, web_conf = self._judge(question, web_hits, "web")
                    trace.append(
                        SearchStep(
                            query=question,
                            hits=web_hits,
                            scores=web_scores,
                            source="web",
                            sufficient=web_sufficient,
                            confidence=web_conf,
                        )
                    )
                    
                    # MODIFIED: Intelligently integrate web results
                    if web_sufficient or web_conf >= 0.4:
                        # Check if this is a recency-focused question
                        is_recency_focused = any(term in question.lower() for term in ["latest", "recent", "new", "current", "2023", "2024"])
                        
                        if is_recency_focused:
                            # For recency questions, prioritize web results more heavily
                            hits = web_hits + kb_hits  # Web first
                        else:
                            # For other questions, balance between KB and web
                            # Take half from each source (or all if one source has fewer)
                            kb_quota = min(len(kb_hits), self.cfg.top_k_keep // 2)
                            web_quota = min(len(web_hits), self.cfg.top_k_keep - kb_quota)
                            
                            hits = web_hits[:web_quota] + kb_hits[:kb_quota]
                    else:
                        hits = kb_hits
                except Exception as e:
                    logging.error(f"Web search failed: {e}")
                    hits = kb_hits
            else:
                hits = kb_hits
        else:
            hits = kb_hits

        # 3) Dedup & trim -----------------------------------------------------
        final_hits = self._dedup(hits)
        final_hits = self._trim_tokens(final_hits)

        return final_hits, trace

    # ------------- judging ---------------------------------------------------
    def _judge(self, question: str, hits: List[Passage], src: str) -> Tuple[bool, float]:
        """Enhanced judging that focuses on relevance and recency when appropriate."""
        if not hits:
            return False, 0.0

        # Google AI Mode responses are typically comprehensive
        if src == "google_ai" and hits:
            # Check if response length indicates a substantial answer
            total_length = sum(len(p.text) for p in hits)
            if total_length > 500:  # Google AI typically provides detailed responses
                return True, 0.9  # High confidence for Google AI responses
        
        # Determine if this is a recency-focused question
        is_recency_focused = any(term in question.lower() for term in ["latest", "recent", "new", "current", "2023", "2024"])
        
        # Determine if this is a technical/domain-specific question
        technical_domains = ["reinforcement learning", "machine learning", "deep learning", "neural network", "algorithm"]
        is_technical = any(term in question.lower() for term in technical_domains)
        
        # Customize prompt based on question type
        judge_instructions = "Your task is to determine if the following text snippets are relevant and likely to help answer the user's question."
        
        if is_recency_focused:
            judge_instructions += " Pay special attention to whether the content contains RECENT information."
        
        if is_technical:
            judge_instructions += " This is a technical question, so evaluate if the content contains specific technical details, not just general information."
        
        # Increase snippet length to give the judge more context
        snippets = "\n".join(f"- {p.text[:800].replace(chr(10), ' ')}…" for p in hits[:3])
        
        prompt = (
            f"You are a relevance judge. {judge_instructions}\n\n"
            f"User Question: \"{question}\"\n\n"
            f"Text Snippets from source '{src}':\n{snippets}\n\n"
            "Based on the snippets, provide an assessment in JSON format with the following fields:\n"
            "- reasoning: Your analysis of the content relevance\n"
            "- sufficient: YES if the content provides a useful answer, NO otherwise\n"
            "- confidence: A number from 0-100 indicating your confidence\n"
            "- recency: (Only if applicable) YES if the content appears to be recent, NO if outdated\n\n"
            "Example: {\"reasoning\": \"The snippets discuss recent advancements in deep RL, which is directly related to the user's question.\", \"sufficient\": \"YES\", \"confidence\": 85, \"recency\": \"YES\"}"
        )

        if self.cfg.log_prompts:
            log.debug("Judge‑prompt:\n%s", prompt)

        raw, _ = self.llm(prompt, temperature=0.1, max_new_tokens=128) # Increased tokens for better reasoning
        raw = raw[len(prompt):].strip()  # Remove the prompt part from the response
        
        try:
            # More robust JSON parsing - find the outermost JSON object
            match = re.search(r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})', raw, re.DOTALL)
            if match:
                json_str = match.group(0)
                data = json.loads(json_str)
                sufficient = str(data.get("sufficient", "")).upper().startswith("Y")
                # Confidence is expected out of 100
                confidence = max(0.0, min(float(data.get("confidence", 0)) / 100.0, 1.0))
            else:
                # If no JSON object found, try a more lenient approach
                sufficient = "YES" in raw.upper() and "SUFFICIENT" in raw.upper()
                confidence_match = re.search(r'confidence[:\s]+(\d+)', raw, re.IGNORECASE)
                confidence = float(confidence_match.group(1))/100.0 if confidence_match else 0.4
        except Exception as e:
            logging.warning(f"Failed to parse judge response: {e}")
            # Set a reasonable default for web content - don't completely discard it
            sufficient, confidence = False, 0.4  # Default to 40% confidence

        # Boost confidence for Google AI responses
        if src == "google_ai" and sufficient:
            confidence = min(confidence + 0.2, 1.0)

        rich_msg = f"{src.upper()} judge → {sufficient} ({confidence:.0%})"
        try:
            rich.print(f"[green]{rich_msg}[/]")
        except Exception:
            log.info(rich_msg)

        return sufficient, confidence

    # ------------- utils -----------------------------------------------------
    def _dedup(self, hits: List[Passage]) -> List[Passage]:
        seen, deduped = set(), []
        for p in hits:
            if p.text not in seen:
                deduped.append(p)
                seen.add(p.text)
        return deduped


    def _trim_tokens(self, hits: List[Passage]) -> List[Passage]:
        kept, used = [], 0
        for p in hits[: self.cfg.top_k_keep]:
            t = len(enc.encode(p.text))
            if used + t > self.cfg.token_budget:
                break
            kept.append(p)
            used += t
        return kept

