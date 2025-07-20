from dataclasses import dataclass
from typing import List, Tuple, Dict
from ..utils.web_search import WebSearcher
import rich

@dataclass
class SearchStep:
    query: str
    hits: List[dict]
    scores: List[float]
    reasoning: str
    source: str
    confidence: float = 0.0

class RAGAgent:
    def __init__(self, retriever, generator, use_web=False, max_steps=3):
        self.retriever = retriever
        self.generator = generator
        self.max_steps = max_steps
        # Pass the same generator to WebSearcher for query generation
        self.web_searcher = WebSearcher(generator_fn=generator) if use_web else None
        self.force_web = use_web
        
    def _evaluate_with_llm(self, question: str, t: float = 0.0) -> Tuple[str, float]:
        """Direct LLM evaluation of knowledge about the topic."""
        eval_prompt = f"""Can you provide a comprehensive answer about: {question}

Instructions:
1. Answer only YES or NO: Do you have sufficient knowledge?
2. On the next line, give a confidence score (just the number 0-100)
3. Then explain your reasoning

Example format:
YES
90
I have detailed knowledge about..."""
        
        try:
            response, _ = self.generator(eval_prompt, t=t)
            lines = response.strip().split('\n')
            
            answer = lines[0].strip().upper()
            if answer not in ['YES', 'NO']:
                answer = 'NO'
                
            try:
                confidence = float(lines[1].strip()) / 100
                confidence = max(0.0, min(1.0, confidence))
            except (IndexError, ValueError):
                confidence = 0.5
                
            return answer, confidence
            
        except Exception as e:
            rich.print(f"[red]Error in LLM evaluation:[/] {str(e)}")
            return 'NO', 0.0

    def _evaluate_results(self, question: str, hits: List[dict], 
                         source: str = "local", t: float = 0.0) -> Tuple[str, float]:
        """Evaluate search results and return decision with confidence."""
        eval_prompt = f"""Review these {source} search results for the question:
'{question}'

Results:
{[h['text'][:200] + '...' for h in hits]}

Instructions:
1. Answer only YES or NO: Are these results sufficient?
2. On the next line, give a confidence score (just the number 0-100)
3. Then explain your reasoning

Example format:
NO
75
The results are insufficient because..."""
        
        try:
            response, _ = self.generator(eval_prompt, t=t)
            lines = response.strip().split('\n')
            
            # Get YES/NO from first line
            answer = lines[0].strip().upper()
            if answer not in ['YES', 'NO']:
                answer = 'NO'  # Default to NO if unclear
                
            # Get confidence from second line
            try:
                confidence = float(lines[1].strip()) / 100
                confidence = max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
            except (IndexError, ValueError):
                confidence = 0.5  # Default confidence if parsing fails
                
            return answer, confidence
            
        except Exception as e:
            rich.print(f"[red]Error evaluating results:[/] {str(e)}")
            return 'NO', 0.0

    def search_and_reflect(self, question: str) -> Tuple[List[dict], List[SearchStep]]:
        steps = []
        context = []
        
        # Step 1: Local Search
        local_hits, local_scores = self.retriever(question)
        local_answer, local_conf = self._evaluate_results(
            question, local_hits, t=0.0
        )
        steps.append(SearchStep(
            question, local_hits, local_scores, local_answer, 'local', local_conf
        ))
        
        # Step 2: Web Search if enabled
        if self.web_searcher:
            try:
                web_results = self.web_searcher.search(question)
                if web_results:
                    web_hits = [{
                        'text': r['text'],
                        'source_uri': r['url'],
                        'title': r.get('title', ''),
                        'source': 'web'
                    } for r in web_results]
                    web_scores = [1.0] * len(web_hits)
                    web_answer, web_conf = self._evaluate_results(
                        question, web_hits, source="web", t=0.0
                    )
                    steps.append(SearchStep(
                        question, web_hits, web_scores, web_answer, 'web', web_conf
                    ))
                    # Add web results first for recency
                    context.extend(web_hits)
            except Exception as e:
                rich.print(f"[red]Web search failed:[/] {str(e)}")
    
        # Add local results
        context.extend(local_hits)
        return context, steps