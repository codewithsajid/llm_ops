from typing import List
import rich

class LLMQueryGenerator:
    """Uses LLM to generate effective search queries."""
    
    def __init__(self, generator_fn):
        self.generate = generator_fn
        
    def get_search_queries(self, question: str, n_queries: int = 3) -> List[str]:
        """Generate search queries using LLM."""
        prompt = f"""Given this question: "{question}"
Generate {n_queries} different web search queries that would help find recent, relevant information.
Focus on finding academic/research sources.

Requirements:
- Remove question words (what, how, etc.)
- Include year 2025 for recency
- Add domain-specific keywords
- Format: One query per line, no numbering

Example for "What are the latest NLP models?":
transformer architecture advances 2025 research papers
large language models comparison arxiv 2025
NLP model architectures github implementations 2025"""

        try:
            response, _ = self.generate(prompt)
            queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
            return queries[:n_queries]
            
        except Exception as e:
            rich.print(f"[red]Query generation failed:[/] {str(e)}")
            # Fallback to basic query
            return [f"{question.replace('?', '').strip()} 2025"]