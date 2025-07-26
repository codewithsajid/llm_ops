# query_generator.py
from typing import List
import rich
import re

class LLMQueryGenerator:
    """Generates clean, effective web search queries."""

    PROMPT_TEMPLATE = """
Generate web search queries for scraping internet to get information about: {question}

IMPORTANT: Return ONLY optimal number of queries (optimality based on question complexity), one per line.
DO NOT include any formatting, numbering, or explanations.
Include year markers (2024, 2025) in some queries.
"""

    def __init__(self, generator_fn):
        self.generate = generator_fn

    def get_search_queries(self, question: str) -> List[str]:
        """Generates and returns a clean list of search queries."""
        prompt = self.PROMPT_TEMPLATE.format(question=question)
        try:
            response, _ = self.generate(prompt, temperature=0.5, max_new_tokens=150)
            # Remove the prompt text if present
            clean_response = response.replace(prompt, "").strip()
            
            # Split response into lines and filter out instructions/headers.
            queries = []
            for line in clean_response.split('\n'):
                line = line.strip()
                # Skip if the line is too short or contains common instruction words.
                if (not line or len(line) < 5 or
                    any(word in line.lower() for word in ["queries:", "note:", "please", "here are", "sure,"])):
                    continue
                # Remove numbering, quotes, and bullets.
                clean_line = re.sub(r'^\d+\.\s*', '', line).strip().strip('"')
                if len(clean_line) > 5:
                    queries.append(clean_line)
            
            if not queries:
                rich.print("[yellow]Query generation failed. Using a fallback query.[/yellow]")
                return [question.replace("?", "").strip() + " latest developments"]
            
            rich.print(f"[green]Successfully parsed {len(queries)} search queries.[/green]")
            return queries

        except Exception as e:
            rich.print(f"[red]An error occurred during query generation: {e}[/red]")
            return [question.replace("?", "").strip()]

