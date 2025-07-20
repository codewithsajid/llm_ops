from typing import List, Dict
from ddgs import DDGS  # Updated import
import requests
from bs4 import BeautifulSoup
import trafilatura
import rich
from .query_generator import LLMQueryGenerator

class WebSearcher:
    def __init__(self, generator_fn, max_results=3):
        self.max_results = max_results
        self.query_generator = LLMQueryGenerator(generator_fn)
        try:
            self.ddgs = DDGS()
            rich.print("[green]Web search initialized successfully[/]")
        except Exception as e:
            rich.print(f"[red]Failed to initialize web search:[/] {str(e)}")
            self.ddgs = None

    def search(self, question: str) -> List[Dict[str, str]]:
        """Perform web search with LLM-generated queries."""
        if not self.ddgs:
            return []

        try:
            # Get LLM-generated search queries
            search_queries = self.query_generator.get_search_queries(question)
            rich.print("[cyan]Generated search queries:[/]")
            for q in search_queries:
                rich.print(f"• {q}")

            results = []
            for query in search_queries:
                try:
                    for result in self.ddgs.text(query, max_results=5):
                        url = result.get('link')
                        if not url or any(r['url'] == url for r in results):
                            continue
                            
                        try:
                            response = requests.get(url, timeout=5)
                            text = trafilatura.extract(
                                response.text,
                                include_comments=False
                            )
                            if text:
                                results.append({
                                    'text': text[:800],
                                    'url': url,
                                    'title': result.get('title', ''),
                                    'source': 'web',
                                    'query': query  # Track which query found this
                                })
                                rich.print(f"[green]Found result via query: {query}[/]")
                                
                                if len(results) >= self.max_results:
                                    return results
                                    
                        except Exception as e:
                            continue
                            
                except Exception as e:
                    rich.print(f"[yellow]Query failed: {str(e)}[/]")
                    continue
                    
            return results
            
        except Exception as e:
            rich.print(f"[red]Web search error: {str(e)}[/]")
            return []