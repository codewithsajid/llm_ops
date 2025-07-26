# llm_ops/utils/web_search.py
from __future__ import annotations
from typing import List, Dict, Set, Tuple
from ddgs import DDGS              # DuckDuckGo Search API
from bs4 import BeautifulSoup
import requests, datetime, rich, re
import trafilatura
from .query_generator import LLMQueryGenerator
import concurrent.futures
from urllib.parse import urlparse
import logging


class WebSearcher:
    """
    Very thin wrapper around DuckDuckGo (via ddgs) that:
    • lets an LLM generate multiple queries
    • fetches & cleans page text with Trafilatura
    • returns at most `max_results` unique snippets
    """

    def __init__(self, generator_fn, max_results: int = 5, timeout: float = 10.0):
        self.max_results = max_results  # Increased from 3 to 5
        self.timeout = timeout  # Increased from 8.0 to 10.0
        self.query_generator = LLMQueryGenerator(generator_fn)

        try:
            self.ddgs = DDGS()
            rich.print("[green]Web search initialised[/]")
        except Exception as e:
            rich.print(f"[red]DDGS init failed:[/] {e}")
            self.ddgs = None

    # ---------- helpers -------------------------------------------------

    _TAG_RE = re.compile(r"<[^>]+>")

    def _html2text(self, html: str, max_chars: int = 3000) -> str:
        """Enhanced HTML-to-text conversion with better handling of technical content."""
        soup = BeautifulSoup(html, "html.parser")
        
        # First, try to extract the main content area
        main_content = soup.find('article') or soup.find('main') or soup.find('div', class_='content')
        
        # If we found a main content area, use that; otherwise use the whole page
        content_to_process = main_content if main_content else soup
        
        # Prioritize paragraphs, lists, and code blocks - common in technical content
        extracted_texts = []
        
        # Get all paragraphs (regular text)
        for p in content_to_process.find_all('p'):
            extracted_texts.append(p.get_text(" ", strip=True))
        
        # Get all list items (often important in technical articles)
        for li in content_to_process.find_all('li'):
            extracted_texts.append("• " + li.get_text(" ", strip=True))
        
        # Get all code blocks (critical for programming topics)
        for code in content_to_process.find_all(['code', 'pre']):
            extracted_texts.append("[CODE] " + code.get_text(" ", strip=True))
        
        # Get all headers (for structure)
        for h in content_to_process.find_all(['h1', 'h2', 'h3']):
            extracted_texts.append(h.get_text(" ", strip=True).upper())
        
        # Join everything with newlines for better structure
        result = "\n".join(t for t in extracted_texts if t.strip())
        
        # If we didn't get much, fall back to whole text
        if len(result) < 500:
            result = soup.get_text(" ", strip=True)
            
        return result[:max_chars]

    def _extract_domain(self, url: str) -> str:
        """Extract the domain from a URL."""
        try:
            return urlparse(url).netloc
        except:
            return url.split('/')[2] if '://' in url else url.split('/')[0]

    def _rank_results(self, results: List[Dict[str, str]], question: str) -> List[Dict[str, str]]:
        """Rank results by relevance and ensure domain diversity."""
        # Group by domain
        by_domain = {}
        for result in results:
            domain = result.get("domain", self._extract_domain(result["url"]))
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(result)
        
        # Take the best result from each domain first
        diverse_results = []
        for domain, domain_results in by_domain.items():
            # Sort domain results by text length (proxy for richness)
            domain_results.sort(key=lambda x: len(x["text"]), reverse=True)
            diverse_results.append(domain_results[0])
        
        # Fill remaining slots with best overall results
        all_remaining = [r for r in results if r not in diverse_results]
        all_remaining.sort(key=lambda x: len(x["text"]), reverse=True)
        
        # Combine with priority for diverse results
        return diverse_results + all_remaining

    # ---------- public API ----------------------------------------------

    def search(self, question: str) -> List[Dict[str, str]]:
        if not self.ddgs:
            rich.print("[red]DuckDuckGo Search (DDGS) not initialized![/]")
            return []

        queries = self.query_generator.get_search_queries(question)
        rich.print("[cyan]LLM-generated queries:[/]")
        for q in queries:
            rich.print(f"• {q}")

        collected, seen_urls = [], set()
        
        # Execute all queries in parallel with threading
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as executor:
            future_to_query = {executor.submit(self._execute_single_query, query, seen_urls): query for query in queries}
            
            for future in concurrent.futures.as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    results, new_seen_urls = future.result()
                    collected.extend(results)
                    seen_urls.update(new_seen_urls)
                    if len(collected) >= self.max_results:
                        break
                except Exception as e:
                    rich.print(f"[yellow]Search failed for query '{query}': {e}[/]")
        
        # Sort by relevance score (if available) or recency
        return self._rank_results(collected, question)[:self.max_results]
    
    def _execute_single_query(self, query: str, seen_urls: Set[str]) -> Tuple[List[Dict[str, str]], Set[str]]:
        """Execute a single search query and return results and updated seen_urls."""
        results = []
        local_seen = set()
        
        try:
            for hit in self.ddgs.text(query, max_results=5):
                url = hit.get("href") or hit.get("link")
                if not url or url in seen_urls or url in local_seen:
                    continue
                    
                try:
                    r = requests.get(url, timeout=self.timeout, headers={"User-Agent": "Mozilla/5.0"})
                    if r.ok:
                        # Rest of your existing extraction logic...
                        text = self._extract_development_info(r.text, query) or trafilatura.extract(r.text) or self._html2text(r.text)
                        if len(text.strip()) >= 400:
                            results.append({
                                "url": url,
                                "title": hit.get("title", "").strip(),
                                "text": text.strip()[:3000],  # Increased from 2000
                                "query": query,
                                "fetched": datetime.datetime.utcnow().isoformat(),
                                "domain": self._extract_domain(url),
                            })
                            local_seen.add(url)
                            rich.print(f"[green]✓ Retrieved:[/] {url}")
                except requests.RequestException as e:
                    rich.print(f"[yellow]Request failed {url}: {e}[/]")
        except Exception as e:
            rich.print(f"[yellow]Search failed for query '{query}': {e}[/]")
            
        return results, local_seen

    def _extract_development_info(self, html, query):
        """Extract recent developments specifically."""
        soup = BeautifulSoup(html, "html.parser")
        
        # Find sections with relevant headings
        development_sections = []
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4']):
            heading_text = heading.get_text().lower()
            if any(term in heading_text for term in ['recent', 'latest', 'new', 'advance', 'breakthrough', 'future', '2023', '2024']):
                # Collect content under this heading
                section_content = []
                for elem in heading.find_next_siblings():
                    if elem.name in ['h1', 'h2', 'h3', 'h4']: 
                        break
                    if elem.name in ['p', 'li', 'ul', 'ol']:
                        section_content.append(elem.get_text())
                
                if section_content:
                    development_sections.append(
                        f"SECTION: {heading.get_text()}\n" + 
                        "\n".join(section_content)
                    )
        
        # If we found development sections, use those
        if development_sections:
            return "\n\n".join(development_sections)[:3000]
        
        # Otherwise fall back to regular extraction
        return None
    
    def _extract_content(self, url, response_text, query):
        """Enhanced content extraction with better error handling."""
        # First attempt: Use trafilatura for clean extraction
        try:
            extracted = trafilatura.extract(
                response_text, 
                favor_recall=True,
                include_comments=False,
                include_tables=True,
                output_format='text'
            )
            if extracted and len(extracted.strip()) >= 400:
                return extracted.strip()[:3000]  # Success case
        except Exception as e:
            logging.debug(f"Trafilatura extraction failed for {url}: {e}")
        
        # Second attempt: Try with more lenient settings
        try:
            extracted = trafilatura.extract(
                response_text,
                favor_recall=True,
                include_images=False,
                include_links=False,
                no_fallback=False,  # Important: Allow fallbacks
                target_language="en"  # Optional: Focus on English content
            )
            if extracted and len(extracted.strip()) >= 400:
                return extracted.strip()[:3000]
        except Exception as e:
            logging.debug(f"Lenient trafilatura extraction failed for {url}: {e}")
        
        # Final fallback: Use BeautifulSoup directly
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response_text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()
            
            # Get text and normalize whitespace
            text = soup.get_text(separator=' ', strip=True)
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            if len(text) >= 400:
                return text[:3000]
        except Exception as e:
            logging.debug(f"BeautifulSoup extraction failed for {url}: {e}")
        
        # If all methods fail, return empty string rather than None
        return ""

