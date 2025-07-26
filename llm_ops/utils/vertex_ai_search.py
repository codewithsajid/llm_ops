import vertexai
from vertexai.generative_models import GenerativeModel, Tool
from dataclasses import dataclass
from typing import List
import os
from datetime import datetime
import re

# Configuration
PROJECT_ID = "apt-terrain-460010-j4"
MODEL_NAME = "gemini-2.5-flash"
OUTPUT_FILE = "llm_ops/utils/vai_search_out.md"

@dataclass
class SearchResult:
    query: str
    response: str
    sources: List[str]
    links: List[str]

class VertexSearch:
    """Optimal multimodal web search - simple and clever"""
    
    def __init__(self):
        vertexai.init(project=PROJECT_ID, location="us-central1")
        search_tool = Tool.from_dict({"google_search": {}})
        
        # The clever part: ask for citations upfront in the search query
        self.model = GenerativeModel(MODEL_NAME, tools=[search_tool])
    
    def search(self, query: str) -> SearchResult:
        """Search with automatic citations by modifying the query"""
        
        # Clever approach: modify query to request citations
        enhanced_query = f"""
{query}

Please provide a comprehensive answer and include citation numbers [1], [2], [3] etc. 
after each factual statement to reference your sources.
"""
        
        try:
            response = self.model.generate_content(enhanced_query)
            sources, links = [], []
            
            # Extract grounding data
            if (hasattr(response, 'candidates') and response.candidates and 
                hasattr(response.candidates[0], 'grounding_metadata')):
                
                metadata = response.candidates[0].grounding_metadata
                
                # Get sources
                if hasattr(metadata, 'grounding_supports'):
                    sources = [support.segment.text 
                              for support in metadata.grounding_supports 
                              if hasattr(support, 'segment')]
                
                # Get links
                if (hasattr(metadata, 'search_entry_point') and 
                    hasattr(metadata.search_entry_point, 'rendered_content')):
                    
                    html = metadata.search_entry_point.rendered_content
                    matches = re.findall(r'href="([^"]*)"[^>]*>([^<]*)</a>', html)
                    
                    links = [f"[{title.strip() or 'Resource'}]({url})" 
                            for url, title in matches 
                            if url.startswith('http') and 'grounding-api-redirect' in url]
            
            return SearchResult(query, response.text, sources, links)
            
        except Exception as e:
            return SearchResult(query, f"Search failed: {e}", [], [])
    
    def save(self, result: SearchResult):
        """Save with clean formatting"""
        content = f"""
## {result.query}
*{datetime.now().strftime("%Y-%m-%d %H:%M")}*

{result.response}

**Sources:**
"""
        
        # Simple enumeration of sources
        for i, source in enumerate(result.sources, 1):
            preview = source[:150] + "..." if len(source) > 150 else source
            content += f"[{i}] {preview}\n\n"
        
        # Add web resources
        if result.links:
            content += "**Web Resources:**\n"
            for link in result.links[:6]:  # Top 6 links
                content += f"• {link}\n"
        
        content += "\n---\n\n"
        
        # Write to file
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(content)

def main():
    search = VertexSearch()
    
    queries = [
        "Latest reinforcement learning research with video or charts/diagrams explanations, and code examples"
    ]
    
    for query in queries:
        result = search.search(query)
        search.save(result)

if __name__ == "__main__":
    main()

