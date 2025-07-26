import vertexai
from vertexai.generative_models import GenerativeModel, Tool
from vertexai.preview.generative_models import grounding
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import json

# --- Configuration ---
PROJECT_ID = "apt-terrain-460010-j4"
MODEL_NAME = "gemini-2.5-flash"

@dataclass
class SearchResult:
    """Clean data structure for search results."""
    text: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class VertexAISearchEngine:
    """Optimal Vertex AI Search implementation with clean separation of concerns."""
    
    def __init__(self, project_id: str, model_name: str, location: str = "us-central1"):
        self.project_id = project_id
        self.model_name = model_name
        self.location = location
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Vertex AI and create model with search capabilities."""
        try:
            vertexai.init(project=self.project_id, location=self.location)
            search_tool = self._create_search_tool()
            
            if search_tool:
                self.model = GenerativeModel(self.model_name, tools=[search_tool])
                print(f"✅ Search-enabled model initialized: {self.model_name}")
            else:
                self.model = GenerativeModel(self.model_name)
                print(f"⚠️  Model initialized without search: {self.model_name}")
                
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            raise
    
    def _create_search_tool(self) -> Optional[Tool]:
        """Create Google Search tool with optimal approach."""
        try:
            return Tool.from_dict({"google_search": {}})
        except Exception as e:
            print(f"Search tool creation failed: {e}")
            return None
    
    def search(self, query: str, extract_sources: bool = True) -> SearchResult:
        """
        Main search method - simple, clean interface.
        
        Args:
            query: Search query
            extract_sources: Whether to extract and process sources
        
        Returns:
            SearchResult with text, sources, and metadata
        """
        try:
            response = self.model.generate_content(query)
            
            sources = []
            metadata = {}
            
            if extract_sources and hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                    sources, metadata = self._extract_grounding_info(candidate.grounding_metadata)
            
            return SearchResult(
                text=response.text,
                sources=sources,
                metadata=metadata
            )
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return SearchResult(text="", sources=[], metadata={"error": str(e)})
    
    def _extract_grounding_info(self, grounding_metadata) -> tuple[List[Dict], Dict]:
        """Extract grounding information efficiently."""
        sources = []
        metadata = {
            'has_search_queries': False,
            'source_count': 0,
            'search_interface_available': False
        }
        
        # Extract text sources
        if hasattr(grounding_metadata, 'grounding_supports'):
            for i, support in enumerate(grounding_metadata.grounding_supports):
                source = {'index': i + 1}
                
                if hasattr(support, 'segment'):
                    source['text'] = support.segment.text
                    source['preview'] = support.segment.text[:200] + "..." if len(support.segment.text) > 200 else support.segment.text
                
                if hasattr(support, 'grounding_chunk_indices'):
                    source['chunk_indices'] = list(support.grounding_chunk_indices)
                
                sources.append(source)
            
            metadata['source_count'] = len(sources)
        
        # Check for search interface
        if hasattr(grounding_metadata, 'search_entry_point'):
            metadata['has_search_queries'] = True
            metadata['search_interface_available'] = True
            
            # Let the AI model analyze the search results instead of manual parsing
            if hasattr(grounding_metadata.search_entry_point, 'rendered_content'):
                metadata['raw_search_data'] = grounding_metadata.search_entry_point.rendered_content
        
        return sources, metadata

class SmartResultProcessor:
    """Use AI to process and categorize results instead of hardcoded rules."""
    
    def __init__(self, search_engine: VertexAISearchEngine):
        self.search_engine = search_engine
    
    def analyze_content_types(self, search_result: SearchResult) -> Dict[str, Any]:
        """Let AI analyze what types of content were found."""
        if not search_result.metadata.get('raw_search_data'):
            return {'content_types': [], 'analysis': 'No search data available'}
        
        analysis_prompt = f"""
        Analyze this search result data and identify the types of content found.
        
        Search Response: {search_result.text[:500]}...
        
        Categorize any multimedia content, research papers, code repositories, tutorials, etc.
        Provide a concise JSON summary with:
        - content_types: list of content types found
        - highlights: key findings
        - platforms: platforms/sources mentioned
        
        Keep it brief and practical.
        """
        
        try:
            # Use a simple model call for analysis
            analysis_model = GenerativeModel(self.search_engine.model_name)
            analysis_response = analysis_model.generate_content(analysis_prompt)
            
            # Try to extract JSON, fallback to text analysis
            try:
                import re
                json_match = re.search(r'\{.*\}', analysis_response.text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass
            
            return {
                'content_types': self._extract_content_types_simple(analysis_response.text),
                'analysis': analysis_response.text[:300]
            }
            
        except Exception as e:
            return {'error': f'Analysis failed: {e}'}
    
    def _extract_content_types_simple(self, text: str) -> List[str]:
        """Simple extraction as fallback."""
        content_indicators = ['video', 'image', 'audio', 'code', 'paper', 'tutorial', 'course', 'documentation']
        found_types = []
        text_lower = text.lower()
        
        for indicator in content_indicators:
            if indicator in text_lower:
                found_types.append(indicator)
        
        return found_types

class ResultDisplay:
    """Clean, adaptive result display."""
    
    @staticmethod
    def display_result(search_result: SearchResult, analysis: Optional[Dict] = None):
        """Display results in a clean, readable format."""
        
        print(f"\n{'='*60}")
        print("🔍 SEARCH RESULTS")
        print(f"{'='*60}")
        
        # Main response
        print(f"\n📄 RESPONSE:")
        print("-" * 40)
        print(search_result.text)
        
        # Sources
        if search_result.sources:
            print(f"\n📚 SOURCES ({len(search_result.sources)}):")
            print("-" * 40)
            for source in search_result.sources:
                print(f"{source['index']}. {source.get('preview', 'No preview available')}")
                if source.get('chunk_indices'):
                    print(f"   📊 References: {source['chunk_indices']}")
                print()
        
        # Analysis (if available)
        if analysis and 'content_types' in analysis:
            print(f"\n🎯 CONTENT ANALYSIS:")
            print("-" * 40)
            if analysis['content_types']:
                print(f"Content types found: {', '.join(analysis['content_types'])}")
            if analysis.get('highlights'):
                print(f"Key insights: {analysis['highlights']}")
            print()
        
        # Metadata
        metadata = search_result.metadata
        if metadata and metadata.get('source_count', 0) > 0:
            print(f"\n📊 METADATA:")
            print("-" * 40)
            print(f"Sources: {metadata['source_count']}")
            print(f"Search interface: {'Available' if metadata.get('search_interface_available') else 'Not available'}")

# --- Main Interface ---
class OptimalSearchSystem:
    """Main system interface - simple and powerful."""
    
    def __init__(self, project_id: str, model_name: str = "gemini-2.5-flash"):
        self.search_engine = VertexAISearchEngine(project_id, model_name)
        self.processor = SmartResultProcessor(self.search_engine)
        self.display = ResultDisplay()
    
    def search(self, query: str, analyze_content: bool = True, display_results: bool = True) -> SearchResult:
        """
        Main search interface.
        
        Args:
            query: What to search for
            analyze_content: Whether to analyze content types using AI
            display_results: Whether to display results automatically
        
        Returns:
            SearchResult object
        """
        # Perform search
        result = self.search_engine.search(query)
        
        # Analyze content if requested
        analysis = None
        if analyze_content and result.sources:
            analysis = self.processor.analyze_content_types(result)
        
        # Display if requested
        if display_results:
            self.display.display_result(result, analysis)
        
        return result
    
    def batch_search(self, queries: List[str]) -> List[SearchResult]:
        """Search multiple queries efficiently."""
        results = []
        for query in queries:
            print(f"\n🔍 Searching: {query}")
            result = self.search(query, display_results=True)
            results.append(result)
            print(f"\n{'-'*80}")
        return results

# --- Usage Example ---
if __name__ == "__main__":
    # Simple, clean usage
    search_system = OptimalSearchSystem(PROJECT_ID)
    
    # Single search
    result = search_system.search("What are the latest developments in Reinforcement Learning?")
    
    # Batch search
    queries = [
        "Machine learning tutorials with code examples",
        "Computer vision research papers 2024",
        "Deep learning frameworks comparison"
    ]
    
    batch_results = search_system.batch_search(queries)
    
    # Programmatic usage (no display)
    silent_result = search_system.search(
        "AI ethics guidelines", 
        analyze_content=True, 
        display_results=False
    )
    
    print(f"Found {len(silent_result.sources)} sources for AI ethics query")

