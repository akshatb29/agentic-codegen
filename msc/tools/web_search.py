# msc/tools/web_search.py
"""
Web Search Tool for Tech Stack and Best Practices Research
"""
import json
import urllib.request
import urllib.parse
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.panel import Panel

console = Console()

class WebSearchTool:
    """Tool for searching web for tech stack recommendations and best practices"""
    
    def __init__(self):
        self.search_endpoints = {
            "duckduckgo": "https://api.duckduckgo.com/",
            # Can add more APIs as needed
        }
    
    def search_tech_stack_recommendations(self, query: str, language: str = "python") -> Dict[str, Any]:
        """Search for tech stack recommendations"""
        console.print(f"🔍 Searching for tech stack recommendations: {query}", style="dim")
        
        search_queries = [
            f"{query} {language} best practices architecture",
            f"{query} {language} framework comparison 2024",
            f"{query} {language} tech stack recommendations",
            f"best {language} libraries for {query}",
            f"{query} {language} project structure patterns"
        ]
        
        results = []
        for search_query in search_queries[:3]:  # Limit to avoid rate limiting
            try:
                result = self._search_duckduckgo(search_query)
                if result:
                    results.append(result)
            except Exception as e:
                console.print(f"⚠️ Search failed for '{search_query}': {e}", style="yellow")
        
        return {
            "search_results": results,
            "recommendations": self._extract_recommendations(results, query, language),
            "search_performed": True
        }
    
    def search_architecture_patterns(self, app_type: str, language: str = "python") -> Dict[str, Any]:
        """Search for architecture patterns for specific app types"""
        console.print(f"🏗️ Searching architecture patterns for {app_type}", style="dim")
        
        search_queries = [
            f"{app_type} architecture patterns {language}",
            f"best practices {app_type} {language} design",
            f"{app_type} project structure {language}",
            f"scalable {app_type} architecture {language}"
        ]
        
        results = []
        for search_query in search_queries[:2]:
            try:
                result = self._search_duckduckgo(search_query)
                if result:
                    results.append(result)
            except Exception as e:
                console.print(f"⚠️ Architecture search failed: {e}", style="yellow")
        
        return {
            "architecture_results": results,
            "patterns": self._extract_architecture_patterns(results, app_type),
            "search_performed": True
        }
    
    def _search_duckduckgo(self, query: str) -> Optional[Dict[str, Any]]:
        """Search using DuckDuckGo instant answers API"""
        try:
            params = {
                "q": query,
                "format": "json",
                "pretty": "1",
                "skip_disambig": "1"
            }
            
            url = self.search_endpoints["duckduckgo"] + "?" + urllib.parse.urlencode(params)
            
            with urllib.request.urlopen(url, timeout=10) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    return {
                        "query": query,
                        "abstract": data.get("Abstract", ""),
                        "abstract_text": data.get("AbstractText", ""),
                        "related_topics": [topic.get("Text", "") for topic in data.get("RelatedTopics", [])[:5]],
                        "infobox": data.get("Infobox", {}),
                        "answer": data.get("Answer", "")
                    }
        except Exception as e:
            console.print(f"DuckDuckGo search error: {e}", style="red")
            return None
    
    def _extract_recommendations(self, search_results: List[Dict], query: str, language: str) -> Dict[str, Any]:
        """Extract tech stack recommendations from search results"""
        recommendations = {
            "frameworks": [],
            "libraries": [],
            "tools": [],
            "patterns": [],
            "reasoning": []
        }
        
        # Common tech stack mappings based on keywords
        tech_mappings = {
            "python": {
                "web": ["FastAPI", "Django", "Flask", "Starlette"],
                "data": ["pandas", "numpy", "matplotlib", "seaborn", "plotly"],
                "ml": ["scikit-learn", "tensorflow", "pytorch", "xgboost"],
                "api": ["FastAPI", "Flask-RESTful", "Django REST"],
                "database": ["SQLAlchemy", "Django ORM", "MongoDB", "PostgreSQL"],
                "testing": ["pytest", "unittest", "coverage"]
            },
            "javascript": {
                "web": ["React", "Vue.js", "Angular", "Next.js"],
                "backend": ["Node.js", "Express.js", "Nest.js"],
                "database": ["MongoDB", "PostgreSQL", "Redis"],
                "testing": ["Jest", "Mocha", "Cypress"]
            }
        }
        
        query_lower = query.lower()
        lang_mappings = tech_mappings.get(language, {})
        
        # Analyze query keywords
        for category, tools in lang_mappings.items():
            if any(keyword in query_lower for keyword in [category, *tools]):
                recommendations[category if category in ["frameworks", "libraries", "tools"] else "frameworks"].extend(tools[:3])
                recommendations["reasoning"].append(f"Detected {category} requirements from query")
        
        # Add search result insights
        for result in search_results:
            if result.get("abstract_text"):
                abstract = result["abstract_text"].lower()
                for category, tools in lang_mappings.items():
                    for tool in tools:
                        if tool.lower() in abstract:
                            if tool not in recommendations.get("frameworks", []) + recommendations.get("libraries", []):
                                recommendations["libraries"].append(tool)
        
        # Remove duplicates and limit
        for key in ["frameworks", "libraries", "tools"]:
            recommendations[key] = list(set(recommendations[key]))[:5]
        
        return recommendations
    
    def _extract_architecture_patterns(self, search_results: List[Dict], app_type: str) -> Dict[str, Any]:
        """Extract architecture patterns from search results"""
        patterns = {
            "recommended_patterns": [],
            "project_structure": {},
            "best_practices": [],
            "considerations": []
        }
        
        # Default patterns based on app type
        pattern_mappings = {
            "web_app": ["MVC", "Layered Architecture", "Repository Pattern"],
            "api": ["REST", "Microservices", "Clean Architecture"],
            "data_analysis": ["ETL Pipeline", "Data Lake", "Notebook-driven"],
            "machine_learning": ["ML Pipeline", "Model-View-Controller", "Feature Store"],
            "gui_app": ["MVP", "MVVM", "Component-based"]
        }
        
        patterns["recommended_patterns"] = pattern_mappings.get(app_type, ["Layered Architecture", "MVC"])
        
        # Add search-based insights
        for result in search_results:
            if result.get("related_topics"):
                for topic in result["related_topics"]:
                    if any(pattern_word in topic.lower() for pattern_word in ["pattern", "architecture", "design"]):
                        patterns["best_practices"].append(topic[:100])
        
        return patterns
    
    def search_sync(self, query: str, search_type: str = "tech_stack", language: str = "python") -> Dict[str, Any]:
        """Synchronous search method"""
        try:
            if search_type == "tech_stack":
                return self.search_tech_stack_recommendations(query, language)
            elif search_type == "architecture":
                return self.search_architecture_patterns(query, language)
            else:
                return {"error": "Unknown search type"}
        except Exception as e:
            console.print(f"❌ Search error: {e}", style="red")
            return {"error": str(e), "search_performed": False}

# Global instance
web_search_tool = WebSearchTool()

def search_tech_stack_recommendations(query: str, language: str = "python") -> Dict[str, Any]:
    """Search for tech stack recommendations"""
    return web_search_tool.search_sync(query, "tech_stack", language)

def search_architecture_patterns(app_type: str, language: str = "python") -> Dict[str, Any]:
    """Search for architecture patterns"""
    return web_search_tool.search_sync(app_type, "architecture", language)
