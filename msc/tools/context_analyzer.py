# msc/tools/context_analyzer.py
import re
from typing import Dict, List, Set
from pathlib import Path

class ContextAnalyzer:
    """Analyzes user requests to suggest relevant files for context."""
    
    @staticmethod
    def analyze_user_request(user_request: str, available_files: Dict[str, str]) -> List[str]:
        """
        Analyze the user request and suggest relevant files based on keywords, 
        file patterns, and content analysis.
        
        Args:
            user_request: The user's request/query
            available_files: Dict of {file_path: file_content}
        
        Returns:
            List of suggested file paths
        """
        suggested_files = []
        request_lower = user_request.lower()
        
        # Extract potential file mentions (explicit file references)
        file_patterns = [
            r'(?:file|modify|update|edit|change|fix)\s+["\']?([^"\'\s]+\.[a-zA-Z0-9]+)["\']?',
            r'["\']([^"\'\s]*\.[a-zA-Z0-9]+)["\']',
            r'\b([a-zA-Z0-9_]+\.(py|txt|json|md|yml|yaml))\b'
        ]
        
        explicit_files = set()
        for pattern in file_patterns:
            matches = re.findall(pattern, request_lower, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    explicit_files.add(match[0])
                else:
                    explicit_files.add(match)
        
        # Find matching files in available files
        for file_path in available_files.keys():
            file_name = Path(file_path).name
            # Check for explicit file mentions
            if any(explicit in file_name.lower() for explicit in explicit_files):
                suggested_files.append(file_path)
                continue
                
            # Check for keyword relevance
            if ContextAnalyzer._is_file_relevant(request_lower, file_path, available_files[file_path]):
                suggested_files.append(file_path)
        
        return suggested_files
    
    @staticmethod
    def _is_file_relevant(request_lower: str, file_path: str, file_content: str) -> bool:
        """Check if a file is relevant based on keywords and content analysis."""
        file_name = Path(file_path).name.lower()
        content_lower = file_content.lower()
        
        # Common keywords for different types of requests
        keyword_mappings = {
            # Framework/architecture keywords
            ('graph', 'workflow', 'pipeline', 'flow'): ['graph.py', 'workflow', 'pipeline'],
            ('agent', 'planner', 'planning'): ['planner', 'agent'],
            ('generate', 'code', 'generator'): ['generator', 'code_generator'],
            ('verify', 'verifier', 'validation'): ['verifier', 'verify'],
            ('correct', 'corrector', 'fix', 'debug'): ['corrector', 'correct'],
            ('critique', 'review', 'analyze'): ['critique', 'review'],
            ('reason', 'reasoning', 'symbolic'): ['reasoner', 'reason'],
            ('prompt', 'template'): ['prompt', 'template'],
            
            # File operation keywords
            ('file', 'filesystem', 'read', 'write'): ['filesystem', 'file'],
            ('execute', 'execution', 'run'): ['execution', 'execute'],
            ('user', 'interaction', 'input'): ['user_interaction', 'interaction'],
            
            # Configuration keywords
            ('config', 'configuration', 'setup'): ['config', 'setup', 'main'],
            ('state', 'data', 'model'): ['state', 'model'],
            ('tool', 'utility', 'helper'): ['tool', 'util'],
        }
        
        # Check keyword relevance
        for request_keywords, file_keywords in keyword_mappings.items():
            if any(keyword in request_lower for keyword in request_keywords):
                if any(file_keyword in file_name for file_keyword in file_keywords):
                    return True
                    
                # Also check content for class/function names
                if any(keyword in content_lower for keyword in file_keywords):
                    return True
        
        # Check for specific Python constructs mentioned in request
        python_constructs = {
            'class': r'class\s+\w+',
            'function': r'def\s+\w+',
            'import': r'import\s+\w+|from\s+\w+\s+import',
        }
        
        for construct in python_constructs:
            if construct in request_lower:
                if re.search(python_constructs[construct], content_lower):
                    return True
        
        return False
    
    @staticmethod
    def get_smart_suggestions(user_request: str, available_files: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Get smart suggestions categorized by relevance.
        
        Returns:
            Dict with 'high_priority', 'medium_priority', 'low_priority' file lists
        """
        suggested = ContextAnalyzer.analyze_user_request(user_request, available_files)
        
        # Categorize suggestions
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for file_path in suggested:
            file_name = Path(file_path).name.lower()
            
            # High priority: main files, exact matches
            if any(pattern in file_name for pattern in ['main.py', 'app.py', '__init__.py']):
                high_priority.append(file_path)
            # Medium priority: agent files, core functionality
            elif any(pattern in file_path.lower() for pattern in ['agent', 'tool', 'core']):
                medium_priority.append(file_path)
            # Low priority: supporting files
            else:
                low_priority.append(file_path)
        
        return {
            'high_priority': high_priority,
            'medium_priority': medium_priority,
            'low_priority': low_priority
        }

if __name__ == '__main__':
    from pathlib import Path
    import pprint

    # Gather all .py files under the project, excluding __pycache__
    base_dir = Path(__file__).parent.parent.parent
    available_files = {}
    for file_path in base_dir.rglob('*.py'):
        if 'pycache' in file_path.parts:
            continue
        try:
            available_files[str(file_path)] = file_path.read_text()
        except Exception:
            pass

    # Prompt user for a test request
    request = input("❓ Enter a user request to analyze: ")
    suggestions = ContextAnalyzer.get_smart_suggestions(request, available_files)

    print("\n=== Suggested Files ===")
    pprint.pprint(suggestions)
    print("=== End of Suggestions ===")
