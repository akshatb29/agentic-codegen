# msc/tools/context_analyzer.py
import re
from typing import Dict, List
from pathlib import Path

# --- Try importing semantic model ---
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from numpy.linalg import norm
    SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    SEMANTIC_AVAILABLE = True
except Exception as e:
    print(f"[ContextAnalyzer] Semantic model unavailable, using fallback. Reason: {e}")
    SEMANTIC_AVAILABLE = False


class ContextAnalyzer:
    """Analyzes user requests to suggest relevant files for context."""

    @staticmethod
    def analyze_user_request(user_request: str, available_files: Dict[str, str]) -> List[str]:
        """
        Analyze the user request and suggest relevant files based on semantic embeddings
        with fallback to keyword-based relevance.
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

        # Semantic embedding of user request (if model available)
        request_vector = None
        if SEMANTIC_AVAILABLE:
            try:
                request_vector = SEMANTIC_MODEL.encode([user_request])[0]
            except Exception as e:
                print(f"[ContextAnalyzer] Semantic encoding failed, using fallback: {e}")
                request_vector = None

        # Compute relevance for each file
        for file_path, content in available_files.items():
            file_name = Path(file_path).name

            # Check explicit file references first
            if any(explicit in file_name.lower() for explicit in explicit_files):
                suggested_files.append(file_path)
                continue

            # Semantic relevance
            if request_vector is not None:
                try:
                    content_vector = SEMANTIC_MODEL.encode([content[:1000]])[0]  # limit for speed
                    similarity = float(np.dot(request_vector, content_vector) / (norm(request_vector) * norm(content_vector)))
                    if similarity > 0.35:  # threshold (tweakable)
                        suggested_files.append(file_path)
                        continue
                except Exception as e:
                    print(f"[ContextAnalyzer] File embedding failed for {file_name}: {e}")
                    # fallback for this file
                    if ContextAnalyzer._fallback_keyword_relevance(request_lower, file_path, content):
                        suggested_files.append(file_path)
            else:
                # fallback keyword matching
                if ContextAnalyzer._fallback_keyword_relevance(request_lower, file_path, content):
                    suggested_files.append(file_path)

        return suggested_files

    @staticmethod
    def _fallback_keyword_relevance(request_lower: str, file_path: str, file_content: str) -> bool:
        """Fallback keyword-based relevance check."""
        file_name = Path(file_path).name.lower()
        content_lower = file_content.lower()

        keyword_mappings = {
            ('graph', 'workflow', 'pipeline', 'flow'): ['graph.py', 'workflow', 'pipeline'],
            ('agent', 'planner', 'planning'): ['planner', 'agent'],
            ('generate', 'code', 'generator'): ['generator', 'code_generator'],
            ('verify', 'verifier', 'validation'): ['verifier', 'verify'],
            ('correct', 'corrector', 'fix', 'debug'): ['corrector', 'correct'],
            ('critique', 'review', 'analyze'): ['critique', 'review'],
            ('reason', 'reasoning', 'symbolic'): ['reasoner', 'reason'],
            ('prompt', 'template'): ['prompt', 'template'],
            ('file', 'filesystem', 'read', 'write'): ['filesystem', 'file'],
            ('execute', 'execution', 'run'): ['execution', 'execute'],
            ('user', 'interaction', 'input'): ['user_interaction', 'interaction'],
            ('config', 'configuration', 'setup'): ['config', 'setup', 'main'],
            ('state', 'data', 'model'): ['state', 'model'],
            ('tool', 'utility', 'helper'): ['tool', 'util'],
        }

        for request_keywords, file_keywords in keyword_mappings.items():
            if any(keyword in request_lower for keyword in request_keywords):
                if any(file_keyword in file_name for file_keyword in file_keywords):
                    return True
                if any(keyword in content_lower for keyword in file_keywords):
                    return True

        python_constructs = {
            'class': r'class\s+\w+',
            'function': r'def\s+\w+',
            'import': r'import\s+\w+|from\s+\w+\s+import',
        }

        for construct in python_constructs:
            if construct in request_lower and re.search(python_constructs[construct], content_lower):
                return True

        return False

    @staticmethod
    def get_smart_suggestions(user_request: str, available_files: Dict[str, str]) -> Dict[str, List[str]]:
        """Categorize suggestions by priority levels."""
        suggested = ContextAnalyzer.analyze_user_request(user_request, available_files)

        high_priority, medium_priority, low_priority = [], [], []
        for file_path in suggested:
            file_name = Path(file_path).name.lower()
            if any(pattern in file_name for pattern in ['main.py', 'app.py', '__init__.py']):
                high_priority.append(file_path)
            elif any(pattern in file_path.lower() for pattern in ['agent', 'tool', 'core']):
                medium_priority.append(file_path)
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

    base_dir = Path(__file__).parent.parent.parent
    available_files = {}
    for file_path in base_dir.rglob('*.py'):
        if 'pycache' in file_path.parts:
            continue
        try:
            available_files[str(file_path)] = file_path.read_text()
        except Exception:
            pass

    request = input("❓ Enter a user request to analyze: ")
    suggestions = ContextAnalyzer.get_smart_suggestions(request, available_files)
    print("\n=== Suggested Files ===")
    pprint.pprint(suggestions)
    print("=== End of Suggestions ===")
