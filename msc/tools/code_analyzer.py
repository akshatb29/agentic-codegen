# msc/tools/code_analyzer.py
import ast
from pathlib import Path
from typing import Dict, List, Set, Optional, Any

class CodeGraphVisitor(ast.NodeVisitor):
    """
    An AST visitor that walks the code's syntax tree to find key structures
    like function definitions, class definitions, and function calls.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.current_class_name: Optional[str] = None
        self.definitions: Dict[str, Dict[str, Any]] = {}
        self.calls: Set[str] = set()
        self.imports: Dict[str, str] = {} # alias -> full_path

    def visit_Import(self, node: ast.Import):
        """Catches 'import module' statements."""
        for alias in node.names:
            self.imports[alias.asname or alias.name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Catches 'from module import function' statements."""
        module_name = node.module or ''
        for alias in node.names:
            # Store the alias and the full potential path it could represent
            self.imports[alias.asname or alias.name] = f"{module_name}.{alias.name}"
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visits a class definition."""
        self.definitions[node.name] = {
            "type": "class",
            "file_path": self.file_path,
            "line": node.lineno,
            "methods": []
        }
        # Track current class to associate methods with it
        self.current_class_name = node.name
        self.generic_visit(node)
        self.current_class_name = None # Reset after visiting class body

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visits a function or method definition."""
        # If inside a class, it's a method
        if self.current_class_name:
            key = f"{self.current_class_name}.{node.name}"
            if self.current_class_name in self.definitions:
                 self.definitions[self.current_class_name]["methods"].append(node.name)
        else:
            key = node.name

        self.definitions[key] = {
            "type": "function",
            "file_path": self.file_path,
            "line": node.lineno,
        }
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Visits a function call."""
        # We try to resolve the name of the function being called
        func = node.func
        if isinstance(func, ast.Name):
            # Simple call like `my_function()`
            self.calls.add(func.id)
        elif isinstance(func, ast.Attribute):
            # Attribute call like `my_object.my_method()`
            # We can't easily resolve the object type here, but we can record the method name
            self.calls.add(func.attr)
        self.generic_visit(node)


class CodeAnalyzer:
    """
    Analyzes a project's Python files to build a structural "knowledge graph"
    of definitions and calls, enabling precise dependency analysis.
    """
    def __init__(self, files_content: Dict[str, str]):
        """
        Initializes the analyzer with the project's file content.

        Args:
            files_content: A dictionary mapping file paths to their string content.
        """
        self.files_content = files_content
        self.analysis_results: Dict[str, Dict[str, Any]] = {}
        self._build_code_graph()

    def _build_code_graph(self):
        """Parses all files and builds the internal code graph."""
        for file_path, content in self.files_content.items():
            if not file_path.endswith('.py'):
                continue
            try:
                tree = ast.parse(content, filename=file_path)
                visitor = CodeGraphVisitor(file_path)
                visitor.visit(tree)
                self.analysis_results[file_path] = {
                    "definitions": visitor.definitions,
                    "calls": visitor.calls,
                    "imports": visitor.imports,
                }
            except SyntaxError as e:
                print(f"[CodeAnalyzer] Skipping {file_path} due to syntax error: {e}")

    def get_function_analysis(self, function_name: str) -> Dict[str, List[str]]:
        """
        Finds where a function is defined and all files that call it.

        Args:
            function_name: The name of the function to analyze (e.g., "calculate_price").

        Returns:
            A dictionary with two keys:
            - 'defining_files': A list of files where the function is defined.
            - 'calling_files': A list of files that call the function.
        """
        defining_files = set()
        calling_files = set()

        # Find where the function is defined
        for file_path, analysis in self.analysis_results.items():
            if function_name in analysis["definitions"]:
                defining_files.add(file_path)

        # Find all files that call this function
        for file_path, analysis in self.analysis_results.items():
            if function_name in analysis["calls"]:
                # To reduce false positives, we could check if the file
                # imports a module that is likely to contain the function,
                # but for now, a direct name match is a good start.
                calling_files.add(file_path)

        # Ensure defining files are not listed in calling files if they just call themselves
        calling_files = calling_files - defining_files

        return {
            "defining_files": sorted(list(defining_files)),
            "calling_files": sorted(list(calling_files)),
        }

if __name__ == '__main__':
    # --- Example Usage ---
    # In a real scenario, you would get this from your FilesystemTool
    base_dir = Path(__file__).parent.parent.parent
    available_files = {}
    for file_path in base_dir.rglob('*.py'):
        if '__pycache__' in file_path.parts or 'site-packages' in file_path.parts:
            continue
        try:
            available_files[str(file_path)] = file_path.read_text(encoding='utf-8')
        except Exception:
            pass # Ignore files that can't be read

    if not available_files:
        print("No Python files found to analyze.")
    else:
        print(f"Analyzing {len(available_files)} Python files...")
        code_analyzer = CodeAnalyzer(available_files)

        # --- Interactive Test ---
        while True:
            func_to_find = input("\nEnter a function name to find (or 'quit'): ").strip()
            if func_to_find.lower() == 'quit':
                break

            analysis = code_analyzer.get_function_analysis(func_to_find)

            print(f"\n--- Analysis for '{func_to_find}' ---")
            if analysis['defining_files']:
                print("✅ Found Definition In:")
                for f in analysis['defining_files']:
                    print(f"   - {f}")
            else:
                print("❌ Function definition not found.")

            if analysis['calling_files']:
                print("\n✅ Found Calls In:")
                for f in analysis['calling_files']:
                    print(f"   - {f}")
            else:
                print("\n❌ No other files seem to call this function.")
            print("-" * (24 + len(func_to_find)))

