# msc/tools/code_quality.py
"""
Code quality and issue detection
"""
import re
from typing import List, Dict, Any
from rich.console import Console

console = Console()

class CodeQualityChecker:
    """Detects common code issues before execution"""
    
    def __init__(self):
        self.issues = []
    
    def check_code(self, code: str, filename: str = "script.py") -> Dict[str, Any]:
        """Comprehensive code quality check"""
        self.issues = []
        
        # Check for markdown
        self._check_markdown(code)
        
        # Check for common syntax issues
        self._check_syntax_issues(code)
        
        # Check for incomplete code
        self._check_completeness(code)
        
        # Language-specific checks
        if filename.endswith('.py'):
            self._check_python_specific(code)
        
        return {
            "has_issues": len(self.issues) > 0,
            "issues": self.issues,
            "warnings": [i for i in self.issues if i["severity"] == "warning"],
            "errors": [i for i in self.issues if i["severity"] == "error"],
            "clean_code": self._auto_fix(code) if self.issues else code
        }
    
    def _check_markdown(self, code: str):
        """Check for markdown code blocks"""
        if "```" in code:
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('```'):
                    self.issues.append({
                        "type": "markdown_block",
                        "severity": "error",
                        "line": i + 1,
                        "message": f"Markdown code block detected: {line.strip()}",
                        "fix": "Remove markdown formatting"
                    })
    
    def _check_syntax_issues(self, code: str):
        """Check for obvious syntax issues"""
        # Check for unmatched quotes
        single_quotes = code.count("'") - code.count("\\'")
        double_quotes = code.count('"') - code.count('\\"')
        
        if single_quotes % 2 != 0:
            self.issues.append({
                "type": "unmatched_quotes",
                "severity": "error",
                "message": "Unmatched single quotes detected",
                "fix": "Check quote pairing"
            })
        
        if double_quotes % 2 != 0:
            self.issues.append({
                "type": "unmatched_quotes", 
                "severity": "error",
                "message": "Unmatched double quotes detected",
                "fix": "Check quote pairing"
            })
    
    def _check_completeness(self, code: str):
        """Check if code appears complete"""
        stripped = code.strip()
        
        if not stripped:
            self.issues.append({
                "type": "empty_code",
                "severity": "error",
                "message": "Code is empty",
                "fix": "Provide actual code"
            })
            return
        
        # Check for placeholder text
        placeholders = ["TODO", "FIXME", "...", "# Your code here", "pass  # TODO"]
        for placeholder in placeholders:
            if placeholder in code:
                self.issues.append({
                    "type": "placeholder",
                    "severity": "warning",
                    "message": f"Placeholder text found: {placeholder}",
                    "fix": "Replace with actual implementation"
                })
    
    def _check_python_specific(self, code: str):
        """Python-specific checks"""
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            # Check for common Python issues
            if line.strip().startswith('from ... import'):
                self.issues.append({
                    "type": "incomplete_import",
                    "severity": "error",
                    "line": i + 1,
                    "message": "Incomplete import statement",
                    "fix": "Specify module name"
                })
    
    def _auto_fix(self, code: str) -> str:
        """Attempt to automatically fix simple issues"""
        # Remove markdown blocks
        fixed_code = re.sub(r'^```[a-zA-Z]*\n', '', code, flags=re.MULTILINE)
        fixed_code = re.sub(r'\n```$', '', fixed_code, flags=re.MULTILINE)
        fixed_code = re.sub(r'^```$', '', fixed_code, flags=re.MULTILINE)
        
        return fixed_code.strip()
    
    def report_issues(self):
        """Display issues to user"""
        if not self.issues:
            console.print("✅ No code quality issues detected", style="green")
            return
        
        console.print("\n⚠️ Code Quality Issues Detected:", style="yellow bold")
        
        for issue in self.issues:
            severity_color = "red" if issue["severity"] == "error" else "yellow"
            console.print(f"  {issue['severity'].upper()}: {issue['message']}", style=severity_color)
            if "line" in issue:
                console.print(f"    Line {issue['line']}", style="dim")
            console.print(f"    Fix: {issue['fix']}", style="blue")
            console.print()

# Global instance
code_checker = CodeQualityChecker()
