# msc/tools/execution.py
"""
Simple execution tool - project-based Docker or local execution
"""
import tempfile
import subprocess
import os
import re
from typing import Dict, Any
from rich.console import Console

from .simple_project_docker import simple_docker_manager

console = Console()

def extract_code_from_markdown(code_text: str) -> str:
    """Extract actual code from markdown code blocks"""
    # Remove markdown code blocks
    code_text = re.sub(r'^```[a-zA-Z]*\n', '', code_text, flags=re.MULTILINE)
    code_text = re.sub(r'\n```$', '', code_text, flags=re.MULTILINE)
    code_text = re.sub(r'^```$', '', code_text, flags=re.MULTILINE)
    
    # Clean up extra whitespace
    code_text = code_text.strip()
    
    return code_text

def run_code(code: str, filename: str = "script.py", use_docker: bool = True, 
             user_request: str = "", project_name: str = "", language: str = "", 
             ask_reuse: bool = True) -> Dict[str, Any]:
    """
    Main execution function - simple project-based Docker or local execution
    
    Args:
        code: Code to execute (any supported language)
        filename: Name for the script file (determines language)
        use_docker: Whether to use Docker (project-based) or local execution
        user_request: Original user request for project naming
        project_name: Suggested project name (passed from agents)
        language: Target language (passed from agents)
        ask_reuse: Whether to ask user about container reuse preference
        
    Returns:
        Execution result dictionary
    """
    # Clean code from markdown blocks
    cleaned_code = extract_code_from_markdown(code)
    
    if use_docker:
        console.print("🚀 Executing with language-specific Docker container...", style="blue")
        try:
            # Debug: Check current project state
            console.print(f"🔍 Debug: current_project = {simple_docker_manager.current_project}", style="dim")
            
            # Auto-setup project if not done yet, ask user only on first setup
            if not simple_docker_manager.current_project:
                console.print("🔧 Setting up new project...", style="blue")
                simple_docker_manager.get_or_create_project(user_request, project_name, language, ask_reuse)
            else:
                console.print("♻️ Reusing existing project setup", style="dim")
            
            result = simple_docker_manager.execute_code(cleaned_code, filename, user_request)
            if result.get("success"):
                return result
            else:
                # Docker failed, fallback to local
                console.print("⚠️ Docker execution failed, falling back to local execution", style="yellow")
                console.print("💻 Executing locally...", style="blue")
                return _run_local(cleaned_code, filename)
        except Exception as e:
            # Docker completely failed, fallback to local
            console.print(f"⚠️ Docker error: {e}", style="yellow")
            console.print("💻 Falling back to local execution...", style="blue")
            return _run_local(cleaned_code, filename)
    else:
        console.print("💻 Executing locally...", style="yellow")
        return _run_local(cleaned_code, filename)

def _run_local(code: str, filename: str) -> Dict[str, Any]:
    """Local execution fallback"""
    try:
        # Create temporary file with appropriate extension
        suffix = _get_file_extension(filename)
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        # Get execution command
        cmd = _get_local_execution_command(temp_path, filename)
        
        # Execute
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # Clean up
        os.unlink(temp_path)
        
        success = result.returncode == 0
        console.print(f"{'✅' if success else '❌'} Local execution {'completed' if success else 'failed'}")
        
        return {
            "success": success,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "filename": filename,
            "mode": "local"
        }
        
    except subprocess.TimeoutExpired:
        if 'temp_path' in locals():
            os.unlink(temp_path)
        return {
            "success": False,
            "stdout": "",
            "stderr": "Execution timed out after 30 seconds",
            "filename": filename,
            "mode": "local"
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "filename": filename,
            "mode": "local"
        }

def _get_file_extension(filename: str) -> str:
    """Get file extension for temporary files"""
    if filename.endswith(('.py', '.pyw')):
        return '.py'
    elif filename.endswith('.js'):
        return '.js'
    elif filename.endswith('.go'):
        return '.go'
    elif filename.endswith('.rs'):
        return '.rs'
    elif filename.endswith(('.c', '.cpp', '.cc', '.cxx')):
        return '.cpp' if any(filename.endswith(ext) for ext in ['.cpp', '.cc', '.cxx']) else '.c'
    elif filename.endswith('.java'):
        return '.java'
    else:
        return '.py'  # Default to Python

def _get_local_execution_command(temp_path: str, filename: str) -> list:
    """Get local execution command based on file type"""
    if filename.endswith(('.py', '.pyw')):
        return ['python3', temp_path]
    elif filename.endswith('.js'):
        return ['node', temp_path]
    elif filename.endswith('.go'):
        return ['go', 'run', temp_path]
    elif filename.endswith('.java'):
        # Java requires compilation first
        return ['javac', temp_path, '&&', 'java', temp_path[:-5]]  # Remove .java
    else:
        # Default to Python
        return ['python3', temp_path]

# Backward compatibility aliases for existing agents
execute_code_locally = _run_local
execute_python_code = run_code

# Enhanced execution function for agents that need to pass project/language info
def execute_with_context(code: str, filename: str = "script.py", user_request: str = "",
                        project_name: str = "", language: str = "", use_docker: bool = True,
                        ask_reuse: bool = False) -> Dict[str, Any]:
    """Execute code with full context - used by agents (auto-reuse by default)"""
    return run_code(code, filename, use_docker, user_request, project_name, language, ask_reuse)

def run_code_interactive(code: str, filename: str = "script.py", **kwargs) -> Dict[str, Any]:
    """Interactive version that always asks user about container reuse"""
    return run_code(code, filename, ask_reuse=True, **kwargs)

def run_code_auto(code: str, filename: str = "script.py", **kwargs) -> Dict[str, Any]:
    """Automated version that always reuses containers without asking"""
    return run_code(code, filename, ask_reuse=False, **kwargs)

def run_code_safe(code: str, filename: str = "script.py", **kwargs) -> Dict[str, Any]:
    """Safe version that asks user permission before execution"""
    from .safe_testing import safe_tester
    
    # Ask user before any execution
    if not safe_tester.propose_test(
        description=f"Execute {filename} with Docker/local fallback",
        command="Docker execution with automatic local fallback",
        code=code,
        expected_outcome="Safe execution in controlled environment"
    ):
        return {"success": False, "error": "User declined execution", "skipped": True}
    
    return run_code(code, filename, ask_reuse=True, **kwargs)
