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

# Import simple_docker_manager conditionally to avoid circular imports
try:
    from .simple_project_docker import simple_docker_manager
    DOCKER_AVAILABLE = True
except ImportError:
    simple_docker_manager = None
    DOCKER_AVAILABLE = False

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
             ask_reuse: bool = True, state: dict = None) -> Dict[str, Any]:
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
        state: LLM planning state with execution commands
        
    Returns:
        Execution result dictionary
    """
    # Clean code from markdown blocks
    cleaned_code = extract_code_from_markdown(code)
    
    if use_docker and DOCKER_AVAILABLE:
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
                # Docker failed, fallback to local with terminal window
                console.print("⚠️ Docker execution failed, falling back to local execution", style="yellow")
                console.print("💻 Executing locally with terminal window...", style="blue")
                return _run_local_with_terminal(cleaned_code, filename, state)
        except Exception as e:
            # Docker completely failed, fallback to local with terminal window
            console.print(f"⚠️ Docker error: {e}", style="yellow")
            console.print("💻 Falling back to local execution with terminal window...", style="blue")
            return _run_local_with_terminal(cleaned_code, filename, state)
    else:
        if use_docker and not DOCKER_AVAILABLE:
            console.print("⚠️ Docker requested but not available, using local execution", style="yellow")
        console.print("💻 Executing locally with terminal window...", style="yellow")
        return _run_local_with_terminal(cleaned_code, filename, state)

def _run_local_with_terminal(code: str, filename: str, state: dict = None) -> Dict[str, Any]:
    """Execute code in external terminal window - like Copilot"""
    try:
        from pathlib import Path
        
        # BUILD EXECUTION PLAN FROM AGENT STATE
        project_name = "untitled-project"
        packages_to_install = []
        
        if state:
            # Get project name from LLM planning
            project_name = state.get("project_name", "untitled-project")
            
            # Get packages from package analysis and software design
            package_analysis = state.get("package_analysis", {})
            software_design = state.get("software_design", {})
            
            # From package analysis
            if package_analysis and "packages" in package_analysis:
                packages_to_install.extend(package_analysis["packages"])
            
            # From software design requirements
            if isinstance(software_design, dict):
                requirements = software_design.get("requirements", [])
                packages_to_install.extend(requirements)
            else:
                # Handle Pydantic model
                requirements = getattr(software_design, 'requirements', [])
                packages_to_install.extend(requirements)
            
            # Remove duplicates
            packages_to_install = list(set(packages_to_install))
                
        # Create project directory
        project_dir = Path(project_name)
        project_dir.mkdir(exist_ok=True)
        
        # Save file in project directory
        local_filename = filename if filename else "main.py"
        local_file_path = project_dir / local_filename
        
        with open(local_file_path, 'w') as f:
            f.write(code)
        console.print(f"📄 File saved: [bold cyan]{local_file_path}[/bold cyan]", style="green")
        
        # SHOW EXECUTION PLAN TO USER
        execution_plan = []
        if packages_to_install:
            execution_plan.append(f"📦 Install packages: {', '.join(packages_to_install)}")
        execution_plan.append(f"🚀 Execute: python {local_filename}")
        
        console.print("\n[bold cyan]📋 EXECUTION PLAN:[/bold cyan]")
        for i, step in enumerate(execution_plan, 1):
            console.print(f"  {i}. {step}")
        
        # ASK FOR PERMISSION BEFORE EXECUTION
        try:
            from .user_interaction import user_confirmation_tool
            user_approval = user_confirmation_tool("\n🔥 Execute this plan?")
        except ImportError:
            # Fallback to simple input if user_interaction module not available
            user_approval = input("\n🔥 Execute this plan? (y/n): ").strip().lower()
        
        if not user_approval.lower() in ["y", "yes", "true", "1"]:
            console.print("❌ Execution cancelled by user", style="red")
            return {
                "success": False,
                "stdout": "",
                "stderr": "Execution cancelled by user",
                "filename": filename,
                "mode": "cancelled"
            }
        
        # NOW EXECUTE IN TERMINAL WINDOW
        console.print("\n🚀 [bold green]EXECUTING IN TERMINAL...[/bold green]")
        
        # Build command string for terminal execution
        commands = []
        commands.append(f"cd '{project_dir}'")
        commands.append("echo '🚀 Starting execution...'")
        
        # Add package installation commands if needed
        if packages_to_install:
            commands.append(f"echo '📦 Installing packages: {', '.join(packages_to_install)}'")
            for package in packages_to_install:
                commands.append(f"pip install {package}")
        
        # Add main execution command
        commands.append(f"echo '▶️ Running {local_filename}...'")
        commands.append(f"python {local_filename}")
        commands.append("echo '✅ Execution completed!'")
        commands.append("echo 'Press Enter to close or Ctrl+C to exit'")
        commands.append("read")
        
        # Join commands with && for sequential execution
        full_command = " && ".join(commands)
        
        # Try to open in external terminal for visibility
        try:
            # Try gnome-terminal first
            console.print("🖥️ Opening execution in new terminal window...")
            subprocess.Popen([
                'gnome-terminal', 
                '--', 'bash', '-c', full_command
            ])
            console.print("✅ Execution started in new terminal window")
            
            return {
                "success": True,
                "stdout": f"Code saved to {local_file_path} and executed in new terminal",
                "stderr": "",
                "filename": filename,
                "mode": "terminal_execution",
                "project_path": str(project_dir)
            }
            
        except FileNotFoundError:
            try:
                # Try xterm fallback
                console.print("🖥️ Trying xterm...")
                subprocess.Popen(['xterm', '-e', 'bash', '-c', full_command])
                console.print("✅ Execution started in xterm window")
                
                return {
                    "success": True,
                    "stdout": f"Code saved to {local_file_path} and executed in xterm",
                    "stderr": "",
                    "filename": filename,
                    "mode": "terminal_execution",
                    "project_path": str(project_dir)
                }
                
            except FileNotFoundError:
                # Fallback to simple local execution without terminal window
                console.print("⚠️ No external terminal found - running simple local execution")
                return _run_local(code, filename)
        
    except Exception as e:
        console.print(f"❌ Setup error: {e}", style="red")
        # Fallback to simple local execution
        return _run_local(code, filename)

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
                        ask_reuse: bool = False, state: dict = None) -> Dict[str, Any]:
    """Execute code with full context - used by agents (auto-reuse by default)"""
    return run_code(code, filename, use_docker, user_request, project_name, language, ask_reuse, state)

def run_code_interactive(code: str, filename: str = "script.py", **kwargs) -> Dict[str, Any]:
    """Interactive version that always asks user about container reuse"""
    return run_code(code, filename, ask_reuse=True, **kwargs)

def run_code_auto(code: str, filename: str = "script.py", **kwargs) -> Dict[str, Any]:
    """Automated version that always reuses containers without asking"""
    return run_code(code, filename, ask_reuse=False, **kwargs)

def run_code_safe(code: str, filename: str = "script.py", **kwargs) -> Dict[str, Any]:
    """Safe version that asks user permission before execution"""
    try:
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
    except ImportError:
        # Fallback if safe_testing module is not available
        console.print("⚠️ Safe testing module not available, proceeding with normal execution", style="yellow")
        return run_code(code, filename, ask_reuse=True, **kwargs)
