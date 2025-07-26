# msc/agents/package_manager.py
"""
Package Management Agent - Intelligent dependency resolution
"""
import re
import json
from typing import Dict, Any, List
from rich.console import Console
from langchain_core.prompts import ChatPromptTemplate

from msc.state import AgentState
from msc.tools import get_llm, load_prompt

console = Console()

def package_analyzer_agent(state: AgentState) -> Dict[str, Any]:
    """Analyze code to predict required packages before execution"""
    console.rule("[bold blue]PACKAGE ANALYZER: Predicting Dependencies[/bold blue]")
    
    code = state.get("corrected_code") or state.get("generated_code")
    if not code:
        return {"package_analysis": {"packages": [], "confidence": 0}}
    
    # Use LLM to intelligently analyze dependencies
    llm = get_llm("package_analyzer")
    
    prompt = ChatPromptTemplate.from_template("""
You are an expert Python package dependency analyzer.

TASK: Analyze the code and predict ALL required packages that need to be installed.

CRITICAL INSTRUCTIONS:
- Output ONLY a JSON object, no markdown, no explanations
- Include common packages that might be missed
- Consider indirect dependencies
- Map import names to actual pip package names

JSON Format:
{
  "packages": ["package1", "package2"],
  "confidence": 0.95,
  "reasoning": ["why package1 needed", "why package2 needed"]
}

Common mappings to remember:
- cv2 → opencv-python
- PIL → Pillow  
- sklearn → scikit-learn
- tensorflow → tensorflow-cpu (for faster install)
- torch → torch torchvision torchaudio

Code to analyze:
{code}
""")
    
    try:
        chain = prompt | llm
        response = chain.invoke({"code": code})
        
        # Parse LLM response as JSON
        analysis = json.loads(response.content.strip())
        console.print(f"📦 Predicted packages: {analysis.get('packages', [])}", style="blue")
        
        return {"package_analysis": analysis}
        
    except Exception as e:
        console.print(f"⚠️ Package analysis failed: {e}", style="yellow")
        # Fallback to regex-based detection
        return {"package_analysis": _fallback_package_detection(code)}

def package_installer_agent(state: AgentState) -> Dict[str, Any]:
    """Install predicted packages in Docker container"""
    console.rule("[bold green]PACKAGE INSTALLER: Installing Dependencies[/bold green]")
    
    analysis = state.get("package_analysis", {})
    packages = analysis.get("packages", [])
    
    if not packages:
        console.print("📦 No packages to install", style="dim")
        return {"package_installation": {"success": True, "installed": []}}
    
    # Get Docker container access
    from msc.tools.simple_project_docker import simple_docker_manager
    
    if not simple_docker_manager.current_project:
        console.print("⚠️ No active Docker project", style="yellow")
        return {"package_installation": {"success": False, "error": "No Docker project"}}
    
    # Install packages
    installed_packages = []
    failed_packages = []
    
    for package in packages:
        try:
            success = _install_package_in_current_container(package)
            if success:
                installed_packages.append(package)
                console.print(f"✅ Installed: {package}", style="green")
            else:
                failed_packages.append(package)
                console.print(f"❌ Failed: {package}", style="red")
        except Exception as e:
            failed_packages.append(package)
            console.print(f"❌ Error installing {package}: {e}", style="red")
    
    return {
        "package_installation": {
            "success": len(failed_packages) == 0,
            "installed": installed_packages,
            "failed": failed_packages
        }
    }

def dependency_resolver_agent(state: AgentState) -> Dict[str, Any]:
    """Intelligent dependency resolution for complex scenarios"""
    console.rule("[bold yellow]DEPENDENCY RESOLVER: Complex Resolution[/bold yellow]")
    
    # Get execution error details
    verifier_report = state.get("verifier_report", {})
    error_output = verifier_report.get("stderr", "") + verifier_report.get("stdout", "")
    
    if not error_output:
        return {"dependency_resolution": {"action": "none"}}
    
    # Use LLM for intelligent error resolution
    llm = get_llm("dependency_resolver")
    
    prompt = ChatPromptTemplate.from_template("""
You are an expert at resolving Python dependency and environment issues.

TASK: Analyze the error and provide a specific resolution strategy.

CRITICAL INSTRUCTIONS:
- Output ONLY a JSON object, no markdown
- Provide actionable solutions
- Consider environment conflicts, version issues, system dependencies

JSON Format:
{
  "action": "install_packages|fix_imports|system_deps|version_conflict|other",
  "packages": ["pkg1", "pkg2"],
  "commands": ["cmd1", "cmd2"],
  "reasoning": "why this solution",
  "confidence": 0.9
}

Error Output:
{error_output}

Code Context:
{code}
""")
    
    code = state.get("corrected_code") or state.get("generated_code")
    
    try:
        chain = prompt | llm
        response = chain.invoke({
            "error_output": error_output,
            "code": code
        })
        
        resolution = json.loads(response.content.strip())
        console.print(f"🔧 Resolution strategy: {resolution.get('action')}", style="yellow")
        console.print(f"🧠 Reasoning: {resolution.get('reasoning')}", style="dim")
        
        return {"dependency_resolution": resolution}
        
    except Exception as e:
        console.print(f"⚠️ Dependency resolution failed: {e}", style="red")
        return {"dependency_resolution": {"action": "fallback"}}

def _fallback_package_detection(code: str) -> Dict[str, Any]:
    """Fallback regex-based package detection"""
    import_patterns = [
        r'import\s+(\w+)',
        r'from\s+(\w+)\s+import',
        r'import\s+(\w+\.\w+)'  # Handle sub-packages
    ]
    
    detected_imports = set()
    for pattern in import_patterns:
        matches = re.findall(pattern, code)
        detected_imports.update(matches)
    
    # Map to actual packages
    package_map = {
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'sklearn': 'scikit-learn',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'tensorflow': 'tensorflow-cpu',
        'torch': 'torch',
        'keras': 'keras',
        'requests': 'requests',
        'flask': 'flask',
        'django': 'django'
    }
    
    packages = []
    for imp in detected_imports:
        if imp in package_map:
            packages.append(package_map[imp])
        elif not imp.startswith('_') and imp not in ['os', 'sys', 'json', 'time', 're']:
            packages.append(imp)  # Assume package name matches import
    
    return {
        "packages": list(set(packages)),
        "confidence": 0.7,
        "reasoning": ["Regex-based detection"]
    }

def _install_package_in_current_container(package_name: str) -> bool:
    """Install package in the current Docker container"""
    try:
        from msc.tools.simple_project_docker import simple_docker_manager
        import docker
        
        client = docker.from_env()
        
        # Get current container name
        if simple_docker_manager.session_dir:
            session_name = simple_docker_manager.session_dir.name
            if session_name == f"session-{simple_docker_manager.current_language}":
                container_name = f"msc-{simple_docker_manager.current_language}-reusable"
            else:
                timestamp = session_name.split('-')[-1]
                container_name = f"msc-{simple_docker_manager.current_language}-{timestamp}"
        else:
            container_name = f"msc-{simple_docker_manager.current_language or 'python'}-reusable"
        
        container = client.containers.get(container_name)
        
        # Install package
        install_cmd = f"pip install {package_name}"
        result = container.exec_run(["bash", "-c", install_cmd])
        
        return result.exit_code == 0
        
    except Exception as e:
        console.print(f"❌ Container package install failed: {e}", style="red")
        return False
