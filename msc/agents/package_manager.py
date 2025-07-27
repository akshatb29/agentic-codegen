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
    
    try:
        prompt = ChatPromptTemplate.from_template(load_prompt("package_analyzer.txt"))
        chain = prompt | llm
        response = chain.invoke({"code": code})
        
        # Clean and parse LLM response as JSON
        content = response.content.strip()
        
        # Extract JSON from markdown blocks if present
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()
        
        # Handle cases where LLM returns just text or malformed JSON
        if not content or content.startswith("I ") or "packages" not in content:
            # Extract packages from code directly as fallback
            import re
            imports = re.findall(r'(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
            package_map = {
                'numpy': 'numpy', 'np': 'numpy',
                'pandas': 'pandas', 'pd': 'pandas', 
                'matplotlib': 'matplotlib', 'plt': 'matplotlib',
                'torch': 'torch', 'tensorflow': 'tensorflow',
                'sklearn': 'scikit-learn', 'cv2': 'opencv-python',
                'PIL': 'Pillow', 'requests': 'requests'
            }
            packages = [package_map.get(imp, imp) for imp in imports if imp in package_map]
            analysis = {
                "packages": list(set(packages)),
                "confidence": 0.7,
                "reasoning": ["Extracted from import statements"]
            }
        else:
            # Try to parse JSON
            try:
                analysis = json.loads(content)
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                content = content.replace("'", '"')  # Single to double quotes
                content = re.sub(r',\s*}', '}', content)  # Remove trailing commas
                content = re.sub(r',\s*]', ']', content)  # Remove trailing commas in arrays
                analysis = json.loads(content)
        
        console.print(f"📦 Predicted packages: {analysis.get('packages', [])}", style="blue")
        
        return {"package_analysis": analysis}
        
    except Exception as e:
        console.print(f"❌ Package analysis failed: {e}", style="red")
        return {"error": f"Package analysis failed: {e}"}

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
    
    code = state.get("corrected_code") or state.get("generated_code")
    
    try:
        prompt = ChatPromptTemplate.from_template(load_prompt("dependency_resolver.txt"))
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
        console.print(f"❌ Dependency resolution failed: {e}", style="red")
        return {"error": f"Dependency resolution failed: {e}"}

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
