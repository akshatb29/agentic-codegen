# msc/agents/docker_workflow_agent.py
"""
Docker Workflow Agent: Integrates Docker functionality into LangGraph workflow
"""
from typing import Dict, Any
from rich.console import Console

from msc.state import AgentState
from .docker_agent import docker_agent
from ..tools.docker_tools import docker_executor

console = Console()

def docker_workflow_agent(state: AgentState) -> Dict[str, Any]:
    """
    Docker agent for LangGraph workflow - handles containerized execution and testing
    """
    print("=" * 60)
    print("🐳 DOCKER AGENT: Containerized Execution")
    print("=" * 60)
    
    try:
        # Get the generated code from state
        code = state.get("corrected_code") or state.get("generated_code", "")
        if not code:
            console.print("⚠️ No code found to execute in Docker", style="yellow")
            return {
                "docker_execution_results": {
                    "success": False,
                    "error": "No code available for Docker execution",
                    "stdout": "",
                    "stderr": "No code provided"
                }
            }
        
        user_request = state.get("user_request", "")
        current_file = state.get("current_file_name", "main.py")
        
        console.print(f"🔍 Analyzing code for Docker execution: {current_file}", style="blue")
        
        # Step 1: Analyze code and generate Docker specification
        spec = docker_agent.analyze_code_and_generate_spec(code, user_request)
        console.print(f"✅ Generated Docker spec: {spec.image_name}", style="green")
        
        # Step 2: Build Docker image with meaningful name
        console.print(f"🔨 Building Docker image for: {current_file}", style="blue")
        
        built_image = docker_executor.build_image_from_spec(
            dockerfile_content=spec.dockerfile_content,
            user_request=user_request,
            filename=current_file
        )
        if not built_image:
            return {
                "docker_execution_results": {
                    "success": False,
                    "error": "Docker image build failed",
                    "stdout": "",
                    "stderr": "Image build failed"
                }
            }
        
        # Step 3: Execute unit test for the specific file
        console.print(f"🧪 Running unit test for: {current_file}", style="blue")
        unit_result = docker_executor.execute_file_unit_test(current_file, code, built_image)
        
        # Step 4: If unit test passes, run integration test
        integration_result = None
        if unit_result.get("success", False):
            console.print("🚀 Running integration test", style="blue")
            integration_result = docker_executor.execute_integration_test(current_file, built_image)
        
        # Determine overall success
        unit_success = unit_result.get("success", False)
        integration_success = integration_result.get("success", False) if integration_result else True
        
        overall_success = unit_success and integration_success
        
        # Prepare results
        results = {
            "success": overall_success,
            "unit_test": unit_result,
            "integration_test": integration_result,
            "docker_spec": {
                "image_name": spec.image_name,
                "base_image": spec.base_image,
                "packages": spec.packages
            }
        }
        
        if overall_success:
            console.print("✅ Docker execution successful!", style="green bold")
        else:
            console.print("❌ Docker execution failed", style="red bold")
            console.print(f"Unit test: {'✅' if unit_success else '❌'}", style="green" if unit_success else "red")
            console.print(f"Integration test: {'✅' if integration_success else '❌'}", style="green" if integration_success else "red")
        
        return {"docker_execution_results": results}
        
    except Exception as e:
        console.print(f"❌ Docker agent error: {e}", style="red bold")
        return {
            "docker_execution_results": {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": str(e)
            }
        }
