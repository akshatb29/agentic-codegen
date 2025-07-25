# msc/tools/execution.py
import os
import platform
import subprocess
import uuid
import time
from typing import Dict, Any
from pathlib import Path
from rich.console import Console
import docker
from docker.errors import DockerException, BuildError, ContainerError

from .filesystem import FilesystemTool
from .user_interaction import user_confirmation_tool, user_feedback_tool
# Use compatibility wrapper for now to avoid breaking existing functionality
from .agentic_docker import docker_manager

console = Console()
DOCKER_DIR = Path(__file__).parent.parent.parent / "docker"

def _run_local(file_path: str) -> Dict[str, Any]:
    """Executes a Python script on the host system with some isolation."""
    console.log(f"🔒 [Local Executor] Running '{file_path}' in isolated mode...")
    command = ["python" if platform.system() == "Windows" else "python3", "-I", file_path]
    timeout = 15
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False,
            cwd=os.path.dirname(os.path.abspath(file_path)) or "."
        )
        return {"success": process.returncode == 0, "stdout": process.stdout, "stderr": process.stderr}
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": f"TimeoutError: Execution timed out after {timeout}s."}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e)}


def _run_docker_enhanced(code: str, user_request: str = "", target_file: str = "main.py", project_name: str = None) -> Dict[str, Any]:
    """Enhanced Docker execution with container reuse and project-based naming"""
    console.log(f"🐳 [Docker] Enhanced execution for project: {project_name or 'untitled'}")
    
    try:
        client = docker.from_env()
        client.ping()
    except DockerException:
        return {"success": False, "stdout": "", "stderr": "Docker daemon is not running. Please start Docker."}

    # Setup signal handler for graceful shutdown
    docker_manager.setup_signal_handler()

    # Check for existing containers and offer reuse options
    existing_containers = docker_manager.check_existing_containers()
    selected_container = None
    
    if existing_containers:
        print(f"\n💡 Found {len(existing_containers)} existing development containers")
        for i, container in enumerate(existing_containers):
            status_emoji = "🟢" if container["status"] == "running" else "🔴"
            created_time = container.get('created', 'Unknown')[:19].replace('T', ' ')
            print(f"  {i+1}. {status_emoji} {container['name']} ({container['image']}) - {container['status']}")
        
        choice = user_feedback_tool(
            f"Reuse existing container? (1-{len(existing_containers)}, 'n' for new): ", 
            allow_empty=True
        ).strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(existing_containers):
            selected = existing_containers[int(choice) - 1]
            selected_container = selected["name"]
            print(f"♻️ Using existing container: {selected_container}")
            
            # Update container start point for the new target file
            try:
                # We'll use the existing container's image but update the start point
                container_obj = client.containers.get(selected["id"])
                if container_obj.status == "exited":
                    container_obj.restart()
                
                # Execute in the existing container with updated entry point
                result = docker_manager.execute_in_container_with_entrypoint(
                    selected_container, code, target_file
                )
                return result
                
            except Exception as e:
                print(f"⚠️ Failed to reuse container: {e}")
                selected_container = None

    # If no container selected or reuse failed, create new project-based container
    if not selected_container:
        # Use agentic system to suggest and prepare a minimal Docker image
        try:
            spec = docker_manager.analyze_task_and_suggest_image(code, user_request)
            image_name = docker_manager.get_or_create_image(spec)
            if not image_name:
                return {"success": False, "stdout": "", "stderr": "Failed to prepare Docker environment"}
        except Exception as e:
            console.log(f"⚠️ Agentic image creation failed: {e}")
            return {"success": False, "stdout": "", "stderr": str(e)}
        
        # Create project-based container
        try:
            container_name = docker_manager.create_project_container(
                image_name, code, target_file, project_name or "untitled-project"
            )
            if not container_name:
                console.log("⚠️ Container creation failed, falling back to one-time execution")
                return _run_docker_oneshot(image_name, code)
            
            # Execute in the new container
            result = docker_manager.execute_in_container_with_entrypoint(
                container_name, code, target_file
            )
            
            if not result["success"]:
                console.log("❌ [Docker] Container execution failed.")
                console.log(f"🔍 Exit code: {result.get('exit_code', 'unknown')}")
                console.log(f"💾 Container '{container_name}' preserved for debugging")
                console.log(f"📄 Target file: {target_file}")
                console.log("💡 Container ready for correction flow - packages can be installed dynamically")
                
                # Add container info for correction flow
                result["container_available"] = True
                result["container_supports_pip_install"] = True
            else:
                console.log("✅ [Docker] Code executed successfully in container.")
            
            return result

        except Exception as e:
            console.log(f"❌ [Docker] Container management failed: {e}")
            # Fallback to one-time execution
            return _run_docker_oneshot(image_name, code)


def _run_docker(code: str, user_request: str = "", custom_image_name: str = None) -> Dict[str, Any]:
    """Runs code in minimal stable Docker container, with dynamic package installation capability."""
    console.log(f"🐳 [Docker Executor] Setting up minimal stable environment...")
    try:
        client = docker.from_env()
        client.ping()
    except DockerException:
        return {"success": False, "stdout": "", "stderr": "Docker daemon is not running. Please start Docker."}

    # Setup signal handler for graceful shutdown
    docker_manager.setup_signal_handler()

    # Check for existing containers before any planning or building
    existing_containers = docker_manager.check_existing_containers()
    if existing_containers:
        print(f"\n💡 Found {len(existing_containers)} existing development containers")
        
    # Use agentic system to suggest and prepare a minimal Docker image
    try:
        spec = docker_manager.analyze_task_and_suggest_image(code, user_request)
        if custom_image_name:
            spec.image_name = custom_image_name
        image_name = docker_manager.get_or_create_image(spec)
        if not image_name:
            return {"success": False, "stdout": "", "stderr": "Failed to prepare Docker environment"}
    except Exception as e:
        console.log(f"⚠️ Agentic image creation failed: {e}")
        return {"success": False, "stdout": "", "stderr": str(e)}
    
    # Try to create or reuse a development container (asks user about existing containers)
    try:
        container_name = docker_manager.create_or_reuse_container(image_name, code, ask_before_plan=True)
        if not container_name:
            # Fallback to one-time container execution
            console.log("⚠️ Container management failed, falling back to one-time execution")
            return _run_docker_oneshot(image_name, code)
        
        # Execute in the persistent container
        result = docker_manager.execute_in_container(container_name, code)
        
        if not result["success"]:
            console.log("❌ [Docker] Container execution failed.")
            console.log(f"🔍 Exit code: {result.get('exit_code', 'unknown')}")
            console.log(f"💾 Container '{container_name}' preserved for debugging")
            console.log(f"📄 Script: {result.get('script_name', 'unknown')}")
            console.log("💡 Container ready for correction flow - packages can be installed dynamically")
            
            # Add container info for correction flow
            result["container_available"] = True
            result["container_supports_pip_install"] = True
        else:
            console.log("✅ [Docker] Code executed successfully in container.")
        
        return result

    except Exception as e:
        console.log(f"❌ [Docker] Container management failed: {e}")
        # Fallback to one-time execution
        return _run_docker_oneshot(image_name, code)


def _run_docker_oneshot(image_name: str, code: str) -> Dict[str, Any]:
    """Fallback: Run code in a one-time container (legacy behavior)"""
    try:
        client = docker.from_env()
        
        # Create a unique script for this execution
        script_name = f"exec_script_{uuid.uuid4()}.py"
        script_path_in_host = DOCKER_DIR / script_name
        FilesystemTool.write_file(str(script_path_in_host), code)
        
        console.log(f"🏃 [Docker] Executing code in one-time container...")
        # Run container with volume mount to access the new script
        result = client.containers.run(
            image_name,
            command=f"python -u /app/{script_name}",
            volumes={str(DOCKER_DIR): {'bind': '/app', 'mode': 'ro'}},
            detach=False,
            remove=True,
            stdout=True,
            stderr=True
        )
        
        # Result should be bytes from container output
        stdout = result.decode('utf-8') if isinstance(result, bytes) else str(result)
        return {"success": True, "stdout": stdout, "stderr": ""}

    except BuildError as e:
        console.log("❌ [Docker] Build failed.")
        return {"success": False, "stdout": "", "stderr": f"BuildError: {e.msg}"}
    except ContainerError as e:
        console.log("❌ [Docker] Container run failed.")
        try:
            stderr_msg = f"Container failed with exit code {e.exit_status}: {str(e)}"
        except AttributeError:
            stderr_msg = f"Container failed: {str(e)}"
        return {"success": False, "stdout": "", "stderr": stderr_msg}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": f"An unexpected Docker error occurred: {str(e)}"}
    finally:
        # Cleanup the execution script
        if script_path_in_host.exists():
            os.remove(script_path_in_host)


def run_code(code: str, file_path: str, mode: str, user_request: str = "", project_name: str = None) -> Dict[str, Any]:
    """Top-level function to select execution mode with improved Docker workflow."""
    if not file_path.endswith(".py"):
        return {"success": True, "stdout": "Non-executable file type.", "stderr": ""}
    
    FilesystemTool.write_file(file_path, code)
    
    if mode == 'docker':
        # Enhanced Docker workflow with container options
        return _run_docker_enhanced(code, user_request, file_path, project_name)
    else: # 'local'
        return _run_local(file_path)


def _run_docker_enhanced(code: str, user_request: str = "", target_file: str = "main.py", project_name: str = None) -> Dict[str, Any]:
    """Enhanced Docker execution with container reuse and project-based naming"""
    console.log(f"🐳 [Docker] Enhanced execution for project: {project_name or 'untitled'}")
    
    try:
        client = docker.from_env()
        client.ping()
    except DockerException:
        return {"success": False, "stdout": "", "stderr": "Docker daemon is not running. Please start Docker."}

    # Setup signal handler for graceful shutdown
    docker_manager.setup_signal_handler()

    # Check for existing containers and offer reuse options
    existing_containers = docker_manager.check_existing_containers()
    selected_container = None
    
    if existing_containers:
        print(f"\n💡 Found {len(existing_containers)} existing development containers")
        for i, container in enumerate(existing_containers):
            status_emoji = "🟢" if container["status"] == "running" else "🔴"
            created_time = container.get('created', 'Unknown')[:19].replace('T', ' ')
            print(f"  {i+1}. {status_emoji} {container['name']} ({container['image']}) - {container['status']}")
        
        choice = user_feedback_tool(
            f"Reuse existing container? (1-{len(existing_containers)}, 'n' for new): ", 
            allow_empty=True
        ).strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(existing_containers):
            selected = existing_containers[int(choice) - 1]
            selected_container = selected["name"]
            print(f"♻️ Using existing container: {selected_container}")
            
            # Update container start point for the new target file
            try:
                # We'll use the existing container's image but update the start point
                container_obj = client.containers.get(selected["id"])
                if container_obj.status == "exited":
                    container_obj.restart()
                
                # Execute in the existing container with updated entry point
                result = docker_manager.execute_in_container_with_entrypoint(
                    selected_container, code, target_file
                )
                return result
                
            except Exception as e:
                print(f"⚠️ Failed to reuse container: {e}")
                selected_container = None

    # If no container selected or reuse failed, create new project-based container
    if not selected_container:
        # Use agentic system to suggest and prepare a minimal Docker image
        try:
            spec = docker_manager.analyze_task_and_suggest_image(code, user_request)
            image_name = docker_manager.get_or_create_image(spec)
            if not image_name:
                return {"success": False, "stdout": "", "stderr": "Failed to prepare Docker environment"}
        except Exception as e:
            console.log(f"⚠️ Agentic image creation failed: {e}")
            return {"success": False, "stdout": "", "stderr": str(e)}
        
        # Create project-based container
        try:
            container_name = docker_manager.create_project_container(
                image_name, code, target_file, project_name or "untitled-project"
            )
            if not container_name:
                console.log("⚠️ Container creation failed, falling back to one-time execution")
                return _run_docker_oneshot(image_name, code)
            
            # Execute in the new container
            result = docker_manager.execute_in_container_with_entrypoint(
                container_name, code, target_file
            )
            
            if not result["success"]:
                console.log("❌ [Docker] Container execution failed.")
                console.log(f"🔍 Exit code: {result.get('exit_code', 'unknown')}")
                console.log(f"💾 Container '{container_name}' preserved for debugging")
                console.log(f"📄 Target file: {target_file}")
                console.log("💡 Container ready for correction flow - packages can be installed dynamically")
                
                # Add container info for correction flow
                result["container_available"] = True
                result["container_supports_pip_install"] = True
            else:
                console.log("✅ [Docker] Code executed successfully in container.")
            
            return result

        except Exception as e:
            console.log(f"❌ [Docker] Container management failed: {e}")
            # Fallback to one-time execution
            return _run_docker_oneshot(image_name, code)


# To test this file independently: python -m msc.tools.execution
if __name__ == '__main__':
    console.rule("🧪 Testing Enhanced Docker Execution System")
    
    # Test scenarios with different code types
    test_scenarios = [
        {
            "name": "Basic Python",
            "code": '''
import sys
import os
print("🐍 Basic Python Test")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print("✅ Basic test successful!")
''',
            "expected_category": "general"
        },
        {
            "name": "Data Analysis",
            "code": '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("📊 Data Analysis Test")
data = pd.DataFrame({'x': [1,2,3,4], 'y': [2,4,6,8]})
print(f"Data shape: {data.shape}")
print(f"Mean: {data.mean().to_dict()}")
print("✅ Data analysis test successful!")
''',
            "expected_category": "data_analysis"
        },
        {
            "name": "GUI Application", 
            "code": '''
import tkinter as tk

print("🖼️ GUI Application Test")
root = tk.Tk()
root.title("Test Window")
print(f"Tkinter root created: {root}")
print("✅ GUI test successful!")
# Don't run mainloop in test
''',
            "expected_category": "gui_app"
        }
    ]
    
    # Test each scenario
    for i, scenario in enumerate(test_scenarios, 1):
        console.print(f"\n{'='*60}")
        console.print(f"🧪 Test {i}/3: {scenario['name']}")
        console.print(f"Expected category: {scenario['expected_category']}")
        console.print("="*60)
        
        # Test Docker mode with minimal container
        file_name = f"test_minimal_{i}.py"
        docker_result = run_code(
            scenario['code'], 
            file_name, 
            "docker", 
            f"Test {scenario['name']} functionality with minimal container"
        )
        
        console.print(f"\n📋 Test {i} Results:")
        console.print(f"  Success: {docker_result['success']}")
        if docker_result['success']:
            console.print(f"  Output Preview: {docker_result['stdout'][:200]}...")
        else:
            console.print(f"  Error: {docker_result['stderr']}")
            console.print(f"  Container Available: {docker_result.get('container_available', False)}")
            console.print(f"  Can Install Packages: {docker_result.get('container_supports_pip_install', False)}")
        
        # Cleanup test file
        if os.path.exists(file_name):
            os.remove(file_name)
    
    # Final cleanup prompt
    console.print(f"\n{'='*60}")
    console.print("🏁 All tests complete!")
    docker_manager.cleanup_session_images()

__all__ = ['run_code']
