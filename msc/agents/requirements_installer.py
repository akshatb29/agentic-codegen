# msc/agents/requirements_installer.py
"""
Requirements Installer Agent - Install dependencies from planner predictions
"""
from typing import Dict, Any, List
from rich.console import Console
from msc.state import AgentState
from msc.tools.simple_project_docker import simple_docker_manager

console = Console()

def requirements_installer_agent(state: AgentState) -> Dict[str, Any]:
    """Install requirements predicted by the planner"""
    console.rule("[bold green]REQUIREMENTS INSTALLER: Installing Planned Dependencies[/bold green]")
    
    # Check if requirements already installed to avoid repetition
    if state.get("requirements_installed_successfully"):
        console.print("✅ Requirements already installed, skipping repetitive installation", style="green")
        return {"requirements_installation": {"success": True, "installed": [], "message": "Already installed"}}
    
    # Get requirements from software design
    software_design = state.get("software_design", {})
    if isinstance(software_design, dict):
        requirements = software_design.get("requirements", [])
        language = software_design.get("language", "python")
        framework = software_design.get("framework", "")
    else:
        # Handle Pydantic model
        requirements = getattr(software_design, 'requirements', [])
        language = getattr(software_design, 'language', 'python')
        framework = getattr(software_design, 'framework', '')
    
    if not requirements:
        console.print("📦 No requirements specified by planner", style="dim")
        return {"requirements_installation": {"success": True, "installed": [], "message": "No requirements"}}
    
    console.print(f"📋 Planner predicted requirements: {requirements}", style="blue")
    console.print(f"🔍 Language: {language}, Framework: {framework}", style="dim")
    
    # For local execution mode, install packages locally  
    if state.get("execution_mode") == "local":
        console.print("💻 Local execution mode - installing packages locally", style="yellow")
        
        # Actually install packages using pip
        import subprocess
        import sys
        
        installed = []
        failed = []
        
        for req in requirements:
            try:
                console.print(f"📦 Installing {req}...", style="cyan")
                result = subprocess.run([sys.executable, '-m', 'pip', 'install', req], 
                                      capture_output=True, text=True, check=True)
                installed.append(req)
                console.print(f"✅ Successfully installed {req}", style="green")
            except subprocess.CalledProcessError as e:
                failed.append(req)
                console.print(f"❌ Failed to install {req}: {e.stderr}", style="red")
            except Exception as e:
                failed.append(req)
                console.print(f"❌ Error installing {req}: {e}", style="red")
        
        success = len(failed) == 0
        return {
            "requirements_installation": {
                "success": success, 
                "installed": installed, 
                "failed": failed,
                "message": f"Installed {len(installed)}/{len(requirements)} packages"
            },
            "requirements_installed_successfully": success
        }
    
    # Ensure we have a Docker project setup
    if not simple_docker_manager.current_project:
        console.print("🔧 Setting up Docker project for requirements installation...", style="yellow")
        project_name = state.get("project_name", "planned-project")
        simple_docker_manager.get_or_create_project(
            user_request=state.get("user_request", ""),
            suggested_project_name=project_name,
            language=language,
            ask_reuse=False  # Don't ask again if already asked in planner
        )
    
    # Install requirements
    installed_packages = []
    failed_packages = []
    
    try:
        # Get the Docker container
        import docker
        client = docker.from_env()
        
        # Get current container name
        session_name = simple_docker_manager.session_dir.name if simple_docker_manager.session_dir else f"session-{language}"
        if session_name == f"session-{language}":
            container_name = f"msc-{language}-reusable"
        else:
            timestamp = session_name.split('-')[-1]
            container_name = f"msc-{language}-{timestamp}"
        
        console.print(f"🐳 Installing in container: {container_name}", style="cyan")
        
        try:
            container = client.containers.get(container_name)
            if container.status != 'running':
                console.print("🔄 Starting existing container...", style="yellow")
                container.start()
        except docker.errors.NotFound:
            console.print("🏗️ Container not found, creating new one...", style="yellow")
            
            # Create the container using the docker manager's image creation logic
            image_name = f"msc-{language}"
            
            # Ensure image exists first
            if not simple_docker_manager._ensure_image(language, image_name):
                console.print("❌ Failed to create Docker image", style="red")
                return {"requirements_installation": {"success": False, "error": "Image creation failed"}}
            
            # Create container
            try:
                container = client.containers.run(
                    image_name,
                    command="sleep infinity",
                    name=container_name,
                    working_dir="/projects",
                    detach=True,
                    remove=False
                )
                console.print("✅ Container created successfully", style="green")
                
                # Wait a moment for container to be fully ready
                import time
                time.sleep(2)
            except Exception as e:
                console.print(f"❌ Container creation failed: {e}", style="red")
                return {"requirements_installation": {"success": False, "error": f"Container creation failed: {e}"}}
        
        if not container:
            console.print("❌ No container available", style="red")
            return {"requirements_installation": {"success": False, "error": "No container available"}}
        
        # Install each package with enhanced error handling and reasoning
        for package in requirements:
            console.print(f"📦 Installing: {package}", style="blue")
            
            if language == "python":
                # Apply package name mappings with reasoning
                package_map = {
                    'tensorflow': 'tensorflow-cpu',  # CPU version for faster install and broader compatibility
                    'torch': 'torch torchvision torchaudio',  # Common PyTorch bundle
                    'cv2': 'opencv-python',  # Standard OpenCV Python package
                    'PIL': 'Pillow',  # Modern PIL replacement
                    'sklearn': 'scikit-learn',  # Full package name
                    'skimage': 'scikit-image'  # Full package name
                }
                actual_package = package_map.get(package, package)
                
                if actual_package != package:
                    console.print(f"🔄 Mapping {package} → {actual_package}", style="dim")
                
                install_cmd = f"pip install {actual_package}"
                result = container.exec_run(["bash", "-c", install_cmd])
                
                if result.exit_code == 0:
                    installed_packages.append(package)
                    console.print(f"✅ Installed: {package}", style="green")
                else:
                    failed_packages.append(package)
                    error_output = result.output.decode() if result.output else "Unknown error"
                    console.print(f"❌ Failed to install {package}: {error_output[:200]}", style="red")
                    
                    # Enhanced error reasoning
                    if "No module named" in error_output or "Could not find" in error_output:
                        console.print(f"💡 Reasoning: Package '{package}' may not exist or name is incorrect", style="yellow")
                    elif "Permission denied" in error_output:
                        console.print(f"💡 Reasoning: Permission issue, container may need privileged access", style="yellow")
                    elif "Connection" in error_output or "timeout" in error_output:
                        console.print(f"💡 Reasoning: Network connectivity issue", style="yellow")
                    
            elif language == "nodejs":
                result = container.exec_run(["npm", "install", package])
                if result.exit_code == 0:
                    installed_packages.append(package)
                    console.print(f"✅ Installed: {package}", style="green")
                else:
                    failed_packages.append(package)
                    error_output = result.output.decode() if result.output else "Unknown error"
                    console.print(f"❌ Failed to install {package}: {error_output[:200]}", style="red")
        
        # Create requirements file for future reference
        if language == "python" and installed_packages:
            requirements_content = "\n".join(installed_packages)
            container.exec_run(["bash", "-c", f"echo '{requirements_content}' > /projects/requirements.txt"])
            console.print("📄 Created requirements.txt", style="dim")
        
        success = len(failed_packages) == 0
        
        # Enhanced reporting with reasoning
        if failed_packages:
            console.print(f"⚠️ Installation summary: {len(installed_packages)} succeeded, {len(failed_packages)} failed", style="yellow")
            console.print(f"💭 Failed packages may be retried during code execution", style="dim")
        else:
            console.print(f"🎉 All {len(installed_packages)} packages installed successfully!", style="green")
        
        return {
            "requirements_installation": {
                "success": success,
                "installed": installed_packages,
                "failed": failed_packages,
                "total": len(requirements),
                "message": f"Installed {len(installed_packages)}/{len(requirements)} packages"
            },
            "requirements_installed_successfully": success  # Flag to prevent repetition
        }
        
    except Exception as e:
        console.print(f"❌ Requirements installation failed: {e}", style="red")
        return {
            "requirements_installation": {
                "success": False,
                "error": str(e),
                "installed": installed_packages,
                "failed": requirements
            }
        }

def smart_requirements_predictor(user_request: str, language: str = "python") -> List[str]:
    """Smart prediction of requirements based on user request keywords"""
    
    # Define requirement patterns
    patterns = {
        # AI/ML keywords
        'ml_keywords': ['machine learning', 'ml', 'neural network', 'deep learning', 'ai', 'artificial intelligence'],
        'cv_keywords': ['computer vision', 'image processing', 'opencv', 'cv', 'image recognition', 'face detection'],
        'nlp_keywords': ['natural language', 'nlp', 'text processing', 'sentiment analysis', 'language model'],
        'data_keywords': ['data analysis', 'data science', 'statistics', 'visualization', 'pandas', 'numpy'],
        'web_keywords': ['web app', 'website', 'api', 'server', 'flask', 'django', 'fastapi'],
        'gui_keywords': ['gui', 'interface', 'desktop app', 'tkinter', 'qt', 'kivy'],
        'cnn_keywords': ['cnn', 'convolutional', 'imagenet', 'resnet', 'tensorflow', 'keras', 'pytorch']
    }
    
    # Package mappings
    package_sets = {
        'ml_keywords': ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn'],
        'cv_keywords': ['opencv-python', 'numpy', 'matplotlib', 'PIL', 'scikit-image'],
        'nlp_keywords': ['nltk', 'spacy', 'transformers', 'numpy', 'pandas'],
        'data_keywords': ['numpy', 'pandas', 'matplotlib', 'seaborn', 'jupyter'],
        'web_keywords': ['flask', 'requests', 'jinja2'] if 'flask' in user_request.lower() else ['django'] if 'django' in user_request.lower() else ['fastapi', 'uvicorn'],
        'gui_keywords': ['tkinter'] if language == 'python' else [],
        'cnn_keywords': ['tensorflow-cpu', 'numpy', 'matplotlib', 'PIL', 'scikit-learn']
    }
    
    predicted_packages = set()
    request_lower = user_request.lower()
    
    # Check for keyword matches
    for category, keywords in patterns.items():
        if any(keyword in request_lower for keyword in keywords):
            predicted_packages.update(package_sets.get(category, []))
    
    # Always include basic packages for data-related tasks
    if any(word in request_lower for word in ['data', 'analysis', 'plot', 'graph', 'chart']):
        predicted_packages.update(['numpy', 'matplotlib'])
    
    # Remove built-in packages
    builtin_packages = {'os', 'sys', 'json', 'time', 're', 'math', 'random', 'datetime', 'collections'}
    predicted_packages = predicted_packages - builtin_packages
    
    return list(predicted_packages)
