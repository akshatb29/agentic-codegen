# msc/tools/agentic_docker.py
"""
Agentic Docker Management: AI-powered Docker image creation and management
"""
import os
import platform
import subprocess
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

# Load environment variables from .env file
def load_env():
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value.strip('"')

load_env()
import docker
from docker.errors import DockerException, BuildError

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    from pydantic import BaseModel, Field
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️ LLM dependencies not available, using fallback mode")

from .filesystem import FilesystemTool

# Lazy imports to avoid circular dependencies
def get_user_interaction():
    from .user_interaction import user_confirmation_tool, user_feedback_tool
    return user_confirmation_tool, user_feedback_tool

if LLM_AVAILABLE:
    class DockerImageSpec(BaseModel):
        """Specification for a Docker image to be built"""
        image_name: str = Field(description="Name for the Docker image")
        task_category: str = Field(description="Category: data_analysis, web_dev, ml, general, etc.")
        base_image: str = Field(description="Base Docker image to use")
        packages: List[str] = Field(description="Python packages to install")
        system_packages: List[str] = Field(description="System packages to install")
        dockerfile_content: str = Field(description="Complete Dockerfile content")
        reasoning: str = Field(description="Why this configuration was chosen")
else:
    # Fallback class if Pydantic not available
    class DockerImageSpec:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

class AgenticDockerManager:
    """Manages Docker images and containers with AI-powered decision making"""
    
    def __init__(self):
        self._llm = None  # Lazy initialization
        self.docker_dir = Path(__file__).parent.parent.parent / "docker"
        self.images_config_file = self.docker_dir / "managed_images.json"
        self.containers_config_file = self.docker_dir / "managed_containers.json"
        self.managed_images = self._load_managed_images()
        self.managed_containers = self._load_managed_containers()
        self.active_container = None  # Current running container for development
    
    @property
    def llm(self):
        """Lazy init of LLM, passing API key directly."""
        if self._llm is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("❌ Missing GOOGLE_API_KEY in environment.")
            if LLM_AVAILABLE:
                self._llm = ChatGoogleGenerativeAI(
                    api_key=api_key,
                    model="gemini-1.5-flash-latest",
                    temperature=0.3
                )
            else:
                raise RuntimeError("❌ LLM dependencies not available")
        return self._llm
        
    def _load_managed_images(self) -> Dict[str, Dict]:
        """Load configuration of managed images"""
        if self.images_config_file.exists():
            try:
                with open(self.images_config_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _load_managed_containers(self) -> Dict[str, Dict]:
        """Load configuration of managed containers"""
        if self.containers_config_file.exists():
            try:
                with open(self.containers_config_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_managed_images(self):
        """Save configuration of managed images"""
        try:
            with open(self.images_config_file, 'w') as f:
                json.dump(self.managed_images, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save image config: {e}")
    
    def _save_managed_containers(self):
        """Save configuration of managed containers"""
        try:
            with open(self.containers_config_file, 'w') as f:
                json.dump(self.managed_containers, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save container config: {e}")
    
    def _extract_dockerfile_content(self, raw_content: str) -> str:
        """Extract clean Dockerfile content by removing markdown code fences"""
        content = raw_content.strip()
        
        # Remove markdown code fences
        lines = content.split('\n')
        cleaned_lines = []
        in_dockerfile_block = False
        
        for line in lines:
            stripped = line.strip()
            
            # Handle code fence markers
            if stripped.startswith('```dockerfile') or stripped.startswith('````dockerfile'):
                in_dockerfile_block = True
                continue
            elif stripped.startswith('```') or stripped.startswith('````'):
                if in_dockerfile_block:
                    break  # End of dockerfile block
                continue
            
            # If we haven't found a dockerfile block yet, assume everything is dockerfile content
            if not any(line.startswith('```') for line in lines):
                in_dockerfile_block = True
            
            # Add all lines when in dockerfile block
            if in_dockerfile_block:
                cleaned_lines.append(line)
        
        cleaned_content = '\n'.join(cleaned_lines).strip()
        
        # Validate that we have essential Dockerfile commands
        if not cleaned_content.startswith('FROM'):
            print(f"⚠️ Warning: Dockerfile doesn't start with FROM, content: {cleaned_content[:100]}")
            
        return cleaned_content

    def analyze_task_and_suggest_image(self, code: str, user_request: str = "") -> DockerImageSpec:
        """Analyze code and user request to suggest appropriate Docker image"""
        print("🤖 Analyzing task requirements for optimal Docker environment...")
        
        # Generate unique task-based image name automatically
        task_hash = abs(hash(code[:500] + user_request)) % 10000
        
        if not LLM_AVAILABLE:
            return self._create_fallback_spec(code, task_hash)
        
        # Detect application type from code analysis
        app_type = "general"
        if any(gui_lib in code for gui_lib in ['tkinter', 'tk']):
            app_type = "gui_app"
        elif any(web_lib in code for web_lib in ['flask', 'django', 'fastapi', 'streamlit']):
            app_type = "web_app"
        elif any(data_lib in code for data_lib in ['pandas', 'numpy', 'matplotlib']):
            app_type = "data_analysis"
        elif any(ml_lib in code for ml_lib in ['sklearn', 'tensorflow', 'torch']):
            app_type = "machine_learning"
        
        prompt = ChatPromptTemplate.from_template("""
        Create a minimal, stable Docker environment for {app_type} applications.
        
        Code to execute:
        ```python
        {code}
        ```
        
        User request: {user_request}
        
        IMPORTANT: Create a minimal stable container with only essential dependencies.
        Let the coding correction flow handle specific package installation at runtime.
        
        Guidelines for {app_type} applications:
        1. Use python:3.11-slim as base
        2. Install only essential system dependencies (compilers, libraries)
        3. DO NOT pre-install specific Python packages - they'll be installed on demand
        4. For GUI apps: Include python3-tk and X11 basics
        5. For data/ML: Include compilers and math libraries (gcc, gfortran, openblas)
        6. For web apps: Include basic networking tools
        7. Set up non-root user with pip user installs enabled
        8. Set PYTHONUNBUFFERED=1 and proper PATH
        
        Generate a minimal Dockerfile with:
        - Task category: {app_type}
        - Unique image name: msc-{app_type}-{task_hash}
        - Only essential system packages
        - NO specific Python packages (just pip, setuptools, wheel)
        - User environment for package installs
        """)
        
        chain = prompt | self.llm.with_structured_output(DockerImageSpec)
        
        try:
            spec = chain.invoke({
                "code": code[:1500],  # Limit code length
                "user_request": user_request[:400],
                "app_type": app_type,
                "task_hash": task_hash
            })
            
            # Ensure proper naming convention with fallback
            if not spec.image_name or spec.image_name == "cancelled":
                spec.image_name = f"msc-{app_type}-{task_hash}"
            elif not spec.image_name.startswith('msc-'):
                spec.image_name = f"msc-{spec.image_name}-{task_hash}"
            
            # Force minimal packages - override AI if it tries to add too many
            spec.packages = ["pip", "setuptools", "wheel"]
            
            # Clean up Dockerfile content to remove any markdown artifacts
            spec.dockerfile_content = self._extract_dockerfile_content(spec.dockerfile_content)
            
            print(f"🎯 App Type: {app_type}")
            print(f"📦 Generated Image: {spec.image_name}")
            print(f"🧠 Reasoning: {spec.reasoning}")
            print(f"⚡ Minimal container - packages installed on demand")
            
            return spec
        except Exception as e:
            print(f"⚠️ AI analysis failed, using fallback: {e}")
            return self._create_fallback_spec(code, task_hash, app_type)
    
    def _create_fallback_spec(self, code: str, task_hash: int = None, app_type: str = None) -> DockerImageSpec:
        """Create a fallback Docker spec with minimal base dependencies"""
        if task_hash is None:
            task_hash = abs(hash(code[:500])) % 10000
            
        # Minimal base packages - let the correction flow handle specific dependencies
        base_packages = ["pip", "setuptools", "wheel"]
        system_packages = []
        
        # Detect application type if not provided
        if not app_type:
            if any(gui_lib in code for gui_lib in ['tkinter', 'tk']):
                app_type = "gui_app"
            elif any(pkg in code for pkg in ["pandas", "numpy", "matplotlib", "seaborn", "plotly"]):
                app_type = "data_analysis"
            elif any(pkg in code for pkg in ["sklearn", "tensorflow", "torch", "keras"]):
                app_type = "machine_learning"  
            elif any(pkg in code for pkg in ["flask", "django", "fastapi", "streamlit"]):
                app_type = "web_app"
            elif any(pkg in code for pkg in ["scipy", "sympy", "networkx"]):
                app_type = "scientific"
            else:
                app_type = "general"
        
        # Configure minimal setup based on application type
        if app_type == "gui_app":
            task_category = "gui_app"
            # Only add GUI system dependencies - packages installed on demand
            system_packages.extend(["python3-tk", "python3-dev"])
            image_name = f"msc-gui-{task_hash}"
        elif app_type == "data_analysis":
            task_category = "data_analysis"
            # Add common system dependencies for data libraries
            system_packages.extend(["gcc", "g++", "gfortran", "libopenblas-dev", "liblapack-dev"])
            image_name = f"msc-data-{task_hash}"
        elif app_type == "machine_learning":
            task_category = "machine_learning"
            # Add system dependencies for ML libraries
            system_packages.extend(["gcc", "g++", "gfortran", "libopenblas-dev", "liblapack-dev"])
            image_name = f"msc-ml-{task_hash}"
        elif app_type == "web_app":
            task_category = "web_dev"
            # Minimal web dependencies
            system_packages.extend(["curl"])
            image_name = f"msc-web-{task_hash}"
        elif app_type == "scientific":
            task_category = "scientific"
            # Scientific computing system dependencies
            system_packages.extend(["gcc", "g++", "gfortran", "libopenblas-dev"])
            image_name = f"msc-sci-{task_hash}"
        else:
            task_category = "general"
            image_name = f"msc-general-{task_hash}"
        
        # Build system packages installation command
        system_install = ""
        if system_packages:
            packages_str = ' '.join(system_packages)
            system_install = f" \\\n    {packages_str}"
        
        # Minimal stable Dockerfile - packages installed on demand
        dockerfile = f"""FROM python:3.11-slim

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install essential system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++{system_install} \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install only essential Python tools - specific packages installed on demand
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Change ownership and switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PIP_USER=1
ENV PATH="/home/appuser/.local/bin:$PATH"

# Default command - can be overridden
CMD ["python", "-u", "/app/script.py"]"""
        
        return DockerImageSpec(
            image_name=image_name,
            task_category=task_category,
            base_image="python:3.11-slim",
            packages=base_packages,  # Minimal base packages
            system_packages=system_packages,
            dockerfile_content=self._extract_dockerfile_content(dockerfile),
            reasoning=f"Minimal stable container for {app_type} tasks - dependencies installed on demand"
        )
    
    def get_or_create_image(self, spec: DockerImageSpec, force_rebuild: bool = False) -> Optional[str]:
        """Get existing image or create new one based on spec - no user prompts"""
        try:
            client = docker.from_env()
            client.ping()
        except DockerException:
            print("❌ Docker daemon not available")
            return None
        
        # Check if we have this image already
        image_key = f"{spec.task_category}_{spec.image_name}"
        
        if not force_rebuild and image_key in self.managed_images:
            image_name = self.managed_images[image_key]["image_name"]
            try:
                client.images.get(image_name)
                print(f"♻️ Reusing existing image: {image_name}")
                return image_name
            except docker.errors.ImageNotFound:
                print(f"⚠️ Cached image {image_name} not found, rebuilding...")
        
        # Use the suggested name directly - no user input required
        final_image_name = spec.image_name
        
        # Create Dockerfile
        dockerfile_path = self.docker_dir / f"Dockerfile.{final_image_name}"
        FilesystemTool.write_file(str(dockerfile_path), spec.dockerfile_content)
        
        print(f"🔨 Building Docker image: {final_image_name}")
        print(f"📄 Using Dockerfile: {dockerfile_path.name}")
        
        try:
            # Build the image
            image, build_logs = client.images.build(
                path=str(self.docker_dir),
                dockerfile=dockerfile_path.name,
                tag=final_image_name,
                rm=True,
                forcerm=True
            )
            
            # Save to managed images
            self.managed_images[image_key] = {
                "image_name": final_image_name,
                "task_category": spec.task_category,
                "packages": spec.packages,
                "created_at": __import__('time').time(),
                "dockerfile_path": str(dockerfile_path)
            }
            self._save_managed_images()
            
            print(f"✅ Successfully built image: {final_image_name}")
            return final_image_name
            
        except BuildError as e:
            print(f"❌ Build failed: {e}")
            print("🤖 Attempting to debug and fix Dockerfile...")
            return self._debug_and_fix_dockerfile(spec, dockerfile_path, str(e))
        except Exception as e:
            print(f"❌ Unexpected build error: {e}")
            return None
    
    def _debug_and_fix_dockerfile(self, spec: DockerImageSpec, dockerfile_path: Path, error_msg: str) -> Optional[str]:
        """Use AI to debug and fix Dockerfile build issues"""
        print("🔍 AI is analyzing the build error...")
        
        debug_prompt = ChatPromptTemplate.from_template("""
        A Docker build failed with this error:
        {error}
        
        Original Dockerfile:
        {dockerfile}
        
        Please analyze the error and provide a corrected Dockerfile that should build successfully.
        Common issues to check:
        - Package availability in repositories
        - Syntax errors
        - Missing dependencies
        - Base image compatibility
        
        Provide only the corrected Dockerfile content.
        """)
        
        try:
            corrected_dockerfile = self.llm.invoke(debug_prompt.format(
                error=error_msg,
                dockerfile=spec.dockerfile_content
            )).content
            
            # Extract clean Dockerfile content
            corrected_dockerfile = self._extract_dockerfile_content(corrected_dockerfile)
            
            print("🔧 Trying corrected Dockerfile...")
            FilesystemTool.write_file(str(dockerfile_path), corrected_dockerfile)
            
            # Try building again
            client = docker.from_env()
            image, build_logs = client.images.build(
                path=str(self.docker_dir),
                dockerfile=dockerfile_path.name,
                tag=spec.image_name,
                rm=True,
                forcerm=True
            )
            
            print(f"✅ Build successful after AI debugging!")
            return spec.image_name
            
        except Exception as e:
            print(f"❌ AI debugging failed: {e}")
            print("🔄 Falling back to basic image...")
            return "python:3.11-slim"  # Ultimate fallback
    
    def check_existing_containers(self) -> List[Dict]:
        """Check for existing development containers that can be reused"""
        try:
            client = docker.from_env()
            containers = []
            
            # Check running containers
            for container in client.containers.list():
                if container.name.startswith('msc-dev-'):
                    containers.append({
                        "id": container.id[:12],
                        "name": container.name,
                        "status": "running",
                        "image": container.image.tags[0] if container.image.tags else "unknown",
                        "created": container.attrs['Created']
                    })
            
            # Check stopped containers that can be restarted
            for container in client.containers.list(all=True):
                if container.name.startswith('msc-dev-') and container.status == 'exited':
                    containers.append({
                        "id": container.id[:12],
                        "name": container.name,
                        "status": "stopped",
                        "image": container.image.tags[0] if container.image.tags else "unknown",
                        "created": container.attrs['Created']
                    })
            
            return containers
        except Exception as e:
            print(f"⚠️ Could not check containers: {e}")
            return []
    
    def create_or_reuse_container(self, image_name: str, code: str, ask_before_plan: bool = True) -> Optional[str]:
        """Create a new development container or reuse existing one"""
        try:
            client = docker.from_env()
            
            # Check for existing containers first (always ask)
            existing = self.check_existing_containers()
            
            if existing and ask_before_plan:
                print(f"\n🔍 Found {len(existing)} existing development containers:")
                for i, container in enumerate(existing):
                    status_emoji = "🟢" if container["status"] == "running" else "🔴"
                    created_time = container.get('created', 'Unknown')[:19].replace('T', ' ')
                    print(f"  {i+1}. {status_emoji} {container['name']} ({container['image']}) - {container['status']} - {created_time}")
                
                user_confirmation_tool, user_feedback_tool = get_user_interaction()
                choice = user_feedback_tool(
                    f"Reuse existing container? (1-{len(existing)}, Enter for new, 'n' for new): ", 
                    allow_empty=True
                ).strip()
                
                if choice.isdigit() and 1 <= int(choice) <= len(existing):
                    selected = existing[int(choice) - 1]
                    container = client.containers.get(selected["id"])
                    
                    if container.status == "exited":
                        print(f"🔄 Restarting container: {container.name}")
                        container.restart()
                        print(f"✅ Container {container.name} is now running")
                    
                    print(f"♻️ Using existing container: {container.name}")
                    self.active_container = container.name
                    return container.name
                elif choice.lower() == 'n':
                    print("🆕 Creating new container as requested")
            
            # Create new development container with session identifier
            session_id = int(__import__('time').time())
            container_name = f"msc-dev-session-{session_id}"
            
            print(f"🚀 Creating new development container: {container_name}")
            print(f"📦 Using image: {image_name}")
            
            # Create container with persistent volume and interactive mode
            container = client.containers.create(
                image_name,
                command="tail -f /dev/null",  # Keep container running
                name=container_name,
                volumes={
                    str(self.docker_dir): {'bind': '/app', 'mode': 'rw'}
                },
                working_dir="/app",
                detach=True,
                tty=True,
                stdin_open=True,
                environment={"PYTHONPATH": "/app", "PYTHONUNBUFFERED": "1"}
            )
            
            container.start()
            print(f"✅ Container {container_name} started successfully")
            
            # Test container is working
            test_result = container.exec_run("python --version", stdout=True, stderr=True)
            if test_result.exit_code == 0:
                python_version = test_result.output.decode('utf-8').strip()
                print(f"🐍 Container ready: {python_version}")
            else:
                print("⚠️ Container started but Python test failed")
            
            # Save container info
            self.managed_containers[container_name] = {
                "container_id": container.id,
                "image_name": image_name,
                "created_at": __import__('time').time(),
                "session_files": [],
                "session_id": session_id
            }
            self._save_managed_containers()
            
            self.active_container = container_name
            return container_name
            
        except Exception as e:
            print(f"❌ Container creation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def execute_in_container(self, container_name: str, code: str, script_name: str = None) -> Dict[str, Any]:
        """Execute code in an existing container, maintaining session state"""
        try:
            client = docker.from_env()
            container = client.containers.get(container_name)
            
            if container.status != 'running':
                print(f"🔄 Starting stopped container: {container_name}")
                container.start()
                # Wait a moment for container to be ready
                __import__('time').sleep(1)
            
            if not script_name:
                script_name = f"session_script_{int(__import__('time').time())}.py"
            
            script_path = self.docker_dir / script_name
            FilesystemTool.write_file(str(script_path), code)
            
            # Track session files
            if container_name in self.managed_containers:
                if script_name not in self.managed_containers[container_name]["session_files"]:
                    self.managed_containers[container_name]["session_files"].append(script_name)
                    self._save_managed_containers()
            
            print(f"🏃 Executing in container: {container_name}")
            print(f"📄 Running script: {script_name}")
            
            # For GUI applications, we need different handling
            if any(gui_lib in code for gui_lib in ['tkinter', 'tk', 'pygame', 'matplotlib.pyplot']):
                print("🖼️ GUI application detected - running with X11 forwarding...")
                # For GUI apps, try to run with display forwarding
                exec_result = container.exec_run(
                    f"python -u /app/{script_name}",
                    stdout=True,
                    stderr=True,
                    tty=False,
                    environment={"DISPLAY": ":0"}
                )
            else:
                # Regular console application
                exec_result = container.exec_run(
                    f"python -u /app/{script_name}",
                    stdout=True,
                    stderr=True,
                    tty=False
                )
            
            stdout = exec_result.output.decode('utf-8') if exec_result.output else ""
            success = exec_result.exit_code == 0
            
            if not success:
                print(f"⚠️ Script execution failed with exit code: {exec_result.exit_code}")
                print(f"📝 Script preserved at: {script_path}")
                print(f"🐳 Container '{container_name}' preserved for debugging")
                print(f"💡 Agentic workflow can write pip install commands to fix dependencies")
            else:
                print(f"✅ Script executed successfully")
            
            return {
                "success": success,
                "stdout": stdout,
                "stderr": "" if success else stdout,
                "container_name": container_name,
                "script_name": script_name,
                "exit_code": exec_result.exit_code
            }
            
        except Exception as e:
            print(f"❌ Container execution error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Container execution failed: {str(e)}",
                "container_name": container_name
            }
    
    def setup_signal_handler(self):
        """Setup graceful shutdown on Ctrl+C"""
        import signal
        
        def signal_handler(signum, frame):
            print(f"\n\n{'='*60}")
            print("🛑 INTERRUPT DETECTED - GRACEFUL SHUTDOWN")
            print("="*60)
            
            user_confirmation_tool, user_feedback_tool = get_user_interaction()
            
            # Handle active container
            if self.active_container:
                save_container = user_confirmation_tool(
                    f"Save development container '{self.active_container}' for future sessions?",
                    default=True
                )
                
                if not save_container:
                    try:
                        client = docker.from_env()
                        container = client.containers.get(self.active_container)
                        container.stop()
                        container.remove()
                        
                        # Remove from managed containers
                        if self.active_container in self.managed_containers:
                            # Clean up session files
                            for file_name in self.managed_containers[self.active_container].get('session_files', []):
                                file_path = self.docker_dir / file_name
                                if file_path.exists():
                                    os.remove(file_path)
                            
                            del self.managed_containers[self.active_container]
                            self._save_managed_containers()
                        
                        print(f"🗑️ Removed container: {self.active_container}")
                    except Exception as e:
                        print(f"⚠️ Could not remove container: {e}")
                else:
                    print(f"💾 Container '{self.active_container}' saved for future use")
            
            # Ask about other managed containers and images
            if self.managed_containers or self.managed_images:
                cleanup_all = user_confirmation_tool(
                    "Clean up all other managed containers and images?",
                    default=False
                )
                if cleanup_all:
                    if self.managed_containers:
                        self._remove_all_managed_containers()
                    if self.managed_images:
                        self._remove_all_managed_images()
                else:
                    print("✅ Other resources preserved for future sessions")
            
            print("👋 Goodbye!")
            exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
    
    def cleanup_session_images(self):
        """Ask user about cleaning up images at end of session"""
        if not self.managed_images and not self.managed_containers:
            return
        
        print("\n" + "="*60)
        print("🗑️  SESSION CLEANUP")
        print("="*60)
        
        if self.managed_containers:
            print(f"Found {len(self.managed_containers)} managed containers:")
            for name, info in self.managed_containers.items():
                created_time = __import__('time').strftime('%Y-%m-%d %H:%M', 
                                                         __import__('time').localtime(info['created_at']))
                files_count = len(info.get('session_files', []))
                print(f"  🐳 {name} ({info['image_name']}) - {files_count} files - Created: {created_time}")
        
        if self.managed_images:
            print(f"Found {len(self.managed_images)} managed images:")
            for key, info in self.managed_images.items():
                created_time = __import__('time').strftime('%Y-%m-%d %H:%M', 
                                                         __import__('time').localtime(info['created_at']))
                print(f"  📦 {info['image_name']} ({info['task_category']}) - Created: {created_time}")
        
        user_confirmation_tool, user_feedback_tool = get_user_interaction()
        
        # Ask about containers first
        if self.managed_containers:
            cleanup_containers = user_confirmation_tool(
                "Remove all managed containers?",
                default=False
            )
            if cleanup_containers:
                self._remove_all_managed_containers()
        
        # Then ask about images
        if self.managed_images:
            cleanup_images = user_confirmation_tool(
                "Remove all managed images to free up disk space?",
                default=False
            )
            if cleanup_images:
                self._remove_all_managed_images()
        
        if not self.managed_images and not self.managed_containers:
            print("✅ All resources cleaned up")
        else:
            print("✅ Keeping selected resources for future reuse")
    
    def _remove_all_managed_containers(self):
        """Remove all managed Docker containers"""
        try:
            client = docker.from_env()
            removed_count = 0
            
            for name, info in list(self.managed_containers.items()):
                try:
                    container = client.containers.get(info['container_id'])
                    container.stop()
                    container.remove()
                    print(f"🗑️ Removed container: {name}")
                    
                    # Clean up session files
                    for file_name in info.get('session_files', []):
                        file_path = self.docker_dir / file_name
                        if file_path.exists():
                            os.remove(file_path)
                    
                    removed_count += 1
                except Exception as e:
                    print(f"⚠️ Could not remove container {name}: {e}")
            
            # Clear managed containers config
            self.managed_containers.clear()
            self._save_managed_containers()
            
            print(f"✅ Cleaned up {removed_count} containers")
            
        except Exception as e:
            print(f"❌ Container cleanup failed: {e}")
    
    def _remove_all_managed_images(self):
        """Remove all managed Docker images"""
        try:
            client = docker.from_env()
            removed_count = 0
            
            for key, info in list(self.managed_images.items()):
                try:
                    client.images.remove(info['image_name'], force=True)
                    print(f"🗑️  Removed: {info['image_name']}")
                    
                    # Remove Dockerfile
                    dockerfile_path = Path(info.get('dockerfile_path', ''))
                    if dockerfile_path.exists():
                        os.remove(dockerfile_path)
                    
                    removed_count += 1
                except Exception as e:
                    print(f"⚠️ Could not remove {info['image_name']}: {e}")
            
            # Clear managed images config
            self.managed_images.clear()
            self._save_managed_images()
            
            print(f"✅ Cleaned up {removed_count} images")
            
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")

# Global instance
docker_manager = AgenticDockerManager()
