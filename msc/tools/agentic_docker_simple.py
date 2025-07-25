# msc/tools/agentic_docker.py
"""
Simplified Agentic Docker Management: 1 container per run with file-specific rebuilds
"""
import os
import json
import docker
import threading
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
    """Simple Docker manager: 1 container per run with file-specific rebuilds"""
    
    def __init__(self):
        self._llm = None  # Lazy initialization
        self.docker_dir = Path(__file__).parent.parent.parent / "docker"
        self.current_image = None  # Single image for current run
        self.current_container = None  # Single container for current run
        self.session_id = int(__import__('time').time()) % 10000

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
            system_packages.extend(["python3-tk", "python3-dev"])
            image_name = f"msc-gui-{task_hash}"
        elif app_type == "data_analysis":
            task_category = "data_analysis"
            system_packages.extend(["gcc", "g++", "gfortran", "libopenblas-dev", "liblapack-dev"])
            image_name = f"msc-data-{task_hash}"
        elif app_type == "machine_learning":
            task_category = "machine_learning"
            system_packages.extend(["gcc", "g++", "gfortran", "libopenblas-dev", "liblapack-dev"])
            image_name = f"msc-ml-{task_hash}"
        elif app_type == "web_app":
            task_category = "web_dev"
            system_packages.extend(["curl"])
            image_name = f"msc-web-{task_hash}"
        elif app_type == "scientific":
            task_category = "scientific"
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
            packages=base_packages,
            system_packages=system_packages,
            dockerfile_content=self._extract_dockerfile_content(dockerfile),
            reasoning=f"Minimal stable container for {app_type} tasks - dependencies installed on demand"
        )

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

    def build_base_image_async(self, code: str, user_request: str = "") -> str:
        """Build base image in parallel when plan is approved"""
        
        # Analyze and create base spec
        spec = self.analyze_task_and_suggest_image(code, user_request)
        self.current_image = f"msc-session-{self.session_id}"
        
        print(f"🚀 Building base image in parallel: {self.current_image}")
        
        def build_worker():
            try:
                client = docker.from_env()
                dockerfile_path = self.docker_dir / f"Dockerfile.{self.current_image}"
                FilesystemTool.write_file(str(dockerfile_path), spec.dockerfile_content)
                
                client.images.build(
                    path=str(self.docker_dir),
                    dockerfile=dockerfile_path.name,
                    tag=self.current_image,
                    rm=True,
                    forcerm=True
                )
                print(f"✅ Base image ready: {self.current_image}")
            except Exception as e:
                print(f"❌ Base image build failed: {e}")
        
        # Start build in background
        build_thread = threading.Thread(target=build_worker, daemon=True)
        build_thread.start()
        
        return self.current_image
    
    def rebuild_with_entrypoint(self, target_file: str) -> str:
        """Rebuild image with specific file as entrypoint"""
        if not self.current_image:
            raise RuntimeError("No base image available")
        
        file_specific_image = f"{self.current_image}-{target_file.replace('.py', '').replace('/', '_')}"
        
        try:
            client = docker.from_env()
            
            # Read base dockerfile
            base_dockerfile_path = self.docker_dir / f"Dockerfile.{self.current_image}"
            if not base_dockerfile_path.exists():
                raise FileNotFoundError(f"Base dockerfile not found: {base_dockerfile_path}")
            
            base_dockerfile = FilesystemTool.read_file(str(base_dockerfile_path))
            
            # Update CMD to run specific file
            lines = base_dockerfile.split('\n')
            updated_lines = []
            cmd_updated = False
            
            for line in lines:
                if line.strip().startswith('CMD'):
                    updated_lines.append(f'CMD ["python", "-u", "/app/{target_file}"]')
                    cmd_updated = True
                    print(f"📝 Updated CMD to run: {target_file}")
                else:
                    updated_lines.append(line)
            
            if not cmd_updated:
                updated_lines.append(f'CMD ["python", "-u", "/app/{target_file}"]')
                print(f"📝 Added CMD to run: {target_file}")
            
            # Write updated dockerfile
            file_dockerfile_path = self.docker_dir / f"Dockerfile.{file_specific_image}"
            updated_dockerfile = '\n'.join(updated_lines)
            FilesystemTool.write_file(str(file_dockerfile_path), updated_dockerfile)
            
            # Rebuild image
            print(f"🔨 Rebuilding for {target_file}: {file_specific_image}")
            client.images.build(
                path=str(self.docker_dir),
                dockerfile=file_dockerfile_path.name,
                tag=file_specific_image,
                rm=True,
                forcerm=True
            )
            
            print(f"✅ File-specific image ready: {file_specific_image}")
            return file_specific_image
            
        except Exception as e:
            print(f"❌ Rebuild failed for {target_file}: {e}")
            return self.current_image  # Fallback to base image
    
    def execute_file_unit_test(self, target_file: str, code: str) -> Dict[str, Any]:
        """Execute and unit test a specific file"""
        try:
            # Write code to file
            script_path = self.docker_dir / target_file
            FilesystemTool.write_file(str(script_path), code)
            
            # Rebuild image with this file as entrypoint
            file_image = self.rebuild_with_entrypoint(target_file)
            
            # Run container to test the file
            client = docker.from_env()
            print(f"🧪 Unit testing: {target_file}")
            
            result = client.containers.run(
                file_image,
                volumes={str(self.docker_dir): {'bind': '/app', 'mode': 'rw'}},
                working_dir="/app",
                remove=True,
                stdout=True,
                stderr=True
            )
            
            stdout = result.decode('utf-8') if result else ""
            success = True  # If we got here, no exception occurred
            
            print(f"✅ Unit test passed: {target_file}")
            
            return {
                "success": success,
                "stdout": stdout,
                "stderr": "",
                "target_file": target_file,
                "test_type": "unit_test"
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Unit test failed: {target_file} - {error_msg}")
            
            return {
                "success": False,
                "stdout": "",
                "stderr": error_msg,
                "target_file": target_file,
                "test_type": "unit_test"
            }
    
    def execute_integration_test(self, main_file: str = "main.py") -> Dict[str, Any]:
        """Execute full integration test with main file as entrypoint"""
        try:
            # Rebuild with main file as entrypoint
            main_image = self.rebuild_with_entrypoint(main_file)
            
            # Run full application test
            client = docker.from_env()
            print(f"🚀 Integration testing: {main_file}")
            
            result = client.containers.run(
                main_image,
                volumes={str(self.docker_dir): {'bind': '/app', 'mode': 'rw'}},
                working_dir="/app",
                remove=True,
                stdout=True,
                stderr=True
            )
            
            stdout = result.decode('utf-8') if result else ""
            success = True
            
            print(f"✅ Integration test passed: {main_file}")
            
            return {
                "success": success,
                "stdout": stdout,
                "stderr": "",
                "main_file": main_file,
                "test_type": "integration_test"
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Integration test failed: {main_file} - {error_msg}")
            
            return {
                "success": False,
                "stdout": "",
                "stderr": error_msg,
                "main_file": main_file,
                "test_type": "integration_test"
            }
    
    def cleanup_session(self):
        """Clean up current session images"""
        try:
            client = docker.from_env()
            
            # Remove all session images
            for image in client.images.list():
                for tag in image.tags:
                    if f"msc-session-{self.session_id}" in tag:
                        try:
                            client.images.remove(tag, force=True)
                            print(f"🗑️ Removed: {tag}")
                        except Exception as e:
                            print(f"⚠️ Could not remove {tag}: {e}")
            
            # Clean up dockerfiles
            for dockerfile in self.docker_dir.glob(f"Dockerfile.msc-session-{self.session_id}*"):
                try:
                    dockerfile.unlink()
                    print(f"🗑️ Removed: {dockerfile.name}")
                except Exception as e:
                    print(f"⚠️ Could not remove {dockerfile.name}: {e}")
                    
            print(f"✅ Session {self.session_id} cleaned up")
            
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")

# Global instance
docker_manager = AgenticDockerManager()
