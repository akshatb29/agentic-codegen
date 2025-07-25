# msc/agents/docker_agent.py
"""
Docker Agent: AI-powered Docker image creation and specification
"""
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field
    from msc.tools import get_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️ LLM dependencies not available, using fallback mode")

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

class DockerAgent:
    """Agent responsible for analyzing code and generating Docker specifications"""
    
    def __init__(self):
        self._llm = None  # Lazy initialization

    @property
    def llm(self):
        """Lazy init of LLM, passing API key directly."""
        if self._llm is None:
            if LLM_AVAILABLE:
                self._llm = get_llm("docker_agent", temperature=0.3)
            else:
                raise RuntimeError("❌ LLM dependencies not available")
        return self._llm

    def analyze_code_and_generate_spec(self, code: str, user_request: str = "") -> DockerImageSpec:
        """Analyze code and user request to generate Docker image specification"""
        print("🤖 Docker Agent: Analyzing task requirements for optimal Docker environment...")
        
        # Generate unique task-based image name automatically
        task_hash = abs(hash(code[:500] + user_request)) % 10000
        
        if not LLM_AVAILABLE:
            return self._create_fallback_spec(code, task_hash)
        
        # Detect application type from code analysis
        app_type = self._detect_app_type(code)
        
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

    def _detect_app_type(self, code: str) -> str:
        """Detect application type from code analysis"""
        if any(gui_lib in code for gui_lib in ['tkinter', 'tk']):
            return "gui_app"
        elif any(web_lib in code for web_lib in ['flask', 'django', 'fastapi', 'streamlit']):
            return "web_app"
        elif any(data_lib in code for data_lib in ['pandas', 'numpy', 'matplotlib']):
            return "data_analysis"
        elif any(ml_lib in code for ml_lib in ['sklearn', 'tensorflow', 'torch']):
            return "machine_learning"
        else:
            return "general"
    
    def _create_fallback_spec(self, code: str, task_hash: int = None, app_type: str = None) -> DockerImageSpec:
        """Create a fallback Docker spec with minimal base dependencies"""
        if task_hash is None:
            task_hash = abs(hash(code[:500])) % 10000
            
        # Minimal base packages - let the correction flow handle specific dependencies
        base_packages = ["pip", "setuptools", "wheel"]
        system_packages = []
        
        # Detect application type if not provided
        if not app_type:
            app_type = self._detect_fallback_app_type(code)
        
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

    def _detect_fallback_app_type(self, code: str) -> str:
        """Fallback app type detection when LLM is not available"""
        if any(gui_lib in code for gui_lib in ['tkinter', 'tk']):
            return "gui_app"
        elif any(pkg in code for pkg in ["pandas", "numpy", "matplotlib", "seaborn", "plotly"]):
            return "data_analysis"
        elif any(pkg in code for pkg in ["sklearn", "tensorflow", "torch", "keras"]):
            return "machine_learning"  
        elif any(pkg in code for pkg in ["flask", "django", "fastapi", "streamlit"]):
            return "web_app"
        elif any(pkg in code for pkg in ["scipy", "sympy", "networkx"]):
            return "scientific"
        else:
            return "general"

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

# Global instance
docker_agent = DockerAgent()
