# msc/tools/simple_project_docker.py
"""
SIMPLE Project Docker Manager - Clean and minimal with dynamic agent spawning
Only what's actually needed
"""
import os
import docker
import time
import re
import ast
from typing import Dict, Any, List
from pathlib import Path
from .filesystem import FilesystemTool
from rich.prompt import Prompt, Confirm
from rich.console import Console

console = Console()

class SimpleProjectDocker:
    """Minimal Docker manager - just execution, no bloat"""
    
    def __init__(self):
        """Initialize Docker manager with language-based container reuse"""
        self.projects_dir = Path(__file__).parent.parent.parent / "docker" / "projects"
        self.docker_dir = Path(__file__).parent.parent.parent / "docker"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        
        # Use language-based sessions for container reuse
        self.current_language = None
        self.current_project = None
        self.session_dir = None  # Will be set based on language
        
        console.print("🐳 Docker manager initialized (language-based container reuse)", style="dim")
        
    def _ask_user_preference(self, language: str) -> str:
        """Ask user whether to reuse existing container or create new one"""
        # Check if there's an existing container for this language
        try:
            client = docker.from_env()
            # Look for any container that starts with msc-{language}
            existing_containers = [c for c in client.containers.list(all=True) 
                                 if c.name.startswith(f"msc-{language}")]
            
            if existing_containers:
                existing = existing_containers[0]
                status = existing.status
                console.print(f"📦 Found existing {language} container: {existing.name} (Status: {status})", style="yellow")
                
                try:
                    if Confirm.ask(f"🔄 Reuse existing {language} container?", default=True):
                        return "reuse"
                    else:
                        return "new"
                except (EOFError, KeyboardInterrupt):
                    console.print("🔄 Using default: reuse existing container", style="dim")
                    return "reuse"
            else:
                console.print(f"📦 No existing {language} container found, will create new one", style="blue")
                return "new"
        except Exception as e:
            console.print(f"⚠️ Could not check containers: {e}", style="yellow")
            return "new"
        
    def _get_language_session(self, language: str, project_name: str = None, ask_user: bool = True) -> str:
        """Get or create language-based session directory"""
        # Normalize language name
        lang_map = {
            'python': 'python',
            'py': 'python', 
            'javascript': 'nodejs',
            'js': 'nodejs',
            'typescript': 'nodejs',
            'ts': 'nodejs',
            'go': 'go',
            'golang': 'go'
        }
        
        normalized_lang = lang_map.get(language.lower(), 'general')
        
        # Ask user preference if requested
        user_choice = "reuse"  # default
        if ask_user:
            user_choice = self._ask_user_preference(normalized_lang)
        
        if user_choice == "new":
            session_name = f"session-{normalized_lang}-{int(time.time()) % 10000}"
        else:
            session_name = f"session-{normalized_lang}"
        
        # Create session directory if needed
        self.session_dir = self.projects_dir / session_name
        self.session_dir.mkdir(exist_ok=True)
        
        return session_name

    def get_or_create_project(self, user_request: str = "", suggested_project_name: str = "", language: str = "", ask_reuse: bool = True) -> Dict[str, Any]:
        """Get project name and set language-based session"""
        # Set up language-based session first
        self.current_language = language or "general"
        session_name = self._get_language_session(self.current_language, suggested_project_name, ask_user=ask_reuse)
        
        if suggested_project_name:
            project_name = suggested_project_name
        else:
            # Extract simple name from request
            words = re.findall(r'\b[a-zA-Z]+\b', user_request.lower())
            meaningful = [w for w in words if len(w) > 2 and w not in {'create', 'make', 'build', 'the', 'a', 'an'}]
            project_name = "-".join(meaningful[:2]) if meaningful else "project"
        
        # Clean name
        project_name = re.sub(r'[^a-z0-9-_]', '', project_name.lower().replace(' ', '-'))
        if not project_name:
            project_name = f"project-{self.current_language}"
        
        self.current_project = project_name
        console.print(f"✅ Project: {project_name} (Language: {self.current_language}, Session: {session_name})", style="green")
        
        return {"success": True, "project_name": project_name, "session": session_name}

    def execute_code(self, code: str, filename: str = "script.py", user_request: str = "") -> Dict[str, Any]:
        """Execute code in Docker container"""
        if not self.current_project:
            return {"success": False, "error": "No project selected"}
        
        # Detect language and ask about container reuse when language differs from initial setup
        detected_language = self._detect_language(code, filename)
        
        if detected_language != self.current_language:
            console.print(f"🔄 Language updated: {self.current_language} → {detected_language}")
            # Ask about reuse for the detected language
            user_choice = self._ask_user_preference(detected_language)
            if user_choice == "new":
                # Force a new container with timestamp
                self.session_dir = self.projects_dir / f"session-{detected_language}-{int(time.time()) % 10000}"
                self.session_dir.mkdir(exist_ok=True)
            else:
                # Update session to use proper language-based session
                self.session_dir = self.projects_dir / f"session-{detected_language}"
                self.session_dir.mkdir(exist_ok=True)
            self.current_language = detected_language
        
        console.print(f"🔍 Language: {detected_language}")
        
        try:
            # Get or build image
            image_name = f"msc-base-{detected_language}"
            if not self._ensure_image(detected_language, image_name):
                return {"success": False, "error": f"Failed to build {detected_language} image"}
            
            # Execute in container (let missing imports be caught as errors)
            return self._run_in_container(image_name, code, filename, detected_language, user_request)
            
        except Exception as e:
            console.print(f"❌ Error: {e}", style="red")
            return {"success": False, "error": str(e)}

    def _detect_language(self, code: str, filename: str) -> str:
        """Simple language detection"""
        if filename.endswith('.py'):
            return "python"
        elif filename.endswith('.js'):
            return "nodejs"
        elif filename.endswith('.go'):
            return "go"
        elif filename.endswith('.rb'):
            return "ruby"
        else:
            # Guess from content
            if 'import ' in code or 'def ' in code:
                return "python"
            elif 'console.log' in code or 'const ' in code:
                return "nodejs"
            else:
                return "python"  # Default

    def _ensure_image(self, language: str, image_name: str) -> bool:
        """Ensure Docker image exists"""
        try:
            client = docker.from_env()
            
            # Check if exists
            try:
                client.images.get(image_name)
                console.print(f"✅ Using existing {language} image")
                return True
            except docker.errors.ImageNotFound:
                pass
            
            # Build it
            console.print(f"🏗️ Building {language} image...")
            dockerfile_name = f"Dockerfile.{language}"
            
            if not (self.docker_dir / dockerfile_name).exists():
                dockerfile_name = "Dockerfile.simple"
            
            client.images.build(
                path=str(self.docker_dir),
                dockerfile=dockerfile_name,
                tag=image_name,
                rm=True
            )
            
            console.print(f"✅ Built {language} image")
            return True
            
        except Exception as e:
            console.print(f"❌ Image build failed: {e}", style="red")
            return False

    def _run_in_container(self, image_name: str, code: str, filename: str, language: str, user_request: str = "") -> Dict[str, Any]:
        """Run code in container using exec"""
        try:
            client = docker.from_env()
            
            # Determine container name based on session type
            session_name = self.session_dir.name
            if session_name == f"session-{self.current_language}":
                # Reusable session (e.g., session-python)
                container_name = f"msc-{self.current_language}-reusable"
            else:
                # New session with timestamp (e.g., session-python-9077)
                timestamp = session_name.split('-')[-1]
                container_name = f"msc-{self.current_language}-{timestamp}"
            
            console.print(f"🐳 Using container: {container_name}", style="cyan")
            
            # Get or create container
            container = self._get_or_create_container(client, image_name, container_name)
            if not container:
                return {"success": False, "error": "Failed to create container"}
            
            # Setup and execute
            project_path = f"/projects/{self.current_project}"
            
            # Write code to a temporary file and copy it in
            import tempfile
            import base64
            
            # Encode the code to avoid shell escaping issues
            encoded_code = base64.b64encode(code.encode('utf-8')).decode('ascii')
            
            setup_cmd = f"mkdir -p {project_path} && echo '{encoded_code}' | base64 -d > {project_path}/{filename}"
            
            # Setup
            console.print(f"🔧 Setup: Creating {filename} in {project_path}", style="dim")
            exec_result = container.exec_run(["bash", "-c", setup_cmd])
            if exec_result.exit_code != 0:
                error_msg = exec_result.output.decode() if exec_result.output else "No error output"
                console.print(f"❌ Setup failed: {error_msg}", style="red")
                return {"success": False, "error": "Setup failed", "stderr": error_msg}
            
            # Execute using the specialized method with auto-fix capabilities
            return self._execute_in_container(container, code, filename, user_request)
                
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _spawn_execution_error_agent(self, error_context: str, code: str, filename: str, user_request: str):
        """Use consolidated dynamic problem solver for execution errors and auto-fix"""
        try:
            from msc.agents.dynamic_problem_solver import solve_execution_error
            
            console.print("🤖 Dynamic Problem Solver analyzing execution error...", style="yellow")
            
            # Create state for the solver
            mock_state = {
                "user_request": user_request,
                "code": code,
                "filename": filename,
                "error_context": error_context,
                "current_project": self.current_project,
                "current_language": self.current_language
            }
            
            # Use the consolidated solver
            result = solve_execution_error(error_context, mock_state)
            
            # Auto-install system packages if suggested
            system_packages = result.get('system_packages', [])
            if system_packages:
                console.print("🔧 Auto-installing system dependencies...", style="blue")
                try:
                    container = docker.from_env().containers.get(f"msc-{self.current_language}-reusable")
                    
                    # Update package lists first
                    console.print("  📦 Updating package lists...", style="dim")
                    update_result = container.exec_run("apt-get update", privileged=True)
                    
                    # Install system packages
                    for sys_pkg in system_packages[:5]:  # Limit to 5 packages
                        console.print(f"  🔧 Installing {sys_pkg}...", style="dim")
                        install_result = container.exec_run(
                            f"apt-get install -y {sys_pkg}", 
                            privileged=True
                        )
                        if install_result.exit_code == 0:
                            console.print(f"  ✅ Installed {sys_pkg}", style="green")
                        else:
                            console.print(f"  ⚠️ Failed to install {sys_pkg}", style="yellow")
                            
                    # Try to execute the code again after installing system dependencies
                    console.print("🔄 Retrying code execution after system dependency installation...", style="cyan")
                    retry_result = self._execute_in_container(container, code, filename, user_request)
                    
                    if retry_result.get('success'):
                        console.print("🎉 Code execution successful after auto-fix!", style="green")
                        return retry_result
                    else:
                        console.print("⚠️ Code still failing after system dependency installation", style="yellow")
                        
                except Exception as e:
                    console.print(f"  ⚠️ System package installation failed: {e}", style="yellow")
            
            # Install Python packages
            suggested_packages = result.get('suggested_packages', [])
            if suggested_packages:
                console.print("� Auto-installing Python packages...", style="green")
                
                try:
                    container = docker.from_env().containers.get(f"msc-{self.current_language}-reusable")
                    
                    for pkg in suggested_packages[:3]:  # Install top 3
                        console.print(f"  📦 Installing {pkg}...", style="dim")
                        install_success = self._install_package(container, pkg, self.current_language)
                        if install_success:
                            console.print(f"  ✅ Installed {pkg}", style="green")
                        else:
                            console.print(f"  ❌ Failed to install {pkg}", style="red")
                    
                    # Try to execute the code again after installing Python packages
                    console.print("🔄 Retrying code execution after Python package installation...", style="cyan")
                    retry_result = self._execute_in_container(container, code, filename, user_request)
                    
                    if retry_result.get('success'):
                        console.print("🎉 Code execution successful after auto-fix!", style="green")
                        return retry_result
                    else:
                        console.print("⚠️ Code still failing after package installation", style="yellow")
                        
                except Exception as e:
                    console.print(f"  ⚠️ Python package installation failed: {e}", style="yellow")
            
            alternatives = result.get('package_alternatives', {})
            if alternatives:
                console.print("🔄 Alternative packages available:", style="blue")
                for missing, alts in alternatives.items():
                    console.print(f"  {missing} → {alts[:2]}", style="dim")
            
        except Exception as e:
            console.print(f"⚠️ Dynamic problem solver failed: {e}", style="yellow")
        
        return None  # Indicate that auto-fix didn't work
    
    def _execute_in_container(self, container, code: str, filename: str, user_request: str = "") -> Dict[str, Any]:
        """Execute code in a container with proper validation and safety checks"""
        project_path = f"/projects/{self.current_project}"
        
        # STEP 1: Validate and clean the code first
        validated_code = self._validate_and_clean_code(code, filename)
        if not validated_code["valid"]:
            return {"success": False, "error": validated_code["error"], "stderr": validated_code["error"]}
        
        clean_code = validated_code["code"]
        console.print(f"✅ Code validation passed", style="green")
        
        # STEP 2: Check for bulk requirements and install them all at once
        requirements = self._extract_all_requirements(clean_code)
        if requirements:
            bulk_install_result = self._bulk_install_requirements(container, requirements)
            if not bulk_install_result["success"]:
                console.print(f"⚠️ Bulk install partially failed: {bulk_install_result['message']}", style="yellow")
        
        # STEP 3: Write validated code to container
        import base64
        encoded_code = base64.b64encode(clean_code.encode('utf-8')).decode('ascii')
        setup_cmd = f"mkdir -p {project_path} && echo '{encoded_code}' | base64 -d > {project_path}/{filename}"
        
        # Setup
        exec_result = container.exec_run(["bash", "-c", setup_cmd])
        if exec_result.exit_code != 0:
            error_msg = exec_result.output.decode() if exec_result.output else "No error output"
            return {"success": False, "error": "Setup failed", "stderr": error_msg}
        
        # STEP 4: Execute with safe command
        language_cmd = self._get_safe_exec_command(filename, self.current_language)
        exec_result = container.exec_run(language_cmd, workdir=project_path)
        
        stdout = exec_result.output.decode() if exec_result.output else ""
        stderr = ""
        
        if exec_result.exit_code == 0:
            console.print("✅ Success")
            console.print(f"📄 Output:\n{stdout}")
            
            # Store in session
            self._store_in_session(code, filename, self.current_language)
            
            return {
                "success": True,
                "stdout": stdout,
                "stderr": stderr,
                "filename": filename,
                "project": self.current_project,
                "language": self.current_language
            }
        else:
            console.print(f"❌ Failed (exit {exec_result.exit_code})")
            console.print(f"📄 Output:\n{stdout}")
            
            # Return the error - let the agent decide what to do
            return {
                "success": False,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exec_result.exit_code,
                "error": f"Execution failed with exit code {exec_result.exit_code}"
            }

    def _get_or_create_container(self, client, image_name: str, container_name: str):
        """Get existing container or create new one"""
        try:
            container = client.containers.get(container_name)
            if container.status != 'running':
                container.start()
            return container
        except docker.errors.NotFound:
            try:
                return client.containers.run(
                    image_name,
                    command="sleep infinity",
                    name=container_name,
                    working_dir="/projects",
                    detach=True,
                    remove=False
                )
            except Exception as e:
                console.print(f"❌ Container creation failed: {e}", style="red")
                return None

    def _get_exec_command(self, filename: str, language: str) -> List[str]:
        """Get execution command for language"""
        if language == "python":
            return ["python3", filename]
        elif language == "nodejs":
            return ["node", filename]
        elif language == "go":
            return ["go", "run", filename]
        elif language == "ruby":
            return ["ruby", filename]
        else:
            return ["python3", filename]
    
    def _get_safe_exec_command(self, filename: str, language: str) -> List[str]:
        """Get safe execution command with timeout and resource limits"""
        if language == "python":
            return ["timeout", "30", "python3", filename]
        elif language == "nodejs":
            return ["timeout", "30", "node", filename]
        elif language == "go":
            return ["timeout", "30", "go", "run", filename]
        elif language == "ruby":
            return ["timeout", "30", "ruby", filename]
        else:
            return ["timeout", "30", "python3", filename]
    
    def _validate_and_clean_code(self, code: str, filename: str) -> Dict[str, Any]:
        """Validate code format, remove fences, check for dangerous commands"""
        try:
            # STEP 1: Remove markdown code fences
            clean_code = self._remove_code_fences(code)
            
            # STEP 2: Check for dangerous commands
            dangerous_check = self._check_dangerous_commands(clean_code)
            if not dangerous_check["safe"]:
                return {"valid": False, "error": f"Dangerous command detected: {dangerous_check['reason']}", "code": ""}
            
            # STEP 3: Validate syntax for the language
            syntax_check = self._validate_syntax(clean_code, filename)
            if not syntax_check["valid"]:
                return {"valid": False, "error": f"Syntax error: {syntax_check['error']}", "code": ""}
            
            # STEP 4: Check for proper imports and structure
            structure_check = self._validate_code_structure(clean_code, filename)
            if not structure_check["valid"]:
                console.print(f"⚠️ Structure warning: {structure_check['warning']}", style="yellow")
            
            return {"valid": True, "code": clean_code, "warnings": structure_check.get("warnings", [])}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation failed: {str(e)}", "code": ""}
    
    def _remove_code_fences(self, code: str) -> str:
        """Remove markdown code fences and clean formatting"""
        import re
        
        # Remove code fences (```python, ```javascript, etc.)
        code = re.sub(r'^```[a-zA-Z]*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n```$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```$', '', code, flags=re.MULTILINE)
        
        # Remove leading/trailing whitespace
        code = code.strip()
        
        # Normalize line endings
        code = code.replace('\r\n', '\n').replace('\r', '\n')
        
        return code
    
    def _check_dangerous_commands(self, code: str) -> Dict[str, Any]:
        """Check for potentially dangerous commands"""
        dangerous_patterns = [
            r'rm\s+-rf',
            r'sudo\s+',
            r'chmod\s+777',
            r'wget\s+.*\s*\|\s*bash',
            r'curl\s+.*\s*\|\s*bash',
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'open\s*\(\s*["\']\/etc\/',
            r'os\.system\s*\(',
            r'subprocess\.',
            r'import\s+subprocess',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return {"safe": False, "reason": f"Matches dangerous pattern: {pattern}"}
        
        return {"safe": True}
    
    def _validate_syntax(self, code: str, filename: str) -> Dict[str, Any]:
        """Basic syntax validation based on file extension"""
        try:
            if filename.endswith('.py'):
                # Python syntax check
                import ast
                ast.parse(code)
                return {"valid": True}
            elif filename.endswith('.js'):
                # Basic JavaScript validation (check for balanced braces/parentheses)
                if self._check_balanced_delimiters(code):
                    return {"valid": True}
                else:
                    return {"valid": False, "error": "Unbalanced braces or parentheses"}
            else:
                # For other languages, do basic checks
                return {"valid": True}
                
        except SyntaxError as e:
            return {"valid": False, "error": f"Python syntax error: {str(e)}"}
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    def _check_balanced_delimiters(self, code: str) -> bool:
        """Check if braces, parentheses, and brackets are balanced"""
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}
        
        for char in code:
            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack:
                    return False
                if pairs[stack.pop()] != char:
                    return False
        
        return len(stack) == 0
    
    def _validate_code_structure(self, code: str, filename: str) -> Dict[str, Any]:
        """Validate code structure and provide warnings"""
        warnings = []
        
        if filename.endswith('.py'):
            # Check for common Python structure issues
            if 'if __name__ == "__main__"' not in code and 'def ' in code:
                warnings.append("Consider adding 'if __name__ == \"__main__\"' guard")
            
            if 'import' not in code and any(lib in code for lib in ['pandas', 'numpy', 'matplotlib']):
                warnings.append("Missing import statements for detected libraries")
        
        return {"valid": True, "warnings": warnings}
    
    def _extract_all_requirements(self, code: str) -> List[str]:
        """Extract ALL requirements from code for bulk installation"""
        requirements = set()
        
        # Common import to package mappings
        import_to_package = {
            'numpy': 'numpy',
            'np': 'numpy',
            'pandas': 'pandas', 
            'pd': 'pandas',
            'matplotlib': 'matplotlib',
            'plt': 'matplotlib',
            'seaborn': 'seaborn',
            'sns': 'seaborn',
            'sklearn': 'scikit-learn',
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'tensorflow': 'tensorflow-cpu',
            'tf': 'tensorflow-cpu',
            'torch': 'torch',
            'keras': 'keras',
            'requests': 'requests',
            'flask': 'flask',
            'django': 'django',
            'fastapi': 'fastapi',
            'scipy': 'scipy',
            'plotly': 'plotly',
            'streamlit': 'streamlit',
            'beautifulsoup4': 'beautifulsoup4',
            'bs4': 'beautifulsoup4'
        }
        
        # Extract imports
        import re
        
        # Pattern 1: import library
        for match in re.finditer(r'import\s+([a-zA-Z_][a-zA-Z0-9_]*)', code):
            lib = match.group(1)
            if lib in import_to_package:
                requirements.add(import_to_package[lib])
        
        # Pattern 2: from library import
        for match in re.finditer(r'from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import', code):
            lib = match.group(1)
            if lib in import_to_package:
                requirements.add(import_to_package[lib])
        
        # Pattern 3: import alias (import numpy as np)
        for match in re.finditer(r'import\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s+([a-zA-Z_][a-zA-Z0-9_]*)', code):
            lib = match.group(1)
            alias = match.group(2)
            if lib in import_to_package:
                requirements.add(import_to_package[lib])
            elif alias in import_to_package:
                requirements.add(import_to_package[alias])
        
        return list(requirements)
    
    def _bulk_install_requirements(self, container, requirements: List[str]) -> Dict[str, Any]:
        """Install all requirements at once for efficiency"""
        if not requirements:
            return {"success": True, "message": "No requirements to install"}
        
        console.print(f"📦 Installing {len(requirements)} packages in bulk: {', '.join(requirements)}", style="blue")
        
        try:
            # Create a single pip install command
            package_string = ' '.join(requirements)
            install_cmd = f"pip install {package_string}"
            
            console.print(f"🔧 Running: {install_cmd}", style="dim")
            result = container.exec_run(["bash", "-c", install_cmd], workdir="/")
            
            output = result.output.decode() if result.output else ""
            
            if result.exit_code == 0:
                console.print(f"✅ Bulk install successful: {len(requirements)} packages", style="green")
                return {"success": True, "message": f"Installed {len(requirements)} packages", "output": output}
            else:
                console.print(f"❌ Bulk install failed (exit {result.exit_code})", style="red")
                console.print(f"Output: {output}", style="dim")
                
                # Try individual installation as fallback
                return self._fallback_individual_install(container, requirements)
                
        except Exception as e:
            console.print(f"❌ Bulk install exception: {e}", style="red")
            return self._fallback_individual_install(container, requirements)
    
    def _fallback_individual_install(self, container, requirements: List[str]) -> Dict[str, Any]:
        """Fallback to individual package installation"""
        console.print("🔄 Falling back to individual package installation...", style="yellow")
        
        successful = []
        failed = []
        
        for package in requirements:
            try:
                console.print(f"  📦 Installing {package}...", style="dim")
                result = container.exec_run(["pip", "install", package], workdir="/")
                
                if result.exit_code == 0:
                    successful.append(package)
                    console.print(f"  ✅ {package}", style="green")
                else:
                    failed.append(package)
                    console.print(f"  ❌ {package}", style="red")
                    
            except Exception as e:
                failed.append(package)
                console.print(f"  ❌ {package}: {e}", style="red")
        
        return {
            "success": len(failed) == 0,
            "message": f"Individual install: {len(successful)} success, {len(failed)} failed",
            "successful": successful,
            "failed": failed
        }

    def _extract_missing_module(self, error_output: str) -> str:
        """Extract missing module name from error output"""
        import re
        match = re.search(r"ModuleNotFoundError: No module named '([^']+)'", error_output)
        return match.group(1) if match else None

    def _install_package(self, container, package_name: str, language: str) -> bool:
        """Install a package in the container"""
        try:
            if language == "python":
                # Map common package names
                package_map = {
                    'tensorflow': 'tensorflow-cpu',  # Use CPU version for faster install
                    'torch': 'torch torchvision',
                    'cv2': 'opencv-python',
                    'PIL': 'Pillow',
                    'sklearn': 'scikit-learn'
                }
                actual_package = package_map.get(package_name, package_name)
                
                install_cmd = f"pip install {actual_package}"
                console.print(f"🔧 Running: {install_cmd}", style="dim")
                result = container.exec_run(["bash", "-c", install_cmd])
                return result.exit_code == 0
            elif language == "nodejs":
                result = container.exec_run(["npm", "install", package_name])
                return result.exit_code == 0
            return False
        except Exception as e:
            console.print(f"❌ Package install failed: {e}", style="red")
            return False

    def _store_in_session(self, code: str, filename: str, language: str):
        """Store code in session directory"""
        try:
            project_dir = self.session_dir / self.current_project
            project_dir.mkdir(exist_ok=True)
            
            # Store code
            FilesystemTool.write_file(str(project_dir / filename), code)
            
            # Store metadata
            import json
            metadata = {
                "language": language,
                "filename": filename,
                "project": self.current_project,
                "created_at": time.time()
            }
            FilesystemTool.write_file(
                str(project_dir / "metadata.json"), 
                json.dumps(metadata, indent=2)
            )
        except Exception as e:
            console.print(f"⚠️ Session store failed: {e}", style="yellow")

    def copy_session_files(self, destination: str = None) -> Dict[str, Any]:
        """Copy session files"""
        try:
            if not destination:
                current_dir = Path.cwd()
                default_dest = current_dir / f"{self.current_project}-{self.current_language}"
                try:
                    destination = Prompt.ask("Copy to", default=str(default_dest))
                except (EOFError, KeyboardInterrupt):
                    destination = str(default_dest)
                    console.print(f"🔄 Using default destination: {destination}", style="dim")
            
            dest_path = Path(destination)
            dest_path.mkdir(parents=True, exist_ok=True)
            
            import shutil
            for item in self.session_dir.iterdir():
                if item.is_dir():
                    shutil.copytree(item, dest_path / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest_path)
            
            console.print(f"✅ Copied to: {dest_path}", style="green")
            return {"success": True, "destination": str(dest_path)}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def cleanup(self):
        """Clean up containers"""
        try:
            client = docker.from_env()
            containers = client.containers.list(all=True)
            for container in containers:
                if container.name.startswith("msc-"):
                    console.print(f"🗑️ Removed: {container.name}")
                    container.remove(force=True)
        except Exception as e:
            console.print(f"⚠️ Cleanup error: {e}", style="yellow")
    
    def reset_for_new_task(self):
        """Reset state for a new task to ensure fresh prompts"""
        console.print("🔄 Resetting Docker manager for new task", style="dim")
        self.current_project = None
        self.current_language = None
        self.session_dir = None

# Global instance
simple_docker_manager = SimpleProjectDocker()
