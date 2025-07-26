# msc/tools/project_docker_manager.py
"""
Simple Project Docker Manager: Generate language-specific Docker images
Keep base images small and specific per language
"""
import os
import docker
import time
import re
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from .filesystem import FilesystemTool
from rich.prompt import Prompt
from rich.console import Console

console = Console()

class ProjectDockerManager:
    """Simple Docker manager - generates specific images per language"""
    
    def __init__(self):
        self.docker_dir = Path(__file__).parent.parent.parent / "docker"
        self.projects_dir = self.docker_dir / "projects"
        self.session_id = int(time.time()) % 10000
        self.session_dir = self.projects_dir / f"session-{self.session_id}"
        self.current_project = None
        self.current_project_dir = None
        self.base_images = {}  # Cache for base language images
        
        # Ensure directories exist
        self.docker_dir.mkdir(exist_ok=True)
        self.projects_dir.mkdir(exist_ok=True)
        self.session_dir.mkdir(exist_ok=True)
        
        console.print(f"🆔 Session ID: {self.session_id}", style="dim")
        console.print(f"📁 Session directory: {self.session_dir}", style="dim")

    def get_or_create_project(self, user_request: str = "", suggested_project_name: str = "", language: str = "") -> Dict[str, Any]:
        """Get project name from user, create project directory in session"""
        
        # Use provided project name or generate from user request
        if suggested_project_name:
            suggested_name = suggested_project_name
        else:
            suggested_name = self._generate_project_name(user_request)
        
        console.print(f"\n📁 Project Setup", style="bold blue")
        if suggested_name:
            console.print(f"💡 Suggested name: {suggested_name}")
        if language:
            console.print(f"🔍 Target language: {language}")
        
        # Ask user for project name
        project_name = Prompt.ask(
            "Enter project name", 
            default=suggested_name if suggested_name else "my-project"
        )
        
        # Clean project name
        project_name = re.sub(r'[^a-z0-9-_]', '', project_name.lower().replace(' ', '-'))
        if not project_name:
            project_name = f"project-{self.session_id}"
        
        # Create project directory within session
        project_dir = self.session_dir / project_name
        project_dir.mkdir(exist_ok=True)
        
        self.current_project = project_name
        self.current_project_dir = project_dir
        
        console.print(f"✅ Project ready: {project_name}", style="bold green")
        console.print(f"📂 Directory: {project_dir}")
        
        return {
            "success": True,
            "project_name": project_name,
            "project_dir": str(project_dir),
            "session_dir": str(self.session_dir)
        }

    def _generate_project_name(self, user_request: str) -> str:
        """Generate project name from user request"""
        if not user_request:
            return ""
        
        # Extract meaningful words
        words = re.findall(r'\b[a-zA-Z]+\b', user_request.lower())
        
        # Filter out common words
        common_words = {
            'create', 'make', 'build', 'write', 'develop', 'code', 'program', 
            'application', 'app', 'system', 'project', 'script',
            'simple', 'basic', 'new', 'the', 'a', 'an', 'for', 'with', 'that'
        }
        
        meaningful_words = [w for w in words if len(w) > 2 and w not in common_words]
        
        if meaningful_words:
            return "-".join(meaningful_words[:3])
        
        return ""

    def execute_code(self, code: str, filename: str = "script.py", user_request: str = "") -> Dict[str, Any]:
        """Execute code with fallback from Docker to local execution"""
        if not self.current_project:
            return {"success": False, "error": "No project selected"}
        
        try:
            # Detect language
            language = self._detect_language(code, filename)
            console.print(f"🔍 Detected language: {language}", style="cyan")
            
            # Try Docker execution first, fallback to local if Docker has issues
            try:
                # Build or get base image for language
                base_image_name = f"msc-base-{language}"
                build_result = self._ensure_base_image(language, base_image_name)
                
                if build_result["success"]:
                    # Try docker exec approach
                    return self._execute_with_docker_exec(base_image_name, code, filename, language)
                else:
                    raise Exception("Docker image build failed")
                    
            except Exception as docker_error:
                console.print(f"⚠️ Docker execution failed: {docker_error}", style="yellow")
                return {"success": False, "error": str(docker_error)}
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": str(e)
            }

    def _ensure_base_image(self, language: str, base_image_name: str) -> Dict[str, Any]:
        """Ensure base language image exists"""
        try:
            client = docker.from_env()
            
            # Check if base language image exists
            try:
                client.images.get(base_image_name)
                console.print(f"✅ Using existing base {language} image: {base_image_name}", style="green")
                self.base_images[language] = base_image_name
                return {"success": True, "image_name": base_image_name}
            except docker.errors.ImageNotFound:
                pass
            
            # Build base language image
            console.print(f"🏗️ Building base {language} image: {base_image_name}")
            
            # Choose dockerfile
            dockerfile_name = f"Dockerfile.{language}"
            dockerfile_path = self.docker_dir / dockerfile_name
            
            if not dockerfile_path.exists():
                console.print(f"⚠️ No Dockerfile for {language}, using simple base", style="yellow")
                dockerfile_name = "Dockerfile.simple"
            
            # Build base image
            try:
                client.images.build(
                    path=str(self.docker_dir),
                    dockerfile=dockerfile_name,
                    tag=base_image_name,
                    rm=True,
                    forcerm=True
                )
                
                self.base_images[language] = base_image_name
                console.print(f"✅ Base {language} image built: {base_image_name}", style="green")
                return {"success": True, "image_name": base_image_name}
                
            except Exception as e:
                console.print(f"❌ Base image build failed: {e}", style="red")
                return {"success": False, "error": str(e)}
            
        except Exception as e:
            console.print(f"❌ Base image check failed: {e}", style="red")
            return {"success": False, "error": str(e)}

    def _execute_with_docker_exec(self, image_name: str, code: str, filename: str, language: str) -> Dict[str, Any]:
        """Execute code using docker exec with a running container"""
        try:
            client = docker.from_env()
            
            # Create a container name for this session/project
            container_name = f"msc-{self.current_project}-{self.session_id}"
            
            # Check if container already exists and is running
            container = None
            try:
                container = client.containers.get(container_name)
                if container.status != 'running':
                    container.start()
                    console.print(f"🔄 Started existing container: {container_name}", style="blue")
                else:
                    console.print(f"✅ Using running container: {container_name}", style="green")
            except docker.errors.NotFound:
                # Create new container
                console.print(f"🚀 Creating new container: {container_name}", style="blue")
                container = client.containers.run(
                    image_name,
                    command="sleep infinity",  # Keep container running
                    name=container_name,
                    working_dir="/projects",
                    detach=True,
                    remove=False  # Don't auto-remove so we can reuse
                )
                console.print(f"✅ Container created and running: {container_name}", style="green")
            
            # Create project directory inside container
            container_project_path = f"/projects/{self.current_project}"
            
            # Execute setup commands
            setup_commands = [
                f"mkdir -p {container_project_path}",
                f"echo '{code}' > {container_project_path}/{filename}"
            ]
            
            # Add dependency management
            deps_commands = self._get_container_dependency_commands(language, container_project_path)
            all_commands = setup_commands + deps_commands
            
            # Execute all setup commands as a single bash command
            combined_command = " && ".join(all_commands)
            console.print(f"🔧 Setting up container environment", style="blue")
            
            exec_result = container.exec_run(
                ["bash", "-c", combined_command],
                workdir="/projects"
            )
            
            if exec_result.exit_code != 0:
                console.print(f"❌ Setup failed", style="red")
                console.print(f"Error: {exec_result.output.decode()}", style="red")
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": exec_result.output.decode(),
                    "filename": filename,
                    "project": self.current_project,
                    "error": "Container setup failed"
                }
            
            # Execute the actual code
            exec_cmd = self._get_execution_command(filename, language)
            console.print(f"⚡ Executing: {' '.join(exec_cmd)} in container", style="blue")
            
            exec_result = container.exec_run(
                exec_cmd,
                workdir=container_project_path
            )
            
            stdout = exec_result.output.decode('utf-8') if exec_result.output else ""
            
            if exec_result.exit_code == 0:
                console.print(f"✅ Execution completed successfully", style="green")
                console.print(f"📄 Output:\n{stdout}")
                
                # Store code in session for potential copying later
                self._store_code_in_session(code, filename, language)
                
                return {
                    "success": True,
                    "stdout": stdout,
                    "stderr": "",
                    "filename": filename,
                    "project": self.current_project,
                    "language": language,
                    "container_name": container_name,
                    "container_path": container_project_path
                }
            else:
                console.print(f"❌ Execution failed with exit code: {exec_result.exit_code}", style="red")
                console.print(f"📄 Output:\n{stdout}")
                
                return {
                    "success": False,
                    "stdout": stdout,
                    "stderr": "",
                    "filename": filename,
                    "project": self.current_project,
                    "exit_code": exec_result.exit_code,
                    "error": f"Execution failed with exit code {exec_result.exit_code}"
                }
                
        except Exception as e:
            console.print(f"❌ Docker exec execution failed: {e}", style="red")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "filename": filename,
                "project": self.current_project,
                "error": str(e)
            }

    def _execute_in_container_filesystem(self, image_name: str, code: str, filename: str, language: str) -> Dict[str, Any]:
        """Execute code entirely within container filesystem using alternative approach"""
        try:
            client = docker.from_env()
            
            # For Python, let's try a simpler direct execution approach
            if language == "python":
                console.print(f"⚡ Executing Python code directly in container", style="blue")
                
                # Create a simple Python script that does everything
                exec_script = f'''
import os
import sys

# Create project directory
project_path = "/projects/{self.current_project}"
os.makedirs(project_path, exist_ok=True)
os.chdir(project_path)

# Write the code
with open("{filename}", "w") as f:
    f.write("""{code}""")

# Write requirements if needed
if not os.path.exists("requirements.txt"):
    with open("requirements.txt", "w") as f:
        f.write("# Python dependencies\\n")

# Execute the code
exec(open("{filename}").read())
'''
                
                # Run container with Python execution
                result = client.containers.run(
                    image_name,
                    command=["python3", "-c", exec_script],
                    working_dir="/projects",
                    remove=True,
                    stdout=True,
                    stderr=True
                )
                
                stdout = result.decode('utf-8') if result else ""
                
                console.print(f"✅ Execution completed in container", style="green")
                console.print(f"📄 Output:\\n{stdout}")
                
                # Store code in session for potential copying later
                self._store_code_in_session(code, filename, language)
                
                return {
                    "success": True,
                    "stdout": stdout,
                    "stderr": "",
                    "filename": filename,
                    "project": self.current_project,
                    "language": language,
                    "container_path": f"/projects/{self.current_project}"
                }
            
            # For other languages, fall back to the previous method but using sh
            else:
                # Create project directory path inside container
                container_project_path = f"/projects/{self.current_project}"
                
                # Prepare execution commands using sh instead of bash
                setup_commands = [
                    f"mkdir -p {container_project_path}",
                    f"cd {container_project_path}",
                    f"echo '{code}' > {filename}",
                ]
                
                # Add dependency management if needed
                deps_commands = self._get_container_dependency_commands(language, container_project_path)
                setup_commands.extend(deps_commands)
                
                # Add execution command
                exec_cmd = self._get_execution_command(filename, language)
                final_command = " && ".join(setup_commands + [" ".join(exec_cmd)])
                
                console.print(f"⚡ Executing in container: {self.current_project}", style="blue")
                console.print(f"📁 Container path: {container_project_path}", style="dim")
                
                # Run container with sh instead of bash
                result = client.containers.run(
                    image_name,
                    command=["sh", "-c", final_command],
                    working_dir="/projects",
                    remove=True,
                    stdout=True,
                    stderr=True
                )
                
                stdout = result.decode('utf-8') if result else ""
                
                console.print(f"✅ Execution completed in container", style="green")
                console.print(f"📄 Output:\\n{stdout}")
                
                # Store code in session for potential copying later
                self._store_code_in_session(code, filename, language)
                
                return {
                    "success": True,
                    "stdout": stdout,
                    "stderr": "",
                    "filename": filename,
                    "project": self.current_project,
                    "language": language,
                    "container_path": container_project_path
                }
            
        except docker.errors.ContainerError as e:
            stdout = e.stdout.decode('utf-8') if hasattr(e, 'stdout') and e.stdout else ""
            stderr = e.stderr.decode('utf-8') if hasattr(e, 'stderr') and e.stderr else ""
            
            console.print(f"❌ Container execution failed", style="red")
            console.print(f"📄 STDOUT:\\n{stdout}")
            console.print(f"💥 STDERR:\\n{stderr}")
            console.print(f"🔢 Exit code: {getattr(e, 'exit_status', 'unknown')}", style="red")
            
            return {
                "success": False,
                "stdout": stdout,
                "stderr": stderr,
                "filename": filename,
                "project": self.current_project,
                "exit_code": getattr(e, 'exit_status', -1),
                "error": str(e)
            }
        except Exception as e:
            console.print(f"❌ Unexpected error: {e}", style="red")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "filename": filename,
                "project": self.current_project,
                "error": str(e)
            }

    def _get_container_dependency_commands(self, language: str, project_path: str) -> List[str]:
        """Get commands to set up dependencies inside container"""
        commands = []
        
        if language == "python":
            # Create basic requirements.txt if needed
            commands.append(f"if [ ! -f {project_path}/requirements.txt ]; then echo '# Python dependencies' > {project_path}/requirements.txt; fi")
            commands.append(f"if [ -f {project_path}/requirements.txt ] && [ -s {project_path}/requirements.txt ]; then pip3 install -r {project_path}/requirements.txt; fi")
        
        elif language == "nodejs":
            # Create basic package.json if needed
            package_json = '{"name": "' + self.current_project + '", "version": "1.0.0", "dependencies": {}}'
            commands.append(f"if [ ! -f {project_path}/package.json ]; then echo '{package_json}' > {project_path}/package.json; fi")
            commands.append(f"if [ -f {project_path}/package.json ]; then cd {project_path} && npm install; fi")
        
        elif language == "go":
            # Create basic go.mod if needed
            commands.append(f"if [ ! -f {project_path}/go.mod ]; then echo 'module {self.current_project}\\n\\ngo 1.21' > {project_path}/go.mod; fi")
            commands.append(f"if [ -f {project_path}/go.mod ]; then cd {project_path} && go mod download; fi")
        
        return commands

    def _store_code_in_session(self, code: str, filename: str, language: str):
        """Store code in session directory for potential copying"""
        try:
            # Create local session copy for backup/copying purposes only
            project_session_dir = self.session_dir / self.current_project
            project_session_dir.mkdir(exist_ok=True)
            
            # Store the code file
            code_path = project_session_dir / filename
            FilesystemTool.write_file(str(code_path), code)
            
            # Store metadata
            metadata = {
                "language": language,
                "filename": filename,
                "project": self.current_project,
                "created_at": time.time()
            }
            
            import json
            metadata_path = project_session_dir / "metadata.json"
            FilesystemTool.write_file(str(metadata_path), json.dumps(metadata, indent=2))
            
        except Exception as e:
            console.print(f"⚠️ Failed to store session copy: {e}", style="yellow")

    def _detect_language(self, code: str, filename: str) -> str:
        """Detect programming language"""
        if filename.endswith(('.py', '.pyw')):
            return "python"
        elif filename.endswith(('.js', '.mjs', '.ts')):
            return "nodejs"
        elif filename.endswith('.go'):
            return "go"
        elif filename.endswith('.rs'):
            return "rust"
        elif filename.endswith(('.c', '.cpp', '.cc', '.cxx')):
            return "cpp"
        elif filename.endswith('.java'):
            return "java"
        elif filename.endswith('.php'):
            return "php"
        elif filename.endswith('.rb'):
            return "ruby"
        elif filename.endswith('.lua'):
            return "lua"
        elif filename.endswith('.pl'):
            return "perl"
        elif filename.endswith('.r', '.R'):
            return "r"
        else:
            # Try to detect from code content
            if any(keyword in code for keyword in ['import ', 'from ', 'def ', 'class ']):
                return "python"
            elif any(keyword in code for keyword in ['require(', 'const ', 'let ', 'console.log']):
                return "nodejs"
            elif 'package main' in code or 'func main()' in code:
                return "go"
            elif '#include' in code and any(keyword in code for keyword in ['int main', 'void main']):
                return "cpp"
            elif 'public class' in code and 'public static void main' in code:
                return "java"
            else:
                return "python"  # Default

    def _get_execution_command(self, filename: str, language: str) -> List[str]:
        """Get execution command for language"""
        if language == "python":
            return ["python3", filename]
        elif language == "nodejs":
            return ["node", filename]
        elif language == "go":
            return ["go", "run", filename]
        elif language == "ruby":
            return ["ruby", filename]
        elif language == "java":
            # For Java, we need to compile first, then run
            class_name = filename.replace('.java', '')
            return ["bash", "-c", f"javac {filename} && java {class_name}"]
        elif language == "cpp":
            # For C++, compile and run
            binary_name = filename.replace('.cpp', '').replace('.cc', '').replace('.cxx', '')
            return ["bash", "-c", f"g++ -o {binary_name} {filename} && ./{binary_name}"]
        elif language == "php":
            return ["php", filename]
        elif language == "perl":
            return ["perl", filename]
        elif language == "lua":
            return ["lua", filename]
        elif language == "r":
            return ["Rscript", filename]
        elif language == "rust":
            # For Rust, compile and run
            binary_name = filename.replace('.rs', '')
            return ["bash", "-c", f"rustc {filename} && ./{binary_name}"]
        else:
            return ["python3", filename]  # Default

    def list_projects(self) -> List[str]:
        """List all projects in current session"""
        if not self.session_dir.exists():
            return []
        return [d.name for d in self.session_dir.iterdir() if d.is_dir()]

    def generate_project_dockerfile(self) -> Dict[str, Any]:
        """Generate a complete Dockerfile for the project using LLM"""
        if not self.current_project:
            return {"success": False, "error": "No project selected"}
        
        try:
            # Get project info from session backup
            project_session_dir = self.session_dir / self.current_project
            if not project_session_dir.exists():
                return {"success": False, "error": "No project files found in session"}
            
            # Analyze project structure from session backup
            project_structure = self._analyze_session_project_structure(project_session_dir)
            
            # Get LLM to generate Dockerfile
            prompt = f"""Generate a production-ready Dockerfile for this project:

Project Structure:
{project_structure['structure']}

Requirements/Dependencies:
{project_structure['dependencies']}

Language: {project_structure['language']}
Entry Point: {project_structure['entry_point']}

Create a Dockerfile that:
1. Uses appropriate base image for {project_structure['language']}
2. Installs all dependencies
3. Copies project files
4. Sets proper entry point
5. Follows best practices (multi-stage build if needed, minimal layers, etc.)

Return only the Dockerfile content, no explanations."""

            try:
                from .llm_manager import get_llm
                llm = get_llm()
                dockerfile_content = llm.invoke(prompt).content
            except ImportError:
                # Fallback if LLM not available
                dockerfile_content = self._generate_basic_dockerfile(project_structure['language'])
                console.print("⚠️ LLM not available, using basic template", style="yellow")
            
            # Save generated Dockerfile to session
            dockerfile_path = project_session_dir / "Dockerfile.generated"
            FilesystemTool.write_file(str(dockerfile_path), dockerfile_content)
            
            console.print(f"✅ Generated Dockerfile: {dockerfile_path}", style="green")
            
            return {
                "success": True,
                "dockerfile_path": str(dockerfile_path),
                "content": dockerfile_content,
                "project_structure": project_structure
            }
            
        except Exception as e:
            console.print(f"❌ Dockerfile generation failed: {e}", style="red")
            return {"success": False, "error": str(e)}

    def _analyze_session_project_structure(self, project_dir: Path) -> Dict[str, Any]:
        """Analyze project structure from session backup"""
        structure_lines = []
        dependencies = []
        language = "unknown"
        entry_point = "unknown"
        
        # Walk through session project directory
        for root, dirs, files in os.walk(project_dir):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            level = root.replace(str(project_dir), '').count(os.sep)
            indent = ' ' * 2 * level
            structure_lines.append(f"{indent}{os.path.basename(root)}/")
            
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                if not file.startswith('.') and file != "metadata.json":
                    structure_lines.append(f"{subindent}{file}")
                    
                    # Detect language and entry point
                    if file.endswith('.py'):
                        language = "python"
                        if file == "main.py" or file == "app.py":
                            entry_point = file
                    elif file.endswith('.js'):
                        language = "nodejs"
                        if file == "index.js" or file == "app.js":
                            entry_point = file
                    elif file.endswith('.go'):
                        language = "go"
                        if file == "main.go":
                            entry_point = file
        
        # Read metadata if available
        metadata_path = project_dir / "metadata.json"
        if metadata_path.exists():
            try:
                import json
                metadata_content = FilesystemTool.read_file(str(metadata_path))
                metadata = json.loads(metadata_content)
                if metadata.get("language"):
                    language = metadata["language"]
            except:
                pass
        
        return {
            "structure": "\n".join(structure_lines),
            "dependencies": "Dependencies managed inside container",
            "language": language,
            "entry_point": entry_point
        }

    def _analyze_project_structure(self) -> Dict[str, Any]:
        """Analyze project structure for Dockerfile generation"""
        structure_lines = []
        dependencies = []
        language = "unknown"
        entry_point = "unknown"
        
        # Walk through project directory
        for root, dirs, files in os.walk(self.current_project_dir):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            level = root.replace(str(self.current_project_dir), '').count(os.sep)
            indent = ' ' * 2 * level
            structure_lines.append(f"{indent}{os.path.basename(root)}/")
            
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                if not file.startswith('.'):
                    structure_lines.append(f"{subindent}{file}")
                    
                    # Detect language and entry point
                    if file.endswith('.py'):
                        language = "python"
                        if file == "main.py" or file == "app.py":
                            entry_point = file
                    elif file.endswith('.js'):
                        language = "nodejs"
                        if file == "index.js" or file == "app.js":
                            entry_point = file
                    elif file.endswith('.go'):
                        language = "go"
                        if file == "main.go":
                            entry_point = file
                    
                    # Read dependency files
                    file_path = Path(root) / file
                    if file == "requirements.txt":
                        try:
                            content = FilesystemTool.read_file(str(file_path))
                            dependencies.append(f"Python requirements:\n{content}")
                        except:
                            pass
                    elif file == "package.json":
                        try:
                            content = FilesystemTool.read_file(str(file_path))
                            dependencies.append(f"Node.js package.json:\n{content}")
                        except:
                            pass
                    elif file == "go.mod":
                        try:
                            content = FilesystemTool.read_file(str(file_path))
                            dependencies.append(f"Go modules:\n{content}")
                        except:
                            pass
        
        return {
            "structure": "\n".join(structure_lines),
            "dependencies": "\n\n".join(dependencies) if dependencies else "No dependency files found",
            "language": language,
            "entry_point": entry_point
        }

    def copy_session_files(self, destination: str = None) -> Dict[str, Any]:
        """Copy all session files to user-specified location"""
        try:
            if not destination:
                from rich.prompt import Prompt
                # Use current working directory with project name as default
                current_dir = Path.cwd()
                if self.current_project:
                    default_dest = current_dir / f"{self.current_project}-session-{self.session_id}"
                else:
                    default_dest = current_dir / f"msc-session-{self.session_id}"
                
                destination = Prompt.ask(
                    "Enter destination path to copy session files",
                    default=str(default_dest)
                )
            
            dest_path = Path(destination)
            dest_path.mkdir(parents=True, exist_ok=True)
            
            # Copy all session files
            import shutil
            for item in self.session_dir.iterdir():
                if item.is_dir():
                    shutil.copytree(item, dest_path / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest_path)
            
            console.print(f"✅ Session files copied to: {dest_path}", style="green")
            
            return {
                "success": True,
                "destination": str(dest_path),
                "files_copied": [str(item) for item in self.session_dir.iterdir()]
            }
            
        except Exception as e:
            console.print(f"❌ Failed to copy session files: {e}", style="red")
            return {"success": False, "error": str(e)}

    def _generate_basic_dockerfile(self, language: str) -> str:
        """Generate a basic Dockerfile template as fallback"""
        if language == "python":
            return """FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

COPY . .

CMD ["python", "main.py"]
"""
        elif language == "nodejs":
            return """FROM node:18-slim

WORKDIR /app

COPY package*.json ./
RUN if [ -f package.json ]; then npm install; fi

COPY . .

CMD ["node", "index.js"]
"""
        elif language == "go":
            return """FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY go.* ./
RUN if [ -f go.mod ]; then go mod download; fi

COPY . .
RUN go build -o main .

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/main .

CMD ["./main"]
"""
        else:
            return """FROM ubuntu:22.04

WORKDIR /app
COPY . .

CMD ["/bin/bash"]
"""

    def cleanup_session(self, copy_files: bool = None) -> Dict[str, Any]:
        """Clean up session with option to copy files first"""
        try:
            if copy_files is None:
                from rich.prompt import Confirm
                copy_files = Confirm.ask("Copy session files before cleanup?")
            
            result = {"success": True, "copied": False, "cleaned": False}
            
            if copy_files:
                copy_result = self.copy_session_files()
                result["copied"] = copy_result["success"]
                if copy_result["success"]:
                    result["copy_destination"] = copy_result["destination"]
            
            # Clean up Docker resources
            self.cleanup()
            
            # Remove session directory
            import shutil
            if self.session_dir.exists():
                shutil.rmtree(self.session_dir)
                console.print(f"🗑️ Session directory removed: {self.session_dir}", style="yellow")
                result["cleaned"] = True
            
            console.print("✅ Session cleanup completed", style="green")
            return result
            
        except Exception as e:
            console.print(f"❌ Session cleanup failed: {e}", style="red")
            return {"success": False, "error": str(e)}

    def cleanup(self):
        """Clean up containers and images"""
        try:
            client = docker.from_env()
            
            # Remove containers with our prefix
            containers = client.containers.list(all=True)
            for container in containers:
                if container.image.tags and any('msc-' in tag for tag in container.image.tags):
                    container.remove(force=True)
                    console.print(f"🗑️ Removed container: {container.name}")
            
            console.print("✅ Cleanup completed")
            
        except Exception as e:
            console.print(f"⚠️ Cleanup error: {e}")

# Global instance
project_docker_manager = ProjectDockerManager()
