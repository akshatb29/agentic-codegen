# msc/tools/simple_project_docker.py
"""
SIMPLE Project Docker Manager - Clean and minimal
Only what's actually needed
"""
import os
import docker
import time
import re
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
        container_name = f"msc-{language}-reusable"
        try:
            client = docker.from_env()
            existing_containers = [c for c in client.containers.list(all=True) if container_name in c.name]
            
            if existing_containers:
                existing = existing_containers[0]
                status = existing.status
                console.print(f"📦 Found existing {language} container: {container_name} (Status: {status})", style="yellow")
                
                if Confirm.ask(f"🔄 Reuse existing {language} container?", default=True):
                    return "reuse"
                else:
                    return "new"
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
            self.current_language = detected_language
        
        console.print(f"🔍 Language: {detected_language}")
        
        try:
            # Get or build image
            image_name = f"msc-base-{detected_language}"
            if not self._ensure_image(detected_language, image_name):
                return {"success": False, "error": f"Failed to build {detected_language} image"}
            
            # Execute in container (let missing imports be caught as errors)
            return self._run_in_container(image_name, code, filename, detected_language)
            
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

    def _run_in_container(self, image_name: str, code: str, filename: str, language: str) -> Dict[str, Any]:
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
            setup_cmd = f"mkdir -p {project_path} && echo '{code}' > {project_path}/{filename}"
            
            # Setup
            console.print(f"🔧 Setup command: {setup_cmd}", style="dim")
            exec_result = container.exec_run(["bash", "-c", setup_cmd])
            if exec_result.exit_code != 0:
                error_msg = exec_result.output.decode() if exec_result.output else "No error output"
                console.print(f"❌ Setup failed: {error_msg}", style="red")
                return {"success": False, "error": "Setup failed", "stderr": error_msg}
            
            # Execute
            exec_cmd = self._get_exec_command(filename, language)
            console.print(f"🚀 Exec command: {exec_cmd}", style="dim")
            exec_result = container.exec_run(exec_cmd, workdir=project_path)
            
            stdout = exec_result.output.decode('utf-8') if exec_result.output else ""
            stderr = ""
            
            if exec_result.exit_code == 0:
                console.print(f"✅ Success")
                console.print(f"📄 Output:\n{stdout}")
                
                # Store in session
                self._store_in_session(code, filename, language)
                
                return {
                    "success": True,
                    "stdout": stdout,
                    "stderr": stderr,
                    "filename": filename,
                    "project": self.current_project,
                    "language": language
                }
            else:
                console.print(f"❌ Failed (exit {exec_result.exit_code})")
                console.print(f"📄 Output:\n{stdout}")
                
                return {
                    "success": False,
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": exec_result.exit_code,
                    "error": f"Execution failed with exit code {exec_result.exit_code}"
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}

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
                destination = Prompt.ask("Copy to", default=str(default_dest))
            
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
