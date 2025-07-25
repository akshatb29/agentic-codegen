# msc/tools/docker_tools.py
"""
Docker Tools: Pure execution tools for containerized code testing
"""
import os
import docker
import threading
from typing import Dict, Any, Optional
from pathlib import Path

from .filesystem import FilesystemTool

class DockerExecutor:
    """Simple Docker executor: handles image building and code execution only"""
    
    def __init__(self):
        self.docker_dir = Path(__file__).parent.parent.parent / "docker"
        self.current_image = None  # Single image for current run
        self.session_id = int(__import__('time').time()) % 10000
        self.project_name = None  # Will be set based on user request
        self.built_images = []  # Track images built in this session

    def generate_meaningful_name(self, user_request: str = "", filename: str = "") -> str:
        """Generate meaningful Docker image name based on project context"""
        import re
        
        # Extract project type from user request
        project_type = "general"
        if any(word in user_request.lower() for word in ["web", "flask", "django", "fastapi", "server"]):
            project_type = "web"
        elif any(word in user_request.lower() for word in ["data", "analysis", "pandas", "numpy", "plot"]):
            project_type = "data-analysis"
        elif any(word in user_request.lower() for word in ["ml", "machine learning", "ai", "tensorflow", "pytorch"]):
            project_type = "ml"
        elif any(word in user_request.lower() for word in ["gui", "tkinter", "qt", "interface"]):
            project_type = "gui"
        elif any(word in user_request.lower() for word in ["game", "pygame", "graphics"]):
            project_type = "game"
        elif any(word in user_request.lower() for word in ["calculator", "math", "compute"]):
            project_type = "calculator"
        elif any(word in user_request.lower() for word in ["script", "automation", "tool"]):
            project_type = "script"
        
        # Extract key words from user request for naming
        words = re.findall(r'\b[a-zA-Z]+\b', user_request.lower())
        key_words = [w for w in words if len(w) > 3 and w not in ['create', 'make', 'build', 'write', 'python', 'code', 'application', 'program']]
        
        # Create meaningful name
        if key_words:
            project_name = "-".join(key_words[:2])  # Use first 2 meaningful words
        else:
            project_name = filename.replace('.py', '') if filename else 'project'
        
        # Clean the name
        project_name = re.sub(r'[^a-z0-9-]', '', project_name)
        
        # Set project name for session
        self.project_name = project_name
        
        # Format: msc-{type}-{name}-{session}
        return f"msc-{project_type}-{project_name}-{self.session_id}"

    def build_image_from_spec(self, dockerfile_content: str, image_name: str = None, user_request: str = "", filename: str = "") -> str:
        """Build Docker image from Dockerfile content"""
        try:
            client = docker.from_env()
            client.ping()
        except docker.errors.DockerException:
            print("❌ Docker daemon not available")
            return None
        
        try:
            # Generate meaningful name if not provided
            if not image_name:
                image_name = self.generate_meaningful_name(user_request, filename)
            
            # Create Dockerfile
            dockerfile_path = self.docker_dir / f"Dockerfile.{image_name}"
            FilesystemTool.write_file(str(dockerfile_path), dockerfile_content)
            
            print(f"🔨 Building Docker image: {image_name}")
            print(f"📝 Project type: {image_name.split('-')[1] if len(image_name.split('-')) > 1 else 'general'}")
            
            # Build the image
            client.images.build(
                path=str(self.docker_dir),
                dockerfile=dockerfile_path.name,
                tag=image_name,
                rm=True,
                forcerm=True
            )
            
            print(f"✅ Successfully built image: {image_name}")
            self.current_image = image_name
            self.built_images.append(image_name)  # Track built images
            return image_name
            
        except Exception as e:
            print(f"❌ Build failed: {e}")
            return None
            self.built_images.append(image_name)  # Track built images
        
        try:
            # Generate meaningful name if not provided
            if not image_name:
                image_name = self.generate_meaningful_name(user_request, filename)
            
            # Create Dockerfile
            dockerfile_path = self.docker_dir / f"Dockerfile.{image_name}"
            FilesystemTool.write_file(str(dockerfile_path), dockerfile_content)
            
            print(f"🔨 Building Docker image: {image_name}")
            print(f"📝 Project type: {image_name.split('-')[1] if len(image_name.split('-')) > 1 else 'general'}")
            
            # Build the image
            client.images.build(
                path=str(self.docker_dir),
                dockerfile=dockerfile_path.name,
                tag=image_name,
                rm=True,
                forcerm=True
            )
            
            print(f"✅ Successfully built image: {image_name}")
            self.current_image = image_name
            self.built_images.append(image_name)  # Track built images
            return image_name
            
        except Exception as e:
            print(f"❌ Build failed: {e}")
            return None

    def build_image_async(self, dockerfile_content: str, image_name: str) -> str:
        """Build Docker image asynchronously (non-blocking)"""
        print(f"🚀 Building image in parallel: {image_name}")
        
        def build_worker():
            try:
                self.build_image_from_spec(dockerfile_content, image_name)
                print(f"✅ Async build completed: {image_name}")
            except Exception as e:
                print(f"❌ Async build failed for {image_name}: {e}")
        
        # Start build in background
        build_thread = threading.Thread(target=build_worker, daemon=True)
        build_thread.start()
        
        self.current_image = image_name  # Set immediately for reference
        return image_name
    
    def rebuild_with_entrypoint(self, base_image: str, target_file: str) -> str:
        """Rebuild image with specific file as entrypoint"""
        file_specific_image = f"{base_image}-{target_file.replace('.py', '').replace('/', '_')}"
        
        try:
            client = docker.from_env()
            
            # Read base dockerfile
            base_dockerfile_path = self.docker_dir / f"Dockerfile.{base_image}"
            if not base_dockerfile_path.exists():
                print(f"⚠️ Base dockerfile not found: {base_dockerfile_path}")
                return base_image  # Return original if base not found
            
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
            return base_image  # Fallback to base image
    
    def execute_file_unit_test(self, target_file: str, code: str, base_image: str = None) -> Dict[str, Any]:
        """Execute and unit test a specific file"""
        if not base_image:
            base_image = self.current_image
            
        if not base_image:
            return {
                "success": False,
                "stdout": "",
                "stderr": "No base image available",
                "target_file": target_file,
                "test_type": "unit_test"
            }
        
        try:
            # Write code to file
            script_path = self.docker_dir / target_file
            FilesystemTool.write_file(str(script_path), code)
            
            # Rebuild image with this file as entrypoint
            file_image = self.rebuild_with_entrypoint(base_image, target_file)
            
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
    
    def execute_integration_test(self, main_file: str = "main.py", base_image: str = None) -> Dict[str, Any]:
        """Execute full integration test with main file as entrypoint"""
        if not base_image:
            base_image = self.current_image
            
        if not base_image:
            return {
                "success": False,
                "stdout": "",
                "stderr": "No base image available",
                "main_file": main_file,
                "test_type": "integration_test"
            }
        
        try:
            # Rebuild with main file as entrypoint
            main_image = self.rebuild_with_entrypoint(base_image, main_file)
            
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
    
    def offer_to_save_session(self):
        """Offer user choice to save current session as a permanent image"""
        if not self.built_images:
            print("ℹ️ No images built in this session to save.")
            return
        
        try:
            print("\n" + "="*60)
            print("💾 SESSION SAVE OPTIONS")
            print("="*60)
            print(f"📦 Images built in this session:")
            for idx, image in enumerate(self.built_images, 1):
                print(f"   {idx}. {image}")
            
            choice = input(f"\n🤔 Save any images permanently? (y/N): ").strip().lower()
            if choice in ['y', 'yes']:
                self._save_images_permanently()
            else:
                print("🗑️ Images will be cleaned up normally.")
                
        except Exception as e:
            print(f"⚠️ Error in save session: {e}")
    
    def _save_images_permanently(self):
        """Save selected images with meaningful names"""
        try:
            client = docker.from_env()
            
            for image_name in self.built_images:
                save_choice = input(f"\n💾 Save '{image_name}'? (y/N): ").strip().lower()
                if save_choice in ['y', 'yes']:
                    
                    # Suggest a permanent name
                    project_type = image_name.split('-')[1] if len(image_name.split('-')) > 1 else 'general'
                    suggested_name = f"my-{project_type}-project"
                    if self.project_name:
                        suggested_name = f"my-{self.project_name}"
                    
                    permanent_name = input(f"🏷️ Permanent name [{suggested_name}]: ").strip() or suggested_name
                    
                    try:
                        # Get the image and tag it with permanent name
                        image = client.images.get(image_name)
                        image.tag(permanent_name, 'latest')
                        
                        # Also create a timestamped version
                        import datetime
                        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
                        image.tag(permanent_name, timestamp)
                        
                        print(f"✅ Saved as: {permanent_name}:latest and {permanent_name}:{timestamp}")
                        print(f"🔄 Run with: docker run -it {permanent_name}:latest")
                        
                    except Exception as e:
                        print(f"❌ Failed to save {image_name}: {e}")
                        
        except Exception as e:
            print(f"❌ Error saving images: {e}")

    def cleanup_session(self):
        """Enhanced cleanup with save option"""
        # First offer to save before cleanup
        self.offer_to_save_session()
        
        try:
            client = docker.from_env()
            
            print(f"\n🧹 Cleaning up session {self.session_id}...")
            
            # Remove all session images (except saved ones)
            for image in client.images.list():
                for tag in image.tags:
                    if f"msc-" in tag and f"-{self.session_id}" in tag:
                        try:
                            # Check if this image was saved with a permanent name
                            saved_tags = [t for t in image.tags if not t.startswith('msc-')]
                            if saved_tags:
                                print(f"⏭️ Keeping {tag} (saved as {saved_tags[0]})")
                            else:
                                client.images.remove(tag, force=True)
                                print(f"🗑️ Removed: {tag}")
                        except Exception as e:
                            print(f"⚠️ Could not remove {tag}: {e}")
            
            # Clean up dockerfiles
            for dockerfile in self.docker_dir.glob(f"Dockerfile.msc-*-{self.session_id}*"):
                try:
                    dockerfile.unlink()
                    print(f"🗑️ Removed: {dockerfile.name}")
                except Exception as e:
                    print(f"⚠️ Could not remove {dockerfile.name}: {e}")
                    
            print(f"✅ Session {self.session_id} cleaned up")
            
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")

# Global instance
docker_executor = DockerExecutor()
