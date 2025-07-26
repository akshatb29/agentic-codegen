# msc/tools/docker_tools.py
"""
Docker Tools Interface - Simple wrapper
"""
from .simple_project_docker import simple_docker_manager

class DockerTools:
    """Clean interface to Docker execution"""
    
    def get_or_create_project(self, user_request: str = "", suggested_project_name: str = "", language: str = ""):
        """Get or create project"""
        return simple_docker_manager.get_or_create_project(user_request, suggested_project_name, language)
    
    def execute_code(self, code: str, filename: str = "script.py", user_request: str = "", 
                     project_name: str = "", language: str = ""):
        """Execute code in Docker with automatic project setup"""
        # Auto-setup project if not done yet
        if not simple_docker_manager.current_project:
            simple_docker_manager.get_or_create_project(user_request, project_name, language)
        
        return simple_docker_manager.execute_code(code, filename, user_request)
    
    def copy_session_files(self, destination: str = None):
        """Copy session files to destination"""
        return simple_docker_manager.copy_session_files(destination)
    
    def cleanup(self):
        """Cleanup containers"""
        return simple_docker_manager.cleanup()

# Global instance
docker_tools = DockerTools()
