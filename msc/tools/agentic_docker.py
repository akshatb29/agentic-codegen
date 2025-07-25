# msc/tools/agentic_docker.py
"""
Backward compatibility wrapper - redirects to new architecture
"""
from ..agents.docker_agent import docker_agent
from .docker_tools import docker_executor
import signal
import sys

class AgenticDockerManager:
    """Compatibility wrapper for the new Docker agent + tools architecture"""
    
    def __init__(self):
        self.agent = docker_agent
        self.tools = docker_executor  # Use tools instead of executor
        # Compatibility properties
        self.current_image = None
        self.session_id = self.tools.session_id
    
    def analyze_task_and_suggest_image(self, code: str, user_request: str = ""):
        """Compatibility method - redirects to docker agent"""
        return self.agent.analyze_code_and_generate_spec(code, user_request)
    
    def build_base_image_async(self, code: str, user_request: str = ""):
        """Compatibility method - uses agent + tools"""
        spec = self.agent.analyze_code_and_generate_spec(code, user_request)
        image_name = f"msc-session-{self.session_id}"
        self.current_image = self.tools.build_image_async(spec.dockerfile_content, image_name)
        return self.current_image
    
    def prepare_image_async(self, spec):
        """Compatibility method - prepare image from spec"""
        image_name = f"msc-session-{self.session_id}"
        dockerfile_content = spec.dockerfile_content if hasattr(spec, 'dockerfile_content') else spec.get('dockerfile_content', '')
        self.current_image = self.tools.build_image_async(dockerfile_content, image_name)
        return self.current_image
    
    def execute_file_unit_test(self, target_file: str, code: str):
        """Compatibility method - redirects to tools"""
        return self.tools.execute_file_unit_test(target_file, code, self.current_image)
    
    def execute_integration_test(self, main_file: str = "main.py"):
        """Compatibility method - redirects to tools"""
        return self.tools.execute_integration_test(main_file, self.current_image)
    
    def cleanup_session(self):
        """Compatibility method - redirects to tools"""
        return self.tools.cleanup_session()
    
    # Additional compatibility methods needed by execution.py
    def setup_signal_handler(self):
        """Compatibility method - basic signal handling"""
        def signal_handler(sig, frame):
            print("\n🛑 Received interrupt signal, cleaning up...")
            self.cleanup_session()
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)
    
    def check_existing_containers(self):
        """Compatibility method - return empty list for now"""
        return []
    
    def get_or_create_image(self, spec):
        """Compatibility method - build image from spec"""
        user_request = getattr(spec, 'reasoning', '') or getattr(spec, 'task_category', '')
        return self.tools.build_image_from_spec(
            dockerfile_content=spec.dockerfile_content,
            user_request=user_request,
            filename="main.py"
        )
    
    def create_project_container(self, image_name, code, user_request="", ask_before_plan=False):
        """Compatibility method - simplified container creation"""
        return f"container-{self.session_id}"
    
    def create_or_reuse_container(self, image_name, code, ask_before_plan=False):
        """Compatibility method - simplified container creation"""
        return f"container-{self.session_id}"
    
    def execute_in_container(self, container_name, code):
        """Compatibility method - execute code in container"""
        try:
            # Use the new architecture for execution
            result = self.tools.execute_file_unit_test("temp.py", code, self.current_image)
            return {
                "success": result.get("success", False),
                "output": result.get("stdout", ""),
                "error": result.get("stderr", ""),
                "execution_time": 0
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "execution_time": 0
            }
    
    def execute_in_container_with_entrypoint(self, container_name, code, entrypoint="python"):
        """Compatibility method - execute with entrypoint"""
        return self.execute_in_container(container_name, code)
    
    def cleanup_session_images(self):
        """Compatibility method - cleanup session"""
        return self.cleanup_session()

# Global instance for backward compatibility
docker_manager = AgenticDockerManager()

# Note: New code should use:
# from msc.agents.docker_agent import docker_agent
# from msc.tools.docker_tools import docker_executor
