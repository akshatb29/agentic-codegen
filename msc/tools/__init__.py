# msc/tools/__init__.py
"""
Simple tools module - project-based Docker approach only
"""
from .filesystem import FilesystemTool
from .simple_project_docker import simple_docker_manager
from .execution import run_code, execute_with_context, run_code_safe
from .safe_testing import safe_tester
from .prompts import load_prompt
from .user_interaction import user_confirmation_tool, user_feedback_tool
from .context_analyzer import ContextAnalyzer
from .file_selector import FileSelector
from .llm_manager import llm_manager, get_llm, get_agent_config