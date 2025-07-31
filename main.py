import os
import re
import atexit
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt, Confirm

from msc.state import AgentState
from msc.tools import FilesystemTool, FileSelector
from msc.tools.code_analyzer import CodeAnalyzer
from msc.tools.simple_project_docker import simple_docker_manager
from msc.enhanced_planning_graph import get_enhanced_planning_graph
from msc.enhanced_graph import enhanced_graph;

# Initialize Rich Console for better output
console = Console()

# Load environment variables from .env file
load_dotenv()

# Register cleanup function to run on exit
atexit.register(simple_docker_manager.cleanup)

def select_file_context(user_request: str = "") -> dict:
    """
    Enhanced file context selection with smart analysis and multiple selection modes.

    Args:
        user_request: The user's request to analyze for relevant files

    Returns:
        Dict of selected file contexts
    """
    all_files = FilesystemTool.read_directory_contents('.')
    return FileSelector.select_files_interactive(all_files, user_request)

def run_conversation_loop():
    """
    The main interactive loop for the chat-based CLI application.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        console.log("ERROR: GOOGLE_API_KEY not set in environment or .env file.")
        return

    console.print("-" * 80)
    console.print("AGENTIC AI DEVELOPMENT ASSISTANT")
    console.print("-" * 80)
    console.print("Available modes:")
    console.print("  - docker: Project-based Docker execution with isolated environments [DEFAULT]")
    console.print("  - local: Run code locally without containers")
    console.print("\nType your request to build or modify code. Type 'exit' or 'quit' to end.")

    # --- One-time setup ---
    console.print("\n--- Initial Configuration ---")
    enable_got = Confirm.ask("Enable Graph-of-Thoughts for advanced planning?", default=True)

    console.print("\n--- Execution Modes ---")
    console.print("  - docker: Project-based Docker execution (creates project directories)")
    console.print("  - local: Run code locally without containers")
    mode_input = Prompt.ask("Choose execution mode", choices=["docker", "local"], default="docker")
    execution_mode = mode_input

    if execution_mode == "docker":
        console.log("Project-based Docker mode selected")
    else:
        console.log("Local mode selected: Direct local execution")

    app = get_enhanced_planning_graph()
    # ----------------------

    while True:
        try:
            console.print("\n" + "-" * 60)
            console.print("AWAITING YOUR NEXT INSTRUCTION")
            console.print("-" * 60)
            user_request = Prompt.ask("You").strip()

            if user_request.lower() in ["exit", "quit"]:
                console.log("Session ended. Goodbye!")
                break

            # --- Per-task setup ---
            selected_context = select_file_context(user_request)
            structural_analysis = {}
            all_files = FilesystemTool.read_directory_contents('.')

            # --- NEW: Structural Code Analysis ---
            # Use regex to find potential function/method names in the request
            # e.g., "fix my_function", "the error in MyClass.my_method"
            potential_func_names = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_.]+)\b', user_request)

            if potential_func_names:
                console.log("[bold cyan]Found potential function/class names, running structural analysis...[/bold cyan]")
                code_analyzer = CodeAnalyzer(all_files)
                
                # Analyze the last found name as it's often the most relevant
                # Handles both 'function' and 'Class.method' patterns
                func_to_analyze = potential_func_names[-1]
                analysis_result = code_analyzer.get_function_analysis(func_to_analyze)
                
                if analysis_result['defining_files'] or analysis_result['calling_files']:
                    console.log(f"✅ [bold green]Structural analysis for '{func_to_analyze}':[/bold green]")
                    console.log(f"   - Defining Files: {analysis_result['defining_files']}")
                    console.log(f"   - Calling Files: {analysis_result['calling_files']}")
                    structural_analysis[func_to_analyze] = analysis_result

                    # Automatically add these files to the context if they aren't already there
                    all_relevant_files = analysis_result['defining_files'] + analysis_result['calling_files']
                    for file_path in all_relevant_files:
                        if file_path not in selected_context:
                            console.log(f"   -> Adding '[bold yellow]{file_path}[/bold yellow]' to context from analysis.")
                            selected_context[file_path] = all_files.get(file_path, "")
            # --- End of Structural Analysis ---


            initial_state: AgentState = {
                "user_request": user_request,
                "enable_got_planning": enable_got,
                "execution_mode": execution_mode,
                "existing_file_context": selected_context,
                "structural_analysis_context": structural_analysis, # New context added
                "use_docker_execution": execution_mode == "docker", #type:ignore
                # --- Reset other state variables for the new task ---
                "enable_symbolic_reasoning": False,
                "enable_pseudocode_iterations": True,
                "llm_model": "gemini-1.5-flash-latest",
                "messages": [],
                "thoughts_history": [],
                "correction_attempts": 0,
                "plan_approved": None,
                "gen_strategy_approved": None,
            }
            # ------------------------

            console.print("\n" + "-" * 20 + " STARTING NEW TASK " + "-" * 20)

            # Reset Docker manager state for new task to ensure user prompts
            simple_docker_manager.reset_for_new_task()

            for event in app.stream(initial_state, {"recursion_limit": 150}): #type:ignore
                for node, output in event.items():
                    console.log(f"--- Finished Node: {node} ---")

            console.print("\n" + "-" * 25 + " TASK COMPLETE " + "-" * 25)

            # Ask user if they want to copy session files
            if simple_docker_manager.current_project and Confirm.ask("Copy project files to local directory?", default=True):
                simple_docker_manager.copy_session_files()

        except KeyboardInterrupt:
            console.log("\nSession interrupted by user. Goodbye!")
            simple_docker_manager.cleanup()  # Cleanup on interrupt
            break
        except Exception as e:
            console.log(f"[bold red]An unexpected error occurred: {e}[/bold red]")
            console.print_exception(show_locals=True)
            console.log("Restarting loop...")

    # Clean up at end of normal session
    simple_docker_manager.cleanup()

if __name__ == "__main__":
    run_conversation_loop()

