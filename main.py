# main.py
import os
import atexit
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt, Confirm

from msc.state import AgentState
from msc.tools import FilesystemTool, FileSelector
# Use new simple project-based Docker architecture
from msc.tools.simple_project_docker import simple_docker_manager
from msc.enhanced_planning_graph import get_enhanced_planning_graph

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
        print("❌ ERROR: GOOGLE_API_KEY not set in environment or .env file.")
        return

    print("=" * 80)
    print("🚀 AGENTIC AI DEVELOPMENT ASSISTANT 🚀")
    print("=" * 80)
    print("Available modes:")
    print("  • docker: 🧠 Project-based Docker execution with isolated environments [DEFAULT]")
    print("  • local: Run code locally without containers")
    print("\nType your request to build or modify code. Type 'exit' or 'quit' to end.")
    
    # --- One-time setup ---
    print("\n⚙️  Initial Configuration:")
    enable_got = input("Enable Graph-of-Thoughts for advanced planning? [Y/n]: ").strip().lower() not in ['n', 'no']
    
    # Simplified execution mode selection - project-based docker or local
    print("\nExecution modes:")
    print("  • docker: Project-based Docker execution (creates project directories)")
    print("  • local: Run code locally without containers")
    mode_input = input("Choose execution mode [docker,l]: ").strip().lower()
    execution_mode = "local" if mode_input == "l" else "docker"
    
    # Show selected mode
    if execution_mode == "docker":
        print("✅ Project-based Docker mode selected")
        print("   • Each project gets isolated directory")
        print("   • Automatic requirements.txt management")
        print("   • Lightweight Ubuntu container execution")
    else:
        print("✅ Local mode selected: Direct local execution")
    
    app = get_enhanced_planning_graph()
    # ----------------------

    while True:
        try:
            print("\n" + "=" * 60)
            print("🤖 AWAITING YOUR NEXT INSTRUCTION")
            print("=" * 60)
            user_request = input("You: ").strip()

            if user_request.lower() in ["exit", "quit"]:
                print("👋 Session ended. Goodbye!")
                break
            
            # --- Per-task setup ---
            selected_context = select_file_context(user_request)

            initial_state: AgentState = {
                "user_request": user_request,
                "enable_got_planning": enable_got,
                "execution_mode": execution_mode,
                "existing_file_context": selected_context,
                "use_docker_execution": execution_mode == "docker",  # Enable Docker workflow with CMD modifier
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

            print("\n" + "🤖" * 20 + " STARTING NEW TASK " + "🤖" * 20)
            
            # Reset Docker manager state for new task to ensure user prompts
            simple_docker_manager.reset_for_new_task()
            
            for event in app.stream(initial_state, {"recursion_limit": 150}):
                for node, output in event.items():
                    print(f"--- ✅ Finished Node: {node} ---")
            
            print("\n" + "✅" * 25 + " TASK COMPLETE " + "✅" * 25)
            
            # Ask user if they want to copy session files
            from rich.prompt import Confirm
            if simple_docker_manager.current_project and Confirm.ask("💾 Copy project files to local directory?", default=True):
                simple_docker_manager.copy_session_files()

        except KeyboardInterrupt:
            print("\n👋 Session interrupted by user. Goodbye!")
            simple_docker_manager.cleanup()  # Cleanup on interrupt
            break
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
            print("🔄 Restarting loop...")

    # Clean up at end of normal session
    simple_docker_manager.cleanup()  # Cleanup before exit

if __name__ == "__main__":
    run_conversation_loop()