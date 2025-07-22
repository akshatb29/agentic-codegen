# main.py
import os
import atexit
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt, Confirm

from msc.state import AgentState
from msc.tools import FilesystemTool, FileSelector
from msc.tools.agentic_docker import docker_manager
from msc.graph import build_graph

# Load environment variables from .env file
load_dotenv()

# Setup signal handler for graceful Docker cleanup
docker_manager.setup_signal_handler()

# Register cleanup function to run on exit
atexit.register(docker_manager.cleanup_session_images)

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
    print("Type your request to build or modify code. Type 'exit' or 'quit' to end.")
    
    # --- One-time setup ---
    print("\n⚙️  Initial Configuration:")
    enable_got = input("Enable Graph-of-Thoughts for advanced planning? [Y/n]: ").strip().lower() not in ['n', 'no']
    execution_mode = input("Choose execution mode (local/docker) [docker]: ").strip() or "docker"
    app = build_graph()
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
            for event in app.stream(initial_state, {"recursion_limit": 150}):
                for node, output in event.items():
                    print(f"--- ✅ Finished Node: {node} ---")
            
            print("\n" + "✅" * 25 + " TASK COMPLETE " + "✅" * 25)

        except KeyboardInterrupt:
            print("\n👋 Session interrupted by user. Goodbye!")
            docker_manager.cleanup_session_images()  # Clean up on interrupt
            break
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
            print("🔄 Restarting loop...")

    # Clean up at end of normal session
    docker_manager.cleanup_session_images()

if __name__ == "__main__":
    run_conversation_loop()