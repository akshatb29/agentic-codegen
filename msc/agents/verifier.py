# msc/agents/verifier.py
from typing import Dict, Any
from rich.console import Console
from msc.state import AgentState
# Import run_code locally to avoid circular imports

console = Console()

def verifier_agent(state: AgentState) -> Dict[str, Any]:
    print("=" * 60)
    print("⚡ VERIFIER: Executing Code")
    print("=" * 60)
    
    # Import run_code locally to avoid circular import
    from msc.tools.execution import run_code
    code_to_verify = state.get("corrected_code") or state.get("generated_code")
    if not code_to_verify:
        return {"verifier_report": {"success": False, "stderr": "No code found to verify."}}
        
    report = run_code(
        code_to_verify, 
        state["file_path"], 
        state["execution_mode"],
        state.get("user_request", "")  # Pass user request for better Docker image selection
    )
    return {"verifier_report": report}