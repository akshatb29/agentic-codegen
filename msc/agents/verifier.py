# msc/agents/verifier.py
from typing import Dict, Any
from rich.console import Console
from msc.state import AgentState
# Import run_code locally to avoid circular imports

console = Console()

def verifier_agent(state: AgentState) -> Dict[str, Any]:
    print("=" * 60)
    print("VERIFIER: Executing Code")
    print("=" * 60)
    
    # Import run_code locally to avoid circular import
    from msc.tools.execution import run_code
    from msc.tools.code_quality import code_checker
    
    code_to_verify = state.get("corrected_code") or state.get("generated_code")
    if not code_to_verify:
        return {"verifier_report": {"success": False, "stderr": "No code found to verify."}}
    
    # Run code quality check first
    quality_report = code_checker.check_code(code_to_verify, state.get("file_path", "script.py"))
    
    if quality_report["has_issues"]:
        console.print("[VERIFIER] Code quality issues detected:", style="yellow")
        code_checker.report_issues()
        
        # Use auto-fixed code if available
        if quality_report["clean_code"] != code_to_verify:
            console.print(" [VERIFIER] Using auto-fixed code", style="blue")
            code_to_verify = quality_report["clean_code"]
    
    # Check execution mode - convert to boolean
    use_docker_execution = state.get("execution_mode", "docker") == "docker"
    
    report = run_code(
        code_to_verify, 
        state["file_path"], 
        use_docker=use_docker_execution,  # Convert to boolean
        user_request=state.get("user_request", ""),  # Pass user request for better Docker image selection
        project_name=state.get("project_name", ""),  # Let it auto-detect
        language=state.get("language", ""),      # Let it auto-detect
        ask_reuse=False,   # Don't ask again - already chosen at start
        state=dict(state)  # Pass the full state to execution
    )
    
    # Add quality info to report
    report["quality_issues"] = quality_report["issues"]
    report["auto_fixed"] = quality_report["clean_code"] != (state.get("corrected_code") or state.get("generated_code"))
    
    return {"verifier_report": report}