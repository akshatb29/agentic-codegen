# msc/enhanced_graph.py
"""
Enhanced Graph with Unified Docker Management and Parallel Test Generation
"""
from langgraph.graph import StateGraph, END
from msc.state import AgentState
from msc.agents import (
    planner_agent, symbolic_reasoner_agent, pseudocode_refiner_agent,
    nl_to_code_agent, pseudocode_to_code_agent, symbolic_to_code_agent,
    verifier_agent, critique_agent, corrector_agent
)
from msc.agents.requirements_installer import requirements_installer_agent
from msc.tools.execution import run_code

# Initialize execution functions
def run_code_agent(state: AgentState) -> AgentState:
    """Execute generated code using run_code with automatic fallback"""
    code = state.code
    if not code:
        return state
    
    filename = state.filename or "script.py"
    user_request = getattr(state, 'user_request', '')
    project_name = getattr(state, 'project_name', '')
    language = getattr(state, 'language', '')
    
    # Use run_code with Docker-first, local fallback
    result = run_code(
        code=code,
        filename=filename, 
        use_docker=True,
        user_request=user_request,
        project_name=project_name,
        language=language
    )
    
    # Set the verifier_report properly for graph routing
    state.execution_result = result
    state.verifier_report = {"success": result.get("success", False)}
    return state

def should_continue(state: AgentState) -> str:
    """Determine if processing should continue to next file"""
    if not state.get("file_plan_iterator"):
        # All files processed, check if batch execution is needed
        if state.get("parallel_execution_active"):
            return "batch_test_generator"
        return END
    return "planner"  # Use main planner instead of agentic_planner

def route_strategy(state: AgentState) -> str:
    """Route to appropriate code generation strategy"""
    if not state.get("gen_strategy_approved"):
        return "planner"  # Route back to main planner
    strategy = state.get("chosen_gen_strategy")
    return {"Symbolic": "symbolic_reasoner", "Pseudocode": "pseudocode_refiner"}.get(strategy, "nl_to_code")

def check_verification(state: AgentState) -> str:
    """Check verification results and route accordingly"""
    verifier_report = state.get("verifier_report", {})
    critique_details = state.get("critique_feedback_details", {})
    
    # Check if both verification and critique passed
    verification_passed = verifier_report.get("success", False)
    critique_passed = critique_details.get("is_correct_and_runnable", False)
    
    if verification_passed and critique_passed:
        # Choose execution mode based on state
        execution_mode = state.get("execution_mode", "docker")
        
        if execution_mode == "unified_docker":
            return "unified_docker_workflow"
        elif execution_mode == "batch":
            return "batch_execution"
        else:
            return "run_code_agent"  # Default execution with fallback
    
    # Check correction attempts
    if state.get("correction_attempts", 0) >= 2:
        return "finish_file"
    
    return "corrector"

def route_docker_execution(state: AgentState) -> str:
    """Route from Docker execution based on results"""
    verifier_report = state.get("verifier_report", {})
    
    if verifier_report.get("success", False):
        return "finish_file"
    else:
        # If execution failed, route back to corrector
        return "corrector"

def route_batch_execution(state: AgentState) -> str:
    """Route after batch execution"""
    batch_results = state.get("batch_execution_results", {})
    
    if batch_results.get("success", False):
        return "batch_test_generator"
    else:
        return "finish_file"

def finish_file(state: AgentState) -> dict:
    """Finish processing current file and move to next"""
    current_file = state.get("current_file_name", "unknown")
    print(f"✅ SUCCESS: Finished processing {current_file}")
    
    # Remove current file from iterator
    if state.get("file_plan_iterator"):
        state["file_plan_iterator"].pop(0)
    
    # Reset file-specific state
    return {
        "gen_strategy_approved": False,
        "corrected_code": None,
        "generated_code": None,
        "current_file_name": None,
        "current_task_description": None
    }

def batch_test_generator(state: AgentState) -> dict:
    """Generate and execute tests for all completed files"""
    print("🧪 Batch Test Generator: Creating comprehensive test suites")
    
    # This node handles test generation for all files at once
    # The actual test generation is handled by the unified_docker_manager
    
    return {
        "files_completed": True,
        "ready_for_testing": True,
        "test_generation_results": state.get("test_generation_results", {})
    }

def build_enhanced_graph() -> StateGraph:
    """Build enhanced graph with main planner and requirements installer"""
    workflow = StateGraph(AgentState)
    
    # Core agents - removed agentic_planner, using main planner only
    workflow.add_node("planner", planner_agent)  # Main planner with enhanced capabilities
    workflow.add_node("requirements_installer", requirements_installer_agent)  # Add requirements installer
    workflow.add_node("symbolic_reasoner", symbolic_reasoner_agent)
    workflow.add_node("pseudocode_refiner", pseudocode_refiner_agent)
    workflow.add_node("nl_to_code", nl_to_code_agent)
    workflow.add_node("pseudocode_to_code", pseudocode_to_code_agent)
    workflow.add_node("symbolic_to_code", symbolic_to_code_agent)
    workflow.add_node("verifier", verifier_agent)
    workflow.add_node("critique", critique_agent)
    workflow.add_node("corrector", corrector_agent)
    
    # Execution agent
    workflow.add_node("run_code_agent", run_code_agent)  # Simple execution with fallback
    
    # Utility nodes
    workflow.add_node("finish_file", finish_file)
    workflow.add_node("batch_test_generator", batch_test_generator)

    # Entry point - start with main planner
    workflow.set_entry_point("planner")
    
    # Planning phase routing - add requirements installation after planning
    workflow.add_conditional_edges(
        "planner", 
        lambda s: "requirements_installer" if s.get("plan_approved") and s.get("requirements") else route_strategy(s)
    )
    
    # Route from requirements installer to strategy selection
    workflow.add_conditional_edges(
        "requirements_installer",
        lambda s: route_strategy(s)
    )
    
    # Strategy execution
    workflow.add_edge("symbolic_reasoner", "symbolic_to_code")
    workflow.add_conditional_edges(
        "pseudocode_refiner", 
        lambda s: "pseudocode_refiner" if s.get("pseudocode_iterations_remaining", 0) > 0 else "pseudocode_to_code"
    )
    
    # Code generation to verification
    for node in ["nl_to_code", "pseudocode_to_code", "symbolic_to_code"]:
        workflow.add_edge(node, "verifier")
    
    # Verification and critique flow
    workflow.add_edge("verifier", "critique")
    workflow.add_conditional_edges("critique", check_verification)
    workflow.add_edge("corrector", "verifier")
    
    # Docker execution routing
    workflow.add_conditional_edges("run_code_agent", route_docker_execution)
    
    # File completion and continuation
    workflow.add_conditional_edges("finish_file", should_continue)
    workflow.add_edge("batch_test_generator", END)
    
    return workflow.compile()

def create_enhanced_execution_config():
    """Create configuration for enhanced execution modes"""
    return {
        "execution_modes": {
            "unified_docker": {
                "description": "Single container with dynamic CMD modification",
                "benefits": ["Faster execution", "Resource efficient", "LLM-powered CMD"],
                "best_for": ["Multi-file projects", "Iterative development"]
            },
            "batch": {
                "description": "Parallel execution of multiple files with batch test generation",
                "benefits": ["Parallel processing", "Batch API calls", "Comprehensive testing"],
                "best_for": ["Large projects", "Test-driven development"]
            },
            "docker": {
                "description": "Standard Docker execution (legacy)",
                "benefits": ["Simple", "Isolated"],
                "best_for": ["Single files", "Simple scripts"]
            }
        },
        "default_mode": "unified_docker",
        "parallel_test_generation": True,
        "batch_api_calls": True
    }

# Export the enhanced graph
enhanced_graph = build_enhanced_graph()
execution_config = create_enhanced_execution_config()
