# msc/enhanced_package_graph.py
"""
Enhanced graph with intelligent package management subgraph
"""
from langgraph.graph import StateGraph, END
from msc.state import AgentState
from msc.agents import (
    planner_agent, agentic_planner_agent, symbolic_reasoner_agent, pseudocode_refiner_agent,
    nl_to_code_agent, pseudocode_to_code_agent, symbolic_to_code_agent,
    verifier_agent, critique_agent, corrector_agent
)
from msc.agents.package_manager import (
    package_analyzer_agent, package_installer_agent, dependency_resolver_agent
)

def should_continue(state: AgentState) -> str:
    if not state.get("file_plan_iterator"):
        return END
    return "agentic_planner"

def route_strategy(state: AgentState) -> str:
    if not state.get("gen_strategy_approved"):
        return "agentic_planner"
    strategy = state.get("chosen_gen_strategy")
    return {"Symbolic": "symbolic_reasoner", "Pseudocode": "pseudocode_refiner"}.get(strategy, "nl_to_code")

def route_after_generation(state: AgentState) -> str:
    """Route to package analysis before verification"""
    return "package_analyzer"

def route_after_package_analysis(state: AgentState) -> str:
    """Decide whether to pre-install packages"""
    analysis = state.get("package_analysis", {})
    confidence = analysis.get("confidence", 0)
    packages = analysis.get("packages", [])
    
    # If high confidence and packages detected, pre-install
    if confidence > 0.8 and packages:
        return "package_installer"
    else:
        return "verifier"

def route_after_package_install(state: AgentState) -> str:
    """Route to verification after package installation"""
    return "verifier"

def check_verification_enhanced(state: AgentState) -> str:
    """Enhanced verification routing with package management"""
    verifier_report = state.get("verifier_report", {})
    
    # If execution was successful, we're done
    if verifier_report.get("success", False):
        return "finish_file"
    
    # Check if it's a dependency-related error
    error_output = verifier_report.get("stderr", "") + verifier_report.get("stdout", "")
    
    dependency_keywords = [
        "ModuleNotFoundError", "ImportError", "No module named",
        "cannot import", "pip install", "package not found"
    ]
    
    is_dependency_error = any(keyword in error_output for keyword in dependency_keywords)
    correction_attempts = state.get("correction_attempts", 0)
    
    # If it's a dependency error and we haven't tried dependency resolution
    if is_dependency_error and not state.get("dependency_resolution_attempted", False):
        return "dependency_resolver"
    
    # If we've tried too many corrections, give up
    if correction_attempts >= 3:
        return "finish_file"
    
    # Otherwise, try regular correction
    return "corrector"

def route_after_dependency_resolution(state: AgentState) -> str:
    """Route after dependency resolution"""
    resolution = state.get("dependency_resolution", {})
    action = resolution.get("action", "none")
    
    if action == "install_packages":
        return "package_installer"
    elif action in ["fix_imports", "version_conflict"]:
        return "corrector"  # Let corrector handle code fixes
    else:
        return "verifier"  # Try verification again

def finish_file(state: AgentState) -> dict:
    print(f"✅ SUCCESS: Finished processing {state['current_file_name']}")
    state["file_plan_iterator"].pop(0)
    return {
        "gen_strategy_approved": False, 
        "corrected_code": None, 
        "generated_code": None,
        "package_analysis": None,
        "package_installation": None,
        "dependency_resolution": None,
        "dependency_resolution_attempted": False
    }

def mark_dependency_resolution_attempted(state: AgentState) -> dict:
    """Mark that we've attempted dependency resolution"""
    return {"dependency_resolution_attempted": True}

def build_enhanced_package_graph() -> StateGraph:
    """Build graph with intelligent package management"""
    workflow = StateGraph(AgentState)
    
    # Core agents
    workflow.add_node("agentic_planner", agentic_planner_agent)
    workflow.add_node("planner", planner_agent)
    workflow.add_node("symbolic_reasoner", symbolic_reasoner_agent)
    workflow.add_node("pseudocode_refiner", pseudocode_refiner_agent)
    workflow.add_node("nl_to_code", nl_to_code_agent)
    workflow.add_node("pseudocode_to_code", pseudocode_to_code_agent)
    workflow.add_node("symbolic_to_code", symbolic_to_code_agent)
    workflow.add_node("verifier", verifier_agent)
    workflow.add_node("critique", critique_agent)
    workflow.add_node("corrector", corrector_agent)
    
    # Package management agents
    workflow.add_node("package_analyzer", package_analyzer_agent)
    workflow.add_node("package_installer", package_installer_agent)
    workflow.add_node("dependency_resolver", dependency_resolver_agent)
    workflow.add_node("mark_dep_resolution", mark_dependency_resolution_attempted)
    
    workflow.add_node("finish_file", finish_file)

    # Entry point
    workflow.set_entry_point("agentic_planner")
    
    # Planning phase
    workflow.add_conditional_edges("agentic_planner", lambda s: "agentic_planner" if not s.get("plan_approved") else route_strategy(s))
    workflow.add_conditional_edges("planner", lambda s: "planner" if not s.get("plan_approved") else route_strategy(s))
    
    # Generation phase
    workflow.add_edge("symbolic_reasoner", "symbolic_to_code")
    workflow.add_conditional_edges("pseudocode_refiner", lambda s: "pseudocode_refiner" if s.get("pseudocode_iterations_remaining", 0) > 0 else "pseudocode_to_code")
    
    # Route to package analysis after code generation
    for node in ["nl_to_code", "pseudocode_to_code", "symbolic_to_code"]:
        workflow.add_edge(node, "package_analyzer")
    
    # Package management flow
    workflow.add_conditional_edges("package_analyzer", route_after_package_analysis)
    workflow.add_edge("package_installer", "verifier")
    
    # Enhanced verification with dependency resolution
    workflow.add_conditional_edges("verifier", check_verification_enhanced)
    
    # Dependency resolution flow
    workflow.add_edge("dependency_resolver", "mark_dep_resolution")
    workflow.add_conditional_edges("mark_dep_resolution", route_after_dependency_resolution)
    
    # Correction flow
    workflow.add_edge("corrector", "verifier")
    
    # File completion
    workflow.add_conditional_edges("finish_file", should_continue)
    
    return workflow.compile()

# Export the enhanced graph
def get_enhanced_graph():
    """Get the enhanced graph with package management"""
    return build_enhanced_package_graph()
