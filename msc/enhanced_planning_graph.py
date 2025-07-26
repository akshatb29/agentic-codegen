# msc/enhanced_planning_graph.py
"""
Enhanced graph with intelligent planning-based requirements installation
"""
from langgraph.graph import StateGraph, END
from msc.state import AgentState
from msc.agents import (
    planner_agent, agentic_planner_agent, symbolic_reasoner_agent, pseudocode_refiner_agent,
    nl_to_code_agent, pseudocode_to_code_agent, symbolic_to_code_agent,
    verifier_agent, critique_agent, corrector_agent
)
from msc.agents.requirements_installer import requirements_installer_agent

def should_continue(state: AgentState) -> str:
    if not state.get("file_plan_iterator"):
        return END
    return "agentic_planner"

def route_strategy(state: AgentState) -> str:
    if not state.get("gen_strategy_approved"):
        return "agentic_planner"
    strategy = state.get("chosen_gen_strategy")
    return {"Symbolic": "symbolic_reasoner", "Pseudocode": "pseudocode_refiner"}.get(strategy, "nl_to_code")

def route_after_planning(state: AgentState) -> str:
    """Route after planning to install requirements first"""
    # Check if we have requirements to install
    software_design = state.get("software_design", {})
    
    if isinstance(software_design, dict):
        requirements = software_design.get("requirements", [])
    else:
        requirements = getattr(software_design, 'requirements', [])
    
    # If we have requirements and haven't installed them yet, install first
    if requirements and not state.get("requirements_installation", {}).get("success", False):
        return "requirements_installer"
    else:
        # No requirements or already installed, proceed to generation
        return route_strategy(state)

def route_after_requirements_install(state: AgentState) -> str:
    """Route after requirements installation to generation"""
    return route_strategy(state)

def check_verification_enhanced(state: AgentState) -> str:
    """Enhanced verification routing"""
    verifier_report = state.get("verifier_report", {})
    
    # If execution was successful, we're done
    if verifier_report.get("success", False):
        return "finish_file"
    
    # Check if we've tried too many corrections
    correction_attempts = state.get("correction_attempts", 0)
    if correction_attempts >= 3:
        return "finish_file"
    
    # Otherwise, try correction
    return "corrector"

def finish_file(state: AgentState) -> dict:
    print(f"✅ SUCCESS: Finished processing {state['current_file_name']}")
    state["file_plan_iterator"].pop(0)
    return {
        "gen_strategy_approved": False, 
        "corrected_code": None, 
        "generated_code": None
    }

def build_enhanced_planning_graph() -> StateGraph:
    """Build graph with intelligent planning-based requirements installation"""
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
    
    # Requirements installer
    workflow.add_node("requirements_installer", requirements_installer_agent)
    
    workflow.add_node("finish_file", finish_file)

    # Entry point
    workflow.set_entry_point("agentic_planner")
    
    # Planning phase with requirements installation
    workflow.add_conditional_edges("agentic_planner", lambda s: "agentic_planner" if not s.get("plan_approved") else route_after_planning(s))
    workflow.add_conditional_edges("planner", lambda s: "planner" if not s.get("plan_approved") else route_after_planning(s))
    
    # Requirements installation flow
    workflow.add_conditional_edges("requirements_installer", route_after_requirements_install)
    
    # Generation phase
    workflow.add_edge("symbolic_reasoner", "symbolic_to_code")
    workflow.add_conditional_edges("pseudocode_refiner", lambda s: "pseudocode_refiner" if s.get("pseudocode_iterations_remaining", 0) > 0 else "pseudocode_to_code")
    
    # Verification and correction
    for node in ["nl_to_code", "pseudocode_to_code", "symbolic_to_code"]:
        workflow.add_edge(node, "verifier")
    
    workflow.add_conditional_edges("verifier", check_verification_enhanced)
    workflow.add_edge("corrector", "verifier")
    
    # File completion
    workflow.add_conditional_edges("finish_file", should_continue)
    
    return workflow.compile()

# Export the enhanced graph
def get_enhanced_planning_graph():
    """Get the enhanced graph with planning-based requirements installation"""
    return build_enhanced_planning_graph()
