# msc/graph.py
from langgraph.graph import StateGraph, END
from msc.state import AgentState
from msc.agents import (
    planner_agent, agentic_planner_agent, symbolic_reasoner_agent, pseudocode_refiner_agent,
    nl_to_code_agent, pseudocode_to_code_agent, symbolic_to_code_agent,
    verifier_agent, critique_agent, corrector_agent, docker_workflow_agent
)

def should_continue(state: AgentState) -> str:
    if not state.get("file_plan_iterator"):
        return END
    return "agentic_planner"

def route_strategy(state: AgentState) -> str:
    if not state.get("gen_strategy_approved"):
        return "agentic_planner"  # Route back to agentic planner
    strategy = state.get("chosen_gen_strategy")
    return {"Symbolic": "symbolic_reasoner", "Pseudocode": "pseudocode_refiner"}.get(strategy, "nl_to_code")

def check_verification(state: AgentState) -> str:
    if state["verifier_report"]["success"] and state["critique_feedback_details"]["is_correct_and_runnable"]:
        # If Docker execution is enabled, route to docker_agent for containerized testing
        if state.get("use_docker_execution", False):
            return "docker_agent"
        return "finish_file"
    if state.get("correction_attempts", 0) >= 2:
        return "finish_file"
    return "corrector"

def route_from_docker(state: AgentState) -> str:
    """Route from docker_agent based on execution results"""
    docker_results = state.get("docker_execution_results", {})
    if docker_results.get("success", False):
        return "finish_file"
    else:
        # If Docker execution failed, route back to corrector for fixes
        return "corrector"

def finish_file(state: AgentState) -> dict:
    print(f"✅ SUCCESS: Finished processing {state['current_file_name']}")
    state["file_plan_iterator"].pop(0)
    return {"gen_strategy_approved": False, "corrected_code": None, "generated_code": None}

def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)
    
    # Use agentic planner instead of regular planner
    workflow.add_node("agentic_planner", agentic_planner_agent)
    workflow.add_node("planner", planner_agent)  # Keep as fallback
    workflow.add_node("symbolic_reasoner", symbolic_reasoner_agent)
    workflow.add_node("pseudocode_refiner", pseudocode_refiner_agent)
    workflow.add_node("nl_to_code", nl_to_code_agent)
    workflow.add_node("pseudocode_to_code", pseudocode_to_code_agent)
    workflow.add_node("symbolic_to_code", symbolic_to_code_agent)
    workflow.add_node("verifier", verifier_agent)
    workflow.add_node("critique", critique_agent)
    workflow.add_node("corrector", corrector_agent)
    workflow.add_node("docker_agent", docker_workflow_agent)  # New Docker agent for containerized execution
    workflow.add_node("finish_file", finish_file)

    # Start with agentic planner
    workflow.set_entry_point("agentic_planner")
    
    # Route from agentic planner
    workflow.add_conditional_edges("agentic_planner", lambda s: "agentic_planner" if not s.get("plan_approved") else route_strategy(s))
    workflow.add_conditional_edges("planner", lambda s: "planner" if not s.get("plan_approved") else route_strategy(s))
    workflow.add_conditional_edges("finish_file", should_continue)
    
    workflow.add_edge("symbolic_reasoner", "symbolic_to_code")
    workflow.add_conditional_edges("pseudocode_refiner", lambda s: "pseudocode_refiner" if s.get("pseudocode_iterations_remaining", 0) > 0 else "pseudocode_to_code")
    
    for node in ["nl_to_code", "pseudocode_to_code", "symbolic_to_code"]:
        workflow.add_edge(node, "verifier")
        
    workflow.add_edge("verifier", "critique")
    workflow.add_conditional_edges("critique", check_verification)
    workflow.add_edge("corrector", "verifier")
    
    # Docker agent routing
    workflow.add_conditional_edges("docker_agent", route_from_docker)
    
    return workflow.compile()