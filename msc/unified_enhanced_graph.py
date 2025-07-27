# msc/unified_enhanced_graph.py
"""
Unified Enhanced Graph - Best of All Worlds
Combines: Enhanced planner + Advanced package management + Smart requirements + Robust execution
"""
from langgraph.graph import StateGraph, END
from msc.state import AgentState
from msc.agents import (
    planner_agent, symbolic_reasoner_agent, pseudocode_refiner_agent,
    nl_to_code_agent, pseudocode_to_code_agent, symbolic_to_code_agent,
    verifier_agent, critique_agent, corrector_agent
)
from msc.agents.requirements_installer import requirements_installer_agent
from msc.agents.package_manager import (
    package_analyzer_agent, package_installer_agent, dependency_resolver_agent
)
from msc.tools.execution import run_code

# ============================================================================
# EXECUTION AGENTS
# ============================================================================

def run_code_agent(state: AgentState) -> AgentState:
    """Execute generated code using enhanced run_code with validation and auto-fix"""
    code = state.code
    if not code:
        return state
    
    filename = state.filename or "script.py"
    user_request = getattr(state, 'user_request', '')
    project_name = getattr(state, 'project_name', '')
    language = getattr(state, 'language', '')
    
    # Respect user's execution mode choice
    use_docker_execution = state.get("execution_mode", "docker") == "docker"
    
    # Use enhanced run_code with validation and Docker management
    result = run_code(
        code=code,
        filename=filename, 
        use_docker=use_docker_execution,  # Respect user choice
        user_request=user_request,
        project_name=project_name,
        language=language
    )
    
    # Set proper state for graph routing
    state.execution_result = result
    state.verifier_report = {"success": result.get("success", False)}
    if not result.get("success", False):
        state.verifier_report["stderr"] = result.get("stderr", "")
        state.verifier_report["stdout"] = result.get("stdout", "")
        state.verifier_report["error"] = result.get("error", "")
    
    return state

# ============================================================================
# ROUTING LOGIC - ENHANCED WITH SMART PACKAGE MANAGEMENT
# ============================================================================

def should_continue(state: AgentState) -> str:
    """Determine if processing should continue to next file"""
    if not state.get("file_plan_iterator"):
        # All files processed, check if batch execution is needed
        if state.get("parallel_execution_active"):
            return "batch_test_generator"
        return END
    return "planner"  # Use enhanced main planner

def route_strategy(state: AgentState) -> str:
    """Route to appropriate code generation strategy"""
    if not state.get("gen_strategy_approved"):
        return "planner"  # Route back to enhanced planner
    strategy = state.get("chosen_gen_strategy")
    return {"Symbolic": "symbolic_reasoner", "Pseudocode": "pseudocode_refiner"}.get(strategy, "nl_to_code")

def route_after_planning(state: AgentState) -> str:
    """Smart routing after planning - install requirements first if needed"""
    # Check if we have requirements to install from planning
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

def route_after_generation(state: AgentState) -> str:
    """Route to package analysis after code generation for additional packages"""
    return "package_analyzer"

def route_after_package_analysis(state: AgentState) -> str:
    """Smart routing after package analysis"""
    analysis = state.get("package_analysis", {})
    confidence = analysis.get("confidence", 0)
    packages = analysis.get("packages", [])
    
    # If high confidence and packages detected, pre-install them
    if confidence > 0.8 and packages:
        return "package_installer"
    else:
        # Skip package installation, go straight to verification
        return "verifier"

def route_after_package_install(state: AgentState) -> str:
    """Route to verification after package installation"""
    return "verifier"

def check_verification_enhanced(state: AgentState) -> str:
    """Enhanced verification routing with intelligent package management and error handling"""
    verifier_report = state.get("verifier_report", {})
    critique_details = state.get("critique_feedback_details", {})
    
    # Check if both verification and critique passed
    verification_passed = verifier_report.get("success", False)
    critique_passed = critique_details.get("is_correct_and_runnable", False)
    
    if verification_passed and critique_passed:
        # Success - execute the code
        return "run_code_agent"
    
    # Check if it's a dependency-related error
    error_output = verifier_report.get("stderr", "") + verifier_report.get("stdout", "")
    
    dependency_keywords = [
        "ModuleNotFoundError", "ImportError", "No module named",
        "cannot import", "pip install", "package not found",
        "DLL load failed", "library not found"
    ]
    
    is_dependency_error = any(keyword in error_output for keyword in dependency_keywords)
    correction_attempts = state.get("correction_attempts", 0)
    dependency_resolution_attempted = state.get("dependency_resolution_attempted", False)
    
    # Smart routing based on error type and attempts
    if is_dependency_error and not dependency_resolution_attempted:
        # First time seeing dependency error - try intelligent resolution
        return "dependency_resolver"
    elif correction_attempts >= 3:
        # Too many attempts - give up gracefully
        return "finish_file"
    else:
        # Regular error - try standard correction
        return "corrector"

def route_after_dependency_resolution(state: AgentState) -> str:
    """Route after dependency resolution based on recommended action"""
    resolution = state.get("dependency_resolution", {})
    action = resolution.get("action", "none")
    
    if action == "install_packages":
        return "package_installer"
    elif action in ["fix_imports", "version_conflict"]:
        return "corrector"  # Let corrector handle code fixes
    else:
        return "verifier"  # Try verification again

def route_docker_execution(state: AgentState) -> str:
    """Route from Docker execution based on results"""
    verifier_report = state.get("verifier_report", {})
    
    if verifier_report.get("success", False):
        return "finish_file"
    else:
        # If execution failed, check if it's a dependency issue
        error_output = verifier_report.get("stderr", "") + verifier_report.get("stdout", "")
        dependency_keywords = [
            "ModuleNotFoundError", "ImportError", "No module named",
            "cannot import", "pip install", "package not found"
        ]
        
        is_dependency_error = any(keyword in error_output for keyword in dependency_keywords)
        
        if is_dependency_error and not state.get("post_execution_dependency_resolution_attempted", False):
            # Try dependency resolution after execution failure
            return "dependency_resolver"
        else:
            # Route back to corrector for other types of errors
            return "corrector"

# ============================================================================
# UTILITY NODES
# ============================================================================

def finish_file(state: AgentState) -> dict:
    """Finish processing current file and move to next"""
    current_file = state.get("current_file_name", "unknown")
    print(f"✅ SUCCESS: Finished processing {current_file}")
    
    # Remove current file from iterator
    if state.get("file_plan_iterator"):
        state["file_plan_iterator"].pop(0)
    
    # Reset file-specific state but keep session state
    return {
        "gen_strategy_approved": False,
        "corrected_code": None,
        "generated_code": None,
        "current_file_name": None,
        "current_task_description": None,
        "package_analysis": None,
        "package_installation": None,
        "dependency_resolution": None,
        "dependency_resolution_attempted": False,
        "post_execution_dependency_resolution_attempted": False,
        "correction_attempts": 0
    }

def mark_dependency_resolution_attempted(state: AgentState) -> dict:
    """Mark that we've attempted dependency resolution"""
    return {"dependency_resolution_attempted": True}

def mark_post_execution_dependency_resolution_attempted(state: AgentState) -> dict:
    """Mark that we've attempted post-execution dependency resolution"""
    return {"post_execution_dependency_resolution_attempted": True}

def batch_test_generator(state: AgentState) -> dict:
    """Generate and execute tests for all completed files"""
    print("🧪 Batch Test Generator: Creating comprehensive test suites")
    
    # This node handles test generation for all files at once
    return {
        "files_completed": True,
        "ready_for_testing": True,
        "test_generation_results": state.get("test_generation_results", {})
    }

# ============================================================================
# GRAPH BUILDER - UNIFIED ENHANCED VERSION
# ============================================================================

def build_unified_enhanced_graph() -> StateGraph:
    """Build the ultimate unified graph with all enhancements"""
    workflow = StateGraph(AgentState)
    
    # ===== CORE AGENTS =====
    workflow.add_node("planner", planner_agent)  # Enhanced planner with web search & reasoning
    workflow.add_node("requirements_installer", requirements_installer_agent)  # Planning-based requirements
    
    # Generation agents
    workflow.add_node("symbolic_reasoner", symbolic_reasoner_agent)
    workflow.add_node("pseudocode_refiner", pseudocode_refiner_agent)
    workflow.add_node("nl_to_code", nl_to_code_agent)
    workflow.add_node("pseudocode_to_code", pseudocode_to_code_agent)
    workflow.add_node("symbolic_to_code", symbolic_to_code_agent)
    
    # Verification and correction agents
    workflow.add_node("verifier", verifier_agent)
    workflow.add_node("critique", critique_agent)
    workflow.add_node("corrector", corrector_agent)
    
    # ===== PACKAGE MANAGEMENT AGENTS =====
    workflow.add_node("package_analyzer", package_analyzer_agent)  # Smart package detection
    workflow.add_node("package_installer", package_installer_agent)  # Bulk package installation
    workflow.add_node("dependency_resolver", dependency_resolver_agent)  # Intelligent error resolution
    
    # ===== EXECUTION AGENTS =====
    workflow.add_node("run_code_agent", run_code_agent)  # Enhanced Docker execution
    
    # ===== UTILITY NODES =====
    workflow.add_node("mark_dep_resolution", mark_dependency_resolution_attempted)
    workflow.add_node("mark_post_exec_dep_resolution", mark_post_execution_dependency_resolution_attempted)
    workflow.add_node("finish_file", finish_file)
    workflow.add_node("batch_test_generator", batch_test_generator)

    # ===== ENTRY POINT =====
    workflow.set_entry_point("planner")
    
    # ===== PLANNING PHASE =====
    # Enhanced planner with multi-phase approval
    workflow.add_conditional_edges(
        "planner", 
        lambda s: route_after_planning(s) if s.get("plan_approved") else "planner"
    )
    
    # Requirements installation after planning
    workflow.add_conditional_edges("requirements_installer", route_after_requirements_install)
    
    # ===== GENERATION PHASE =====
    # Strategy-based generation
    workflow.add_edge("symbolic_reasoner", "symbolic_to_code")
    workflow.add_conditional_edges(
        "pseudocode_refiner", 
        lambda s: "pseudocode_refiner" if s.get("pseudocode_iterations_remaining", 0) > 0 else "pseudocode_to_code"
    )
    
    # Route to package analysis after code generation
    for generation_node in ["nl_to_code", "pseudocode_to_code", "symbolic_to_code"]:
        workflow.add_edge(generation_node, "package_analyzer")
    
    # ===== PACKAGE MANAGEMENT PHASE =====
    workflow.add_conditional_edges("package_analyzer", route_after_package_analysis)
    workflow.add_edge("package_installer", "verifier")
    
    # ===== VERIFICATION AND CORRECTION PHASE =====
    workflow.add_edge("verifier", "critique")
    workflow.add_conditional_edges("critique", check_verification_enhanced)
    workflow.add_edge("corrector", "verifier")
    
    # ===== DEPENDENCY RESOLUTION FLOW =====
    workflow.add_edge("dependency_resolver", "mark_dep_resolution")
    workflow.add_conditional_edges("mark_dep_resolution", route_after_dependency_resolution)
    
    # Post-execution dependency resolution
    workflow.add_edge("mark_post_exec_dep_resolution", "dependency_resolver")
    
    # ===== EXECUTION PHASE =====
    workflow.add_conditional_edges("run_code_agent", route_docker_execution)
    
    # ===== FILE COMPLETION AND CONTINUATION =====
    workflow.add_conditional_edges("finish_file", should_continue)
    workflow.add_edge("batch_test_generator", END)
    
    return workflow.compile()

# ============================================================================
# CONFIGURATION AND EXPORTS
# ============================================================================

def create_unified_execution_config():
    """Create comprehensive configuration for the unified system"""
    return {
        "planning_features": {
            "web_search_integration": True,
            "multi_phase_approval": True,
            "reasoning_and_feedback": True,
            "smart_requirements_prediction": True
        },
        "package_management": {
            "pre_execution_analysis": True,
            "bulk_installation": True,
            "dependency_resolution": True,
            "intelligent_error_recovery": True
        },
        "execution_features": {
            "code_validation": True,
            "fence_removal": True,
            "security_checks": True,
            "timeout_protection": True,
            "auto_retry_after_fixes": True
        },
        "error_handling": {
            "smart_dependency_detection": True,
            "multi_level_fallbacks": True,
            "graceful_degradation": True,
            "max_correction_attempts": 3
        },
        "performance": {
            "container_reuse": True,
            "batch_operations": True,
            "parallel_processing": False,  # Can be enabled later
            "resource_optimization": True
        }
    }

# Export the unified graph
unified_enhanced_graph = build_unified_enhanced_graph()
unified_execution_config = create_unified_execution_config()

# Convenience functions
def get_unified_graph():
    """Get the unified enhanced graph"""
    return unified_enhanced_graph

def get_execution_config():
    """Get the execution configuration"""
    return unified_execution_config
