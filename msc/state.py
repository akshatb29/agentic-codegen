# msc/state.py
import operator
from typing import TypedDict, List, Optional, Annotated, Dict, Any
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

# (Your AgentState, SoftwareDesign, GenerationStrategy, and CritiqueReport classes go here)
# ... (same as before) ...
class AgentState(TypedDict):
    user_request: str
    enable_got_planning: bool
    enable_symbolic_reasoning: bool
    enable_pseudocode_iterations: bool
    llm_model: str
    execution_mode: str
    messages: Annotated[List[BaseMessage], operator.add]
    thoughts_history: Annotated[List[Dict[str, Any]], operator.add]
    existing_file_context: Dict[str, str]
    
    # Tech stack and project structure
    tech_stack_approved: Optional[bool]
    tech_analysis: Optional[str]
    detected_app_type: Optional[str]
    docker_prep_started: Optional[bool]
    prepared_docker_spec: Optional[Dict[str, Any]]
    
    # Project structure planning
    project_structure_approved: Optional[bool]
    project_structure: Optional[str]
    project_name: Optional[str]
    main_file: Optional[str]
    requirements: Optional[List[str]]
    user_feedback_for_tech_replan: Optional[str]
    user_feedback_for_structure_replan: Optional[str]
    
    # Software design
    software_design: Optional[Dict[str, Any]]
    file_plan_iterator: Optional[List[Dict[str, Any]]]
    current_file_info: Optional[Dict[str, Any]]
    current_file_name: Optional[str]
    current_task_description: Optional[str]
    current_task_complexity_score: Optional[int]
    proof_needed: Optional[bool]
    chosen_gen_strategy: Optional[str]
    pseudocode_iterations_remaining: Optional[int]
    plan_approved: Optional[bool]
    gen_strategy_approved: Optional[bool]
    user_command: Optional[str]
    user_feedback_for_replan: Optional[str]
    user_feedback_for_rethink: Optional[str]
    
    # Generation and testing
    generated_pseudocode: Optional[str]
    generated_symbolic_representation: Optional[str]
    generated_code: Optional[str]
    file_path: Optional[str]
    execution_environment: Optional[str]
    verifier_report: Optional[Dict[str, Any]]
    critique_feedback_details: Optional[Dict[str, Any]]
    corrected_code: Optional[str]
    correction_attempts: int
    
    # New testing phase fields
    files_completed: Optional[bool]
    ready_for_testing: Optional[bool]
    individual_tests_passed: Optional[List[str]]
    individual_tests_failed: Optional[List[str]]
    full_app_test_result: Optional[Dict[str, Any]]
    container_name: Optional[str]

class SoftwareDesign(BaseModel):
    thought: str = Field(description="A brief thought process on how this design was chosen.")
    files: List[Dict[str, Any]] = Field(description="A list of files to be created, each with a name, purpose, etc.")

class GenerationStrategy(BaseModel):
    thought: str = Field(description="Reasoning for the chosen strategy.")
    current_task_complexity_score: int = Field(description="A score from 1-10 indicating task complexity.")
    proof_needed: bool = Field(description="Indicates if the task requires formal proof.")
    chosen_gen_strategy: str = Field(description="The chosen strategy: 'NL', 'Pseudocode', or 'Symbolic'.")

class CritiqueReport(BaseModel):
    summary: str = Field(description="A high-level summary of the code quality.")
    is_correct_and_runnable: bool = Field(description="Whether the code appears correct and addresses the task.")
    suggestions: List[str] = Field(description="Actionable suggestions for improvement.")

# New Pydantic models for explicit Graph-of-Thoughts
class BrainstormedDesigns(BaseModel):
    """A list of several potential software designs."""
    designs: List[SoftwareDesign] = Field(description="A list of 2-3 distinct software design approaches.")

class EvaluatedDesigns(BaseModel):
    """A list of evaluated designs with pros, cons, and a final recommendation."""
    thought: str = Field(description="Your thought process for evaluating the designs.")
    evaluations: List[Dict[str, Any]] = Field(description="A list of evaluations, each with pros, cons, and a score.")
    best_design_index: int = Field(description="The index of the best design from the original list.")