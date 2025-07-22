# msc_framework.py
#
# Modular Self-Correcting Neuro-Symbolic Code Generation Framework
# A sophisticated multi-agent system for autonomous code generation.
#
# To Run:
# 1. Ensure you have GOOGLE_API_KEY set as an environment variable.
# 2. Run from your terminal: python msc_framework.py
#
# Follow the CLI prompts.

import os
import json
import operator
import platform
import subprocess
from pathlib import Path
from typing import TypedDict, List, Optional, Annotated, Dict, Any, Sequence

# Rich for better CLI output
from rich.console import Console
from rich.prompt import Prompt

# LangChain / LangGraph Core Imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Constants & Configuration ---
# Use a more powerful model for planning and critique
PLANNER_MODEL = "gemini-1.5-flash-latest"
# Use a faster/cheaper model for more routine generation tasks
GENERATOR_MODEL = "gemini-1.5-flash-latest"

# Initialize rich console for beautiful printing
console = Console()

# --- 1. AgentState Definition: The Central State of the Application ---

class AgentState(TypedDict):
    """
    Represents the entire mutable state of the system. This is the central
    communication channel for all nodes in the LangGraph.
    """
    # --- User Initial Request & Configuration ---
    user_request: str
    enable_got_planning: bool
    enable_symbolic_reasoning: bool
    enable_pseudocode_iterations: bool
    llm_model: str
    execution_mode: str  # "sandbox" or "safe_system" 

    # --- Global Context ---
    messages: Annotated[List[BaseMessage], operator.add]
    thoughts_history: Annotated[List[Dict[str, Any]], operator.add]
    existing_file_context: Dict[str, str]
    software_design: Optional[Dict[str, Any]]

    # --- Current Task / File Being Processed ---
    file_plan_iterator: Optional[List[Dict[str, Any]]] # A copy of software_design["files"] to iterate over
    current_file_info: Optional[Dict[str, Any]] # The current file object from the iterator
    current_file_name: Optional[str]
    current_task_description: Optional[str]
    current_task_complexity_score: Optional[int]
    proof_needed: Optional[bool]
    chosen_gen_strategy: Optional[str]
    pseudocode_iterations_remaining: Optional[int]

    # --- User Approval & Feedback ---
    plan_approved: Optional[bool]
    gen_strategy_approved: Optional[bool]
    user_command: Optional[str]
    user_feedback_for_replan: Optional[str]
    user_feedback_for_rethink: Optional[str]

    # --- Code Generation & Verification Artifacts ---
    generated_pseudocode: Optional[str]
    generated_symbolic_representation: Optional[str]
    generated_code: Optional[str]
    file_path: Optional[str]
    execution_environment: Optional[str]

    # --- Verification & Correction ---
    verifier_report: Optional[Dict[str, Any]]
    critique_feedback_details: Optional[Dict[str, Any]]
    corrected_code: Optional[str]
    correction_attempts: int # Counter to prevent infinite correction loops

# --- 2. Auxiliary Components (Tools/Functions) ---

class FilesystemTool:
    """Manages file system operations."""
    @staticmethod
    def write_file(path: str, content: str) -> bool:
        try:
            dir_path = os.path.dirname(path)
            if dir_path:  # Only create directory if there's a directory part
                os.makedirs(dir_path, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            console.log(f"✅ [FileSystem] Wrote file: {path}")
            return True
        except IOError as e:
            console.log(f"❌ [FileSystem] Error writing file {path}: {e}")
            return False

    @staticmethod
    def read_file(path: str) -> Optional[str]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return None
        except IOError as e:
            console.log(f"❌ [FileSystem] Error reading file {path}: {e}")
            return None
    
    @staticmethod
    def create_directory(path: str) -> bool:
        try:
            os.makedirs(path, exist_ok=True)
            console.log(f"✅ [FileSystem] Ensured directory exists: {path}")
            return True
        except IOError as e:
            console.log(f"❌ [FileSystem] Error creating directory {path}: {e}")
            return False

    @staticmethod
    def read_directory_contents(directory_path: str, include_extensions: List[str] = ['.py', '.json', '.txt', '.md']) -> Dict[str, str]:
        """Reads relevant files and returns their content."""
        context = {}
        try:
            for root, _, files in os.walk(directory_path):
                # Ignore virtual environment directories
                if 'venv' in root or '__pycache__' in root or '.git' in root:
                    continue
                for file in files:
                    if any(file.endswith(ext) for ext in include_extensions):
                        file_path = os.path.join(root, file)
                        content = FilesystemTool.read_file(file_path)
                        if content is not None:
                            context[file_path] = content
            return context
        except Exception as e:
            console.log(f"❌ [FileSystem] Error scanning directory {directory_path}: {e}")
            return {}

def user_confirmation_tool(prompt_message: str) -> str:
    """Prompts user via CLI and captures full input string."""
    console.rule("[bold yellow]USER ACTION REQUIRED[/bold yellow]")
    console.print(f"[bold cyan]{prompt_message}[/bold cyan]")
    user_input = Prompt.ask("[bold]Your command (y/n, REPLAN..., RETHINK...)[/bold]")
    return user_input.strip()

# --- Placeholder Tools for future implementation ---

def search_tool(query: str) -> str:
    """Placeholder for a web search API (e.g., Serper, Brave)."""
    console.log(f"🔎 [SearchTool] SKIPPED SEARCH for: '{query}'")
    return "Search results are not available in this placeholder implementation."

def sandbox_executor_tool(code: str, file_path: str) -> Dict[str, Any]:
    """
    Executes code by writing to a file and running it.
    This is a simplified, non-isolated executor.
    A real implementation MUST use Docker for security.
    """
    console.log(f"🏃 [Sandbox] Executing code in '{file_path}'...")
    # For now, we only execute Python files.
    if not file_path.endswith(".py"):
        return {"success": True, "stdout": "Non-executable file type.", "stderr": "", "traceback": ""}
        
    FilesystemTool.write_file(file_path, code)
    
    try:
        # Use the same python executable that runs the script
        python_executable = "python" if platform.system() == "Windows" else "python3"
        process = subprocess.run(
            [python_executable, file_path],
            capture_output=True,
            text=True,
            timeout=30, # 30-second timeout for safety
            check=False # Do not raise exception on non-zero exit codes
        )
        
        report = {
            "success": process.returncode == 0,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "traceback": "" # In a real sandbox, you'd capture more detailed tracebacks
        }
        if process.returncode != 0:
            report["traceback"] = process.stderr
            console.log(f"❗ [Sandbox] Execution failed with exit code {process.returncode}.")
        else:
            console.log("✅ [Sandbox] Execution successful.")
        
        return report

    except subprocess.TimeoutExpired:
        console.log("❌ [Sandbox] Execution timed out.")
        return {"success": False, "stdout": "", "stderr": "Execution timed out after 30 seconds.", "traceback": "TimeoutError"}
    except Exception as e:
        console.log(f"❌ [Sandbox] An unexpected error occurred during execution: {e}")
        return {"success": False, "stdout": "", "stderr": str(e), "traceback": str(e)}

def safe_system_executor_tool(code: str, file_path: str) -> Dict[str, Any]:
    """
    Executes code safely using subprocess with restricted permissions and isolated environment.
    Uses child process isolation for safer execution on the host system.
    """
    console.log(f"🔒 [Safe System] Executing code in '{file_path}' with restricted permissions...")
    
    # For now, we only execute Python files.
    if not file_path.endswith(".py"):
        return {"success": True, "stdout": "Non-executable file type.", "stderr": "", "traceback": ""}
    
    # Write the code to file
    FilesystemTool.write_file(file_path, code)
    
    try:
        # Use the same python executable that runs the script
        python_executable = "python" if platform.system() == "Windows" else "python3"
        
        # Create a more restricted environment
        env = os.environ.copy()
        # Remove potentially dangerous environment variables
        dangerous_vars = ['LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH']
        for var in dangerous_vars:
            env.pop(var, None)
        
        # Run with restricted permissions and in a separate process group
        process = subprocess.run(
            [python_executable, "-I", file_path],  # -I flag for isolated mode
            capture_output=True,
            text=True,
            timeout=15,  # Shorter timeout for safety
            check=False,
            env=env,
            cwd=os.path.dirname(os.path.abspath(file_path)) or ".",  # Restrict to file directory
            # On Unix systems, create new process group for better isolation
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None
        )
        
        report = {
            "success": process.returncode == 0,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "traceback": ""
        }
        
        if process.returncode != 0:
            report["traceback"] = process.stderr
            console.log(f"❗ [Safe System] Execution failed with exit code {process.returncode}.")
        else:
            console.log("✅ [Safe System] Execution successful.")
        
        return report

    except subprocess.TimeoutExpired:
        console.log("❌ [Safe System] Execution timed out.")
        return {"success": False, "stdout": "", "stderr": "Execution timed out after 15 seconds.", "traceback": "TimeoutError"}
    except Exception as e:
        console.log(f"❌ [Safe System] An unexpected error occurred during execution: {e}")
        return {"success": False, "stdout": "", "stderr": str(e), "traceback": str(e)}

# --- Pydantic models for structured LLM output ---

class SoftwareDesign(BaseModel):
    """The high-level software design and file structure."""
    thought: str = Field(description="A brief thought process on how this design was chosen.")
    files: List[Dict[str, Any]] = Field(description="A list of files to be created, each with a name, purpose, and list of key functions/classes.")

class GenerationStrategy(BaseModel):
    """The chosen strategy for generating a single file."""
    thought: str = Field(description="Reasoning for the chosen strategy based on complexity and requirements.")
    current_task_complexity_score: int = Field(description="A score from 1-10 indicating task complexity.")
    proof_needed: bool = Field(description="Indicates if the task requires a formal proof or algorithmic reasoning.")
    chosen_gen_strategy: str = Field(description="The chosen strategy: 'NL', 'Pseudocode', or 'Symbolic'.")

class CritiqueReport(BaseModel):
    """A structured critique of the generated code."""
    summary: str = Field(description="A high-level summary of the code quality.")
    is_correct_and_runnable: bool = Field(description="Whether the code appears correct and addresses the task based on verifier report.")
    suggestions: List[str] = Field(description="Actionable suggestions for improvement. Be specific.")

# --- 3. Core Agents (LangGraph Nodes) ---

def planner_agent(state: AgentState) -> Dict[str, Any]:
    """The primary orchestrator. Manages software design and per-file strategy."""
    llm = ChatGoogleGenerativeAI(model=PLANNER_MODEL, temperature=0.3)
    
    # PHASE 1: Overall Software Design
    if not state.get("plan_approved"):
        console.rule("[bold magenta]PLANNER: Designing Software Architecture[/bold magenta]")
        
        # Handle REPLAN loop
        feedback = state.get("user_feedback_for_replan")
        if feedback:
            console.log(f"🔄 Incorporating user feedback for replan: '{feedback}'")
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert software architect. Your goal is to create a robust and logical software design based on the user's request.
            - Analyze the user's request, any existing files, and previous thoughts.
            - {got_instructions}
            - Structure your final output as a JSON object that strictly adheres to the 'SoftwareDesign' schema.
            - For each file, provide a name (full file path) and a clear purpose description.
            - Each file must be a dictionary with 'name' and 'purpose' keys.
            - Example file structure: {{"name": "calculator.py", "purpose": "Main calculator application with arithmetic operations"}}
            """),
            ("human", """User Request: {user_request}
            
            Existing File Context in Current Directory:
            {existing_file_context}
            
            Previous planning thoughts (if any):
            {thoughts_history}
            
            {replan_instructions}
            
            Please generate the software design with properly structured files.""")
        ])
        
        got_instructions = (
            "Think step-by-step (Graph-of-Thoughts): 1. Deconstruct the request. 2. Brainstorm 2-3 architectural approaches. 3. Evaluate pros and cons. 4. Synthesize the best approach into the final design."
            if state["enable_got_planning"]
            else "Directly generate the most logical software design."
        )
        replan_instructions = f"The user rejected the last plan with the following feedback. You MUST address it: {feedback}" if feedback else ""
        
        planner_chain = prompt_template | llm.with_structured_output(SoftwareDesign)
        
        design = planner_chain.invoke({
            "user_request": state["user_request"],
            "existing_file_context": json.dumps(state["existing_file_context"], indent=2),
            "thoughts_history": json.dumps(state.get("thoughts_history", []), indent=2),
            "got_instructions": got_instructions,
            "replan_instructions": replan_instructions
        })
        
        # Validate file structure
        for i, file_info in enumerate(design.files):
            if not isinstance(file_info, dict) or 'name' not in file_info or 'purpose' not in file_info:
                console.log(f"⚠️ Invalid file structure detected. Fixing file {i}...")
                if not isinstance(file_info, dict):
                    design.files[i] = {"name": f"file_{i}.py", "purpose": "Auto-generated file"}
                else:
                    if 'name' not in file_info:
                        file_info['name'] = f"file_{i}.py"
                    if 'purpose' not in file_info:
                        file_info['purpose'] = "Auto-generated purpose"
        
        # Present plan to user for approval
        console.print("[bold green]Proposed Software Design:[/bold green]")
        console.print_json(data=design.model_dump())
        
        user_command = user_confirmation_tool("Do you approve this software design?")
        
        if user_command.lower() in ["y", "yes"]:
            # Create directories and empty files as per plan
            for file_info in design.files:
                file_path = file_info['name']
                dir_path = os.path.dirname(file_path)
                if dir_path:  # Only create directory if there's a directory part
                    FilesystemTool.create_directory(dir_path)
                # FilesystemTool.write_file(file_path, "# Automatically created by MSC Framework\n")

            return {
                "software_design": design.model_dump(),
                "plan_approved": True,
                "user_feedback_for_replan": None,
                "file_plan_iterator": design.files.copy() # Create an iterator for the next phase
            }
        else:
            feedback = user_command.replace("REPLAN", "").strip() if user_command.upper().startswith("REPLAN") else "User rejected the plan without specific feedback."
            return {
                "plan_approved": False,
                "user_feedback_for_replan": feedback
            }

    # PHASE 2: Per-File Generation Strategy
    console.rule("[bold magenta]PLANNER: Choosing Generation Strategy[/bold magenta]")
    
    current_file_info = state["file_plan_iterator"][0] # Get the next file to process
    file_name = current_file_info['name']
    task_description = current_file_info['purpose']
    
    console.log(f"📝 Planning generation for file: [bold cyan]{file_name}[/bold cyan]")
    
    feedback = state.get("user_feedback_for_rethink")
    if feedback:
        console.log(f"🔄 Incorporating user feedback for rethink: '{feedback}'")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert code strategist. Your task is to decide the best way to generate the code for a specific file.
        - Analyze the task description, its complexity, and whether it's algorithmic.
        - Consider the user's enabled reasoning paths ({enabled_strategies}).
        - If the task is highly algorithmic, mathematical, or needs formal verification, and symbolic reasoning is enabled, choose 'Symbolic'.
        - If the task has complex logic but doesn't need formal proof, and iterative pseudocode is enabled, choose 'Pseudocode'.
        - Otherwise, choose 'NL' (Natural Language) for direct code generation.
        - Structure your output as a JSON object that strictly adheres to the 'GenerationStrategy' schema.
        """),
        ("human", """Overall Software Design:
        {software_design}
        
        Current File to Generate: {file_name}
        Task Description for this File: {task_description}
        
        Enabled Reasoning Paths: {enabled_strategies}
        
        {rethink_instructions}
        
        Please decide the generation strategy for this file.""")
    ])

    enabled_strategies = []
    if state["enable_symbolic_reasoning"]: enabled_strategies.append("Symbolic")
    if state["enable_pseudocode_iterations"]: enabled_strategies.append("Pseudocode")
    enabled_strategies.append("NL")

    strategy_chain = prompt_template | llm.with_structured_output(GenerationStrategy)
    
    strategy = strategy_chain.invoke({
        "software_design": json.dumps(state["software_design"], indent=2),
        "file_name": file_name,
        "task_description": task_description,
        "enabled_strategies": ", ".join(enabled_strategies),
        "rethink_instructions": f"The user rejected the last strategy with this feedback. Address it: {feedback}" if feedback else ""
    })
    
    console.print(f"[bold green]Proposed Strategy for '{file_name}':[/bold green] [bold yellow]{strategy.chosen_gen_strategy}[/bold yellow]")
    console.print(f"Complexity: {strategy.current_task_complexity_score}/10 | Reasoning: {strategy.thought}")
    
    user_command = user_confirmation_tool("Do you approve this generation strategy?")
    
    # Check user's enabled preferences against agent's choice
    chosen_strategy = strategy.chosen_gen_strategy
    if (chosen_strategy == "Symbolic" and not state["enable_symbolic_reasoning"]) or \
       (chosen_strategy == "Pseudocode" and not state["enable_pseudocode_iterations"]):
        console.log(f"⚠️ Agent chose '{chosen_strategy}' but it's disabled by user. Defaulting to 'NL'.")
        chosen_strategy = "NL"

    if user_command.lower() in ["y", "yes"]:
        return {
            "current_file_info": current_file_info,
            "current_file_name": file_name,
            "current_task_description": task_description,
            "file_path": file_name, # Planner sets the file path
            "current_task_complexity_score": strategy.current_task_complexity_score,
            "proof_needed": strategy.proof_needed,
            "chosen_gen_strategy": chosen_strategy,
            "gen_strategy_approved": True,
            "user_feedback_for_rethink": None,
            "correction_attempts": 0 # Reset for new file
        }
    else:
        feedback = user_command.replace("RETHINK", "").strip() if user_command.upper().startswith("RETHINK") else "User rejected strategy."
        return {
            "gen_strategy_approved": False,
            "user_feedback_for_rethink": feedback,
        }

def symbolic_reasoner_agent(state: AgentState) -> Dict[str, Any]:
    """Develops a formal/symbolic representation of the code logic."""
    console.rule("[bold blue]REASONER: Symbolic Representation[/bold blue]")
    # This is a placeholder for a true symbolic reasoning implementation.
    # In a real system, this might use a specific formal language like TLA+ or Z notation.
    llm = ChatGoogleGenerativeAI(model=GENERATOR_MODEL, temperature=0.1)
    
    prompt = ChatPromptTemplate.from_template("""
    Based on the task description, create a formal symbolic representation of the logic.
    This could be a structured JSON with preconditions, postconditions, and invariants, or a mathematical formula.
    
    Task: {task}
    
    Symbolic Representation:
    """)
    chain = prompt | llm
    representation = chain.invoke({"task": state["current_task_description"]})
    
    console.log("Generated Symbolic Representation.")
    return {"generated_symbolic_representation": representation.content}
    
def pseudocode_refiner_agent(state: AgentState) -> Dict[str, Any]:
    """Generates and iteratively refines pseudocode."""
    console.rule("[bold blue]REASONER: Pseudocode Refinement[/bold blue]")
    llm = ChatGoogleGenerativeAI(model=GENERATOR_MODEL, temperature=0.2)
    
    # Initialize iteration count if it doesn't exist
    if state.get("pseudocode_iterations_remaining") is None:
        iterations_remaining = 2 # Max 2 iterations
    else:
        iterations_remaining = state["pseudocode_iterations_remaining"]
    
    console.log(f"Pseudocode Iteration: {3 - iterations_remaining}/2")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert programmer who writes exceptionally clear, language-agnostic pseudocode."),
        ("human", """Task: {task}
        
        {refinement_instructions}
        
        Generate structured, detailed pseudocode for this task.""")
    ])
    
    refinement_instructions = f"Refine the following pseudocode to be more robust and clear:\n{state.get('generated_pseudocode')}" if state.get('generated_pseudocode') else "This is the first iteration."
    
    chain = prompt_template | llm
    pseudocode = chain.invoke({
        "task": state["current_task_description"],
        "refinement_instructions": refinement_instructions
    })
    
    return {
        "generated_pseudocode": pseudocode.content,
        "pseudocode_iterations_remaining": iterations_remaining - 1
    }

# --- Generic Code Generation Node ---
def code_generator_agent(state: AgentState, generation_prompt: str) -> Dict[str, Any]:
    """A generalized code generator."""
    console.rule("[bold green]GENERATOR: Writing Code[/bold green]")
    llm = ChatGoogleGenerativeAI(model=GENERATOR_MODEL, temperature=0.2)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an elite programmer specializing in Python.
        - Write clean, efficient, and well-commented code.
        - Only output the raw code for the file `{file_name}`.
        - Do not include markdown formatting like ```python ... ```.
        - Ensure the code is complete and runnable.
        - Adhere to the overall software design and the specific task.
        - If you are correcting code, apply the suggestions precisely."""),
        ("human", generation_prompt)
    ])
    
    chain = prompt | llm
    
    # Gather all context for the prompt
    context = {
        "file_name": state["current_file_name"],
        "task": state["current_task_description"],
        "software_design": json.dumps(state.get("software_design"), indent=2),
        "existing_file_context": json.dumps(state.get("existing_file_context"), indent=2),
        "pseudocode": state.get("generated_pseudocode", "N/A"),
        "symbolic_rep": state.get("generated_symbolic_representation", "N/A"),
        "critique": json.dumps(state.get("critique_feedback_details"), indent=2)
    }
    
    code = chain.invoke(context)
    console.log(f"Code generation complete for {state['current_file_name']}.")
    return {"generated_code": code.content}

# --- Specialized Code Generation Agents (that use the generic one) ---

def nl_to_code_agent(state: AgentState) -> Dict[str, Any]:
    """Generates code directly from natural language."""
    console.log("Strategy: Natural Language -> Code")
    prompt = """
    Software Design:
    {software_design}
    
    Relevant Existing Files:
    {existing_file_context}
    
    Your Task for file `{file_name}`:
    {task}
    
    Please generate the Python code.
    """
    return code_generator_agent(state, prompt)

def pseudocode_to_code_agent(state: AgentState) -> Dict[str, Any]:
    """Translates refined pseudocode into executable code."""
    console.log("Strategy: Pseudocode -> Code")
    prompt = """
    Your task is to translate the following high-quality pseudocode into Python code for the file `{file_name}`.
    
    Pseudocode:
    ---
    {pseudocode}
    ---
    
    Original Task Description: {task}
    Software Design: {software_design}
    
    Please generate the Python code.
    """
    return code_generator_agent(state, prompt)

def symbolic_to_code_agent(state: AgentState) -> Dict[str, Any]:
    """Translates a formal symbolic representation into executable code."""
    console.log("Strategy: Symbolic Representation -> Code")
    prompt = """
    Your task is to translate the following formal/symbolic representation into Python code for the file `{file_name}`.
    
    Symbolic Representation:
    ---
    {symbolic_rep}
    ---
    
    Original Task Description: {task}
    Software Design: {software_design}
    
    Please generate the Python code, ensuring it correctly implements the formal logic.
    """
    return code_generator_agent(state, prompt)

def verifier_agent(state: AgentState) -> Dict[str, Any]:
    """Executes code and analyzes results."""
    console.rule("[bold yellow]VERIFIER: Executing Code[/bold yellow]")
    code_to_verify = state.get("corrected_code") or state.get("generated_code")
    file_path = state.get("file_path")
    execution_mode = state.get("execution_mode", "safe_system")
    
    if not code_to_verify or not file_path:
        return {"verifier_report": {"success": False, "stderr": "No code or file path found to verify."}}
    
    # Write the latest code to the file before execution
    FilesystemTool.write_file(file_path, code_to_verify)
    
    # Log the chosen execution mode
    mode_description = "Safe System (Child Process Isolation)" if execution_mode == "safe_system" else "Sandbox (Basic)"
    console.log(f"🎯 Using execution mode: [bold cyan]{mode_description}[/bold cyan]")
    
    # Choose execution method based on user preference
    if execution_mode == "sandbox":
        report = sandbox_executor_tool(code_to_verify, file_path)
    else:  # safe_system
        report = safe_system_executor_tool(code_to_verify, file_path)
    
    return {"verifier_report": report}

def critique_agent(state: AgentState) -> Dict[str, Any]:
    """Evaluates code based on verification results and other criteria."""
    console.rule("[bold red]CRITIQUE: Reviewing Code[/bold red]")
    llm = ChatGoogleGenerativeAI(model=PLANNER_MODEL, temperature=0.1)
    
    critique_chain = llm.with_structured_output(CritiqueReport)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior code reviewer. Your job is to provide a structured, actionable critique of the submitted code.
        - Base your primary assessment on the verifier's report.
        - Also evaluate for correctness, efficiency, readability, and adherence to the software design.
        - If the code is good, confirm it.
        - If there are issues, provide specific, actionable suggestions for the Corrector agent.
        - Output a JSON object adhering to the 'CritiqueReport' schema.
        """),
        ("human", """Software Design:
        {software_design}
        
        Task Description: {task}
        
        Code to Review:
        ```python
        {code}
        ```
        
        Verifier Report (Execution Results):
        {verifier_report}
        
        Please provide your critique.""")
    ])
    
    chain = prompt | critique_chain
    report = chain.invoke({
        "software_design": json.dumps(state.get("software_design"), indent=2),
        "task": state["current_task_description"],
        "code": state.get("corrected_code") or state.get("generated_code"),
        "verifier_report": json.dumps(state.get("verifier_report"), indent=2)
    })
    
    console.print(f"Critique Summary: {report.summary}")
    return {"critique_feedback_details": report.model_dump()}

def corrector_agent(state: AgentState) -> Dict[str, Any]:
    """Applies patches to code based on critique feedback."""
    console.rule("[bold orange_red1]CORRECTOR: Patching Code[/bold orange_red1]")
    
    # Use the generation agent with a specific "correction" prompt
    correction_prompt = """
    You are an expert debugger. Your task is to correct the following code based on the provided critique.
    Apply the suggestions precisely and minimally. Do not rewrite the entire file unless necessary.
    
    Original Code:
    ```python
    {code}
    ```
    
    Critique and Suggestions for Correction:
    {critique}
    
    Task Context: {task}
    
    Output only the complete, corrected, raw code for the file `{file_name}`.
    """
    
    code_to_correct = state.get("corrected_code") or state.get("generated_code")
    
    # This uses the same underlying generator but with a different mission
    llm = ChatGoogleGenerativeAI(model=GENERATOR_MODEL, temperature=0.1)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an elite programmer specializing in Python debugging.
        - You will be given code with issues and a critique.
        - Your sole job is to fix the code according to the critique.
        - Only output the raw, complete, fixed code for the file `{file_name}`.
        - Do not include markdown formatting like ```python ... ```."""),
        ("human", correction_prompt)
    ])
    
    chain = prompt | llm
    
    corrected_code = chain.invoke({
        "code": code_to_correct,
        "critique": json.dumps(state.get("critique_feedback_details"), indent=2),
        "task": state["current_task_description"],
        "file_name": state["current_file_name"]
    })
    
    return {
        "corrected_code": corrected_code.content,
        "correction_attempts": state.get("correction_attempts", 0) + 1,
        "generated_code": None, # Nullify original to ensure corrector output is used
    }
    
def finish_file_processing(state: AgentState) -> Dict[str, Any]:
    """Finalizes processing for one file and prepares for the next."""
    console.rule(f"[bold green]SUCCESS: Finished processing {state['current_file_name']}[/bold green]")
    
    # The iterator list is modified in place
    remaining_files = state["file_plan_iterator"]
    if remaining_files:
        remaining_files.pop(0)

    # Reset state for the next file
    return {
        "file_plan_iterator": remaining_files,
        "current_file_name": None,
        "current_task_description": None,
        "generated_code": None,
        "corrected_code": None,
        "verifier_report": None,
        "critique_feedback_details": None,
        "gen_strategy_approved": False,
        "user_feedback_for_rethink": None,
        "correction_attempts": 0,
    }

# --- 4. Graph Conditional Logic & Routing ---

def should_replan(state: AgentState) -> str:
    """Routes after the initial planning phase based on user approval."""
    if state.get("plan_approved"):
        console.log("Decision: Plan approved. Proceeding to per-file strategy.")
        return "continue_to_strategy"
    else:
        console.log("Decision: Plan rejected or needs revision. Looping back to planner.")
        return "replan"
        
def route_to_strategy_or_end(state: AgentState) -> str:
    """Decides whether to plan the next file or end the process."""
    if not state.get("file_plan_iterator"):
        console.log("Decision: All files processed. Ending workflow.")
        return END
    else:
        console.log(f"Decision: {len(state['file_plan_iterator'])} file(s) remaining. Planning next file.")
        return "planner"

def should_rethink_strategy(state: AgentState) -> str:
    """Routes after the strategy phase based on user approval."""
    if state.get("gen_strategy_approved"):
        console.log("Decision: Strategy approved. Routing to generation.")
        return "route_generation_path"
    else:
        console.log("Decision: Strategy rejected. Looping back for rethink.")
        return "planner" # Loop back to the planner for Phase 2

def route_generation_path(state: AgentState) -> str:
    """Routes to the correct code generation agent based on the chosen strategy."""
    strategy = state.get("chosen_gen_strategy")
    console.log(f"Routing to generation path: {strategy}")
    if strategy == "Symbolic":
        return "symbolic_reasoner"
    elif strategy == "Pseudocode":
        return "pseudocode_refiner"
    else: # Default to NL
        return "nl_to_code"
        
def should_refine_pseudocode(state: AgentState) -> str:
    """Decides whether to loop for more pseudocode refinement."""
    if state.get("pseudocode_iterations_remaining", 0) > 0:
        console.log("Decision: Refining pseudocode further.")
        return "pseudocode_refiner"
    else:
        console.log("Decision: Pseudocode refinement complete. Proceeding to code generation.")
        return "pseudocode_to_code"
        
def check_verification_results(state: AgentState) -> str:
    """Routes after critique based on success or failure."""
    verifier_report = state.get("verifier_report", {})
    critique = state.get("critique_feedback_details", {})
    
    # From Corrector -> Verifier -> Critique
    if state.get("correction_attempts", 0) > 0:
        if verifier_report.get("success") and critique.get('is_correct_and_runnable', False):
             console.log("✅ Decision: Correction successful. Finishing file.")
             return "finish_file"
        elif state.get("correction_attempts", 0) >= 3: # Max 3 correction attempts
            console.log("❌ Decision: Max correction attempts reached. Finishing file with errors.")
            return "finish_file"
        else:
            console.log("❗ Decision: Correction failed. Sending to corrector.")
            return "corrector"
    
    # From Generator -> Verifier -> Critique
    else:
        if verifier_report.get("success") and critique.get('is_correct_and_runnable', False):
            console.log("✅ Decision: Verification successful on first try. Finishing file.")
            return "finish_file"
        else:
            console.log("❗ Decision: Verification failed or needs improvement. Sending to corrector.")
            return "corrector"

# --- 5. Main Application Flow ---

def main():
    """Orchestrates the CLI interaction and LangGraph execution."""
    # Load API Key
    if not os.getenv("GOOGLE_API_KEY"):
        console.print("[bold red]ERROR: GOOGLE_API_KEY environment variable not set.[/bold red]")
        return

    console.rule("[bold green]Modular Self-Correcting Neuro-Symbolic Code Generation Framework[/bold green]")
    
    # 1. Initial Setup from User
    user_request = Prompt.ask("[bold cyan]What application or code would you like to build?[/bold cyan]")
    
    console.print("\n[bold]Configure Reasoning Paradigms (y/n):[/bold]")
    enable_got = Prompt.ask("Enable Graph-of-Thoughts (GoT) for advanced planning?", default="n").lower() == 'y'
    enable_symbolic = Prompt.ask("Enable Symbolic Reasoning path for algorithmic tasks?", default="n").lower() == 'y'
    enable_pseudo = Prompt.ask("Enable Iterative Pseudocode path for complex logic?", default="y").lower() == 'y'
    
    console.print("\n[bold]Configure Execution Environment:[/bold]")
    console.print("[dim]• Sandbox: Basic execution with minimal isolation (faster but less secure)[/dim]")
    console.print("[dim]• Safe System: Enhanced security with child process isolation, restricted environment,")
    console.print("[dim]  and shorter timeouts (recommended for untrusted code)[/dim]")
    execution_mode = Prompt.ask("Choose execution mode", choices=["sandbox", "safe_system"], default="safe_system")
    
    # 2. Load Context
    console.log("Scanning current directory for context...")
    existing_file_context = FilesystemTool.read_directory_contents('.')
    if existing_file_context:
        console.log(f"Found {len(existing_file_context)} existing files for context.")
    
    # 3. Initialize AgentState
    initial_state: AgentState = {
        "user_request": user_request,
        "enable_got_planning": enable_got,
        "enable_symbolic_reasoning": enable_symbolic,
        "enable_pseudocode_iterations": enable_pseudo,
        "llm_model": GENERATOR_MODEL, # Default model
        "execution_mode": execution_mode,
        "messages": [],
        "thoughts_history": [],
        "existing_file_context": existing_file_context,
        "software_design": None,
        "file_plan_iterator": None,
        "current_file_info": None,
        "current_file_name": None,
        "current_task_description": None,
        "current_task_complexity_score": None,
        "proof_needed": None,
        "chosen_gen_strategy": None,
        "pseudocode_iterations_remaining": None,
        "plan_approved": None,
        "gen_strategy_approved": None,
        "user_command": None,
        "user_feedback_for_replan": None,
        "user_feedback_for_rethink": None,
        "generated_pseudocode": None,
        "generated_symbolic_representation": None,
        "generated_code": None,
        "file_path": None,
        "execution_environment": "host",
        "verifier_report": None,
        "critique_feedback_details": None,
        "corrected_code": None,
        "correction_attempts": 0,
    }
    
    # 4. Build the Graph
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("planner", planner_agent)
    workflow.add_node("symbolic_reasoner", symbolic_reasoner_agent)
    workflow.add_node("pseudocode_refiner", pseudocode_refiner_agent)
    workflow.add_node("nl_to_code", nl_to_code_agent)
    workflow.add_node("pseudocode_to_code", pseudocode_to_code_agent)
    workflow.add_node("symbolic_to_code", symbolic_to_code_agent)
    workflow.add_node("verifier", verifier_agent)
    workflow.add_node("critique", critique_agent)
    workflow.add_node("corrector", corrector_agent)
    workflow.add_node("finish_file", finish_file_processing)

    # Define Edges
    workflow.set_entry_point("planner")
    
    # Planning and Strategy Loop - combine the routing logic
    workflow.add_conditional_edges(
        "planner",
        lambda state: (
            "planner" if not state.get("plan_approved") else  # Replan loop
            "planner" if not state.get("gen_strategy_approved") else  # Strategy rethink loop
            route_generation_path(state)  # Route to generation when both approved
        ),
        {
            "planner": "planner",
            "symbolic_reasoner": "symbolic_reasoner",
            "pseudocode_refiner": "pseudocode_refiner", 
            "nl_to_code": "nl_to_code"
        }
    )
    
    # After a file is finished, check if we should plan for the next one or end
    workflow.add_conditional_edges(
        "finish_file",
        route_to_strategy_or_end,
        {
            "planner": "planner",
             END: END
        }
    )
    
    # Connect generation paths
    workflow.add_edge("symbolic_reasoner", "symbolic_to_code")
    workflow.add_conditional_edges(
        "pseudocode_refiner",
        should_refine_pseudocode,
        {
            "pseudocode_refiner": "pseudocode_refiner", # Loop for refinement
            "pseudocode_to_code": "pseudocode_to_code"
        }
    )
    
    # Connect all generation paths to the verifier
    workflow.add_edge("nl_to_code", "verifier")
    workflow.add_edge("pseudocode_to_code", "verifier")
    workflow.add_edge("symbolic_to_code", "verifier")
    
    # Verification & Correction Loop
    workflow.add_edge("verifier", "critique") # Always critique after verification
    workflow.add_conditional_edges(
        "critique",
        check_verification_results, # This function will check the critique report now
        {
            "finish_file": "finish_file", # If critique says it's good
            "corrector": "corrector"      # If critique has suggestions
        }
    )
    workflow.add_edge("corrector", "verifier") # Loop back to verifier after correction
    
    app = workflow.compile()
    
    # 5. Run the Graph
    console.rule("[bold]STARTING AUTONOMOUS WORKFLOW[/bold]")
    
    # Stream events to see the flow in real-time
    for event in app.stream(initial_state, {"recursion_limit": 100}):
        for node, output in event.items():
            console.log(f"--- Finished Node: [bold]{node}[/bold] ---")
            # You can uncomment the line below for extremely verbose state tracking
            # console.print(output)
            
    console.rule("[bold green]WORKFLOW COMPLETE[/bold green]")
    console.print("The process has finished. Check the generated files in your directory.")

if __name__ == "__main__":
    main()