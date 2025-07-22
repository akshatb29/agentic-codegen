# msc/agents/planner.py
import json
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console

from msc.state import AgentState, SoftwareDesign, GenerationStrategy, BrainstormedDesigns, EvaluatedDesigns
from msc.tools import user_confirmation_tool, FilesystemTool, load_prompt

console = Console()
PLANNER_MODEL = "gemini-1.5-flash-latest"

def _run_graph_of_thoughts(llm, state: AgentState) -> SoftwareDesign:
    """A more explicit implementation of the GoT process with robust error handling."""
    print("🤔 [Planner] Engaging Graph-of-Thoughts process...")
    
    try:
        # 1. Generation: Brainstorm multiple designs
        print("  1. Generating multiple design options...")
        gen_prompt = ChatPromptTemplate.from_template("Brainstorm 2-3 different software designs for this request:\n\n{request}")
        gen_chain = gen_prompt | llm.with_structured_output(BrainstormedDesigns)
        brainstormed = gen_chain.invoke({"request": state["user_request"]})
        
        # Validate brainstormed designs
        if not brainstormed or not brainstormed.designs or len(brainstormed.designs) == 0:
            print("  ⚠️  Warning: No designs generated, falling back to direct approach")
            return _fallback_direct_design(llm, state)
        
        # 2. Evaluation: Critique the generated designs
        print("  2. Evaluating the pros and cons of each design...")
        eval_prompt = ChatPromptTemplate.from_template("Evaluate the following software designs. For each, list pros and cons, give a score from 1-10, and then state which index is the best choice.\n\n{designs}")
        eval_chain = eval_prompt | llm.with_structured_output(EvaluatedDesigns)
        evaluated = eval_chain.invoke({"designs": brainstormed.model_dump_json(indent=2)})
        
        # Validate evaluation
        if (not evaluated or 
            evaluated.best_design_index is None or 
            evaluated.best_design_index < 0 or 
            evaluated.best_design_index >= len(brainstormed.designs)):
            print("  ⚠️  Warning: Invalid evaluation, using first design")
            return brainstormed.designs[0]
        
        print(f"  3. Synthesizing the best design (Option #{evaluated.best_design_index})...")
        # 3. Synthesis: Return the best design
        return brainstormed.designs[evaluated.best_design_index]
        
    except Exception as e:
        print(f"  ❌ Error in GoT process: {e}")
        print("  ⚠️  Falling back to direct design generation...")
        return _fallback_direct_design(llm, state)

def _fallback_direct_design(llm, state: AgentState) -> SoftwareDesign:
    """Fallback method for direct design generation when GoT fails."""
    try:
        prompt = ChatPromptTemplate.from_template(load_prompt("planner_design.txt"))
        chain = prompt | llm.with_structured_output(SoftwareDesign)
        design = chain.invoke({
            "user_request": state["user_request"],
            "existing_file_context": json.dumps(state["existing_file_context"], indent=2),
            "thoughts_history": [], "got_instructions": "", "replan_instructions": ""
        })
        return design
    except Exception as e:
        console.log(f"  [red]Error in fallback design: {e}[/red]")
        # Ultimate fallback - create a minimal valid design
        return SoftwareDesign(
            thought="Fallback design due to errors in planning process",
            files=[{
                "name": "main.py",
                "purpose": f"Main implementation for: {state['user_request'][:100]}...",
                "dependencies": [],
                "key_functions": ["main"]
            }]
        )

def planner_agent(state: AgentState) -> Dict[str, Any]:
    llm = ChatGoogleGenerativeAI(model=PLANNER_MODEL, temperature=0.3)
    
    # PHASE 1: Overall Software Design
    if not state.get("plan_approved"):
        console.rule("[bold magenta]PLANNER: Designing Software Architecture[/bold magenta]")
        
        try:
            if state["enable_got_planning"]:
                design = _run_graph_of_thoughts(llm, state)
            else: # Direct generation
                design = _fallback_direct_design(llm, state)
            
            # Validate the design
            if not design or not design.files or len(design.files) == 0:
                console.log("[red]Error: Invalid design generated[/red]")
                return {"plan_approved": False, "user_feedback_for_replan": "Invalid design generated"}
            
            console.print("[bold green]Proposed Software Design:[/bold green]")
            console.print_json(data=design.model_dump())
            
            user_command = user_confirmation_tool("Do you approve this software design?")
            
            # More robust response handling
            if user_command.lower() in ["y", "yes", "true", "1"]:
                return {
                    "software_design": design.model_dump(), 
                    "plan_approved": True, 
                    "file_plan_iterator": design.files.copy()
                }
            elif user_command.lower() in ["n", "no", "false", "0"]:
                return {
                    "plan_approved": False, 
                    "user_feedback_for_replan": "User rejected the design"
                }
            else:
                # Treat any other response as feedback
                return {
                    "plan_approved": False, 
                    "user_feedback_for_replan": user_command
                }
        except Exception as e:
            console.log(f"[red]Error in planning phase: {e}[/red]")
            return {"plan_approved": False, "user_feedback_for_replan": f"Planning error: {str(e)}"}

    # PHASE 2: Per-File Generation Strategy
    console.rule("[bold magenta]PLANNER: Choosing Generation Strategy[/bold magenta]")
    
    if not state.get("file_plan_iterator") or len(state["file_plan_iterator"]) == 0:
        console.log("[yellow]No more files to process[/yellow]")
        return {"plan_approved": False}
    
    current_file_info = state["file_plan_iterator"][0]
    file_name = current_file_info.get('name', 'unknown_file.py')
    console.log(f"📝 Planning generation for file: [bold cyan]{file_name}[/bold cyan]")
    
    try:
        prompt = ChatPromptTemplate.from_template(load_prompt("planner_strategy.txt"))
        chain = prompt | llm.with_structured_output(GenerationStrategy)
        
        strategy = chain.invoke({
            "software_design": json.dumps(state["software_design"], indent=2),
            "file_name": file_name,
            "task_description": current_file_info.get('purpose', 'No description available'),
            "enabled_strategies": "NL, Pseudocode, Symbolic", 
            "rethink_instructions": ""
        })
        
        console.print(f"[bold green]Proposed Strategy for '{file_name}':[/bold green] [bold yellow]{strategy.chosen_gen_strategy}[/bold yellow]")
        user_command = user_confirmation_tool("Do you approve this generation strategy?")

        if user_command.lower() in ["y", "yes", "true", "1"]:
            return {
                "current_file_info": current_file_info, 
                "current_file_name": file_name,
                "current_task_description": current_file_info.get('purpose', 'No description'), 
                "file_path": file_name,
                "chosen_gen_strategy": strategy.chosen_gen_strategy, 
                "gen_strategy_approved": True,
                "correction_attempts": 0,
            }
        elif user_command.lower() in ["n", "no", "false", "0"]:
            return {
                "gen_strategy_approved": False, 
                "user_feedback_for_rethink": "User rejected the strategy"
            }
        else:
            return {
                "gen_strategy_approved": False, 
                "user_feedback_for_rethink": user_command
            }
    except Exception as e:
        console.log(f"[red]Error in strategy selection: {e}[/red]")
        # Fallback to NL strategy
        return {
            "current_file_info": current_file_info, 
            "current_file_name": file_name,
            "current_task_description": current_file_info.get('purpose', 'No description'), 
            "file_path": file_name,
            "chosen_gen_strategy": "NL", 
            "gen_strategy_approved": True,
            "correction_attempts": 0,
        }