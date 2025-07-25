# msc/agents/planner.py
import json
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console

from msc.state import AgentState, SoftwareDesign, GenerationStrategy, BrainstormedDesigns, EvaluatedDesigns
from msc.tools import user_confirmation_tool, FilesystemTool, load_prompt, get_llm
# Note: docker_manager imported locally to avoid circular imports

console = Console()

def _analyze_tech_stack_and_architecture(llm, state: AgentState) -> Dict[str, Any]:
    """Analyze user request and propose tech stack with detailed reasoning"""
    print("🔍 [Planner] Analyzing tech stack requirements...")
    
    tech_analysis_prompt = ChatPromptTemplate.from_template("""
    Analyze this user request and propose the optimal tech stack and architecture:
    
    User Request: {user_request}
    
    Please provide:
    1. Recommended tech stack (libraries, frameworks, tools)
    2. Architecture pattern (MVC, microservices, etc.)
    3. Key dependencies and why they're needed
    4. Potential challenges and solutions
    5. Alternative approaches and trade-offs
    
    Be specific about:
    - Required Python packages
    - System dependencies
    - Development vs production requirements
    - Performance considerations
    - Security implications
    
    Format your response as JSON with clear reasoning for each choice.
    """)
    
    try:
        response = llm.invoke(tech_analysis_prompt.format(user_request=state["user_request"]))
        
        # Extract tech stack info for Docker preparation
        tech_content = response.content
        
        # Detect application type for Docker
        app_type = "general"
        tech_lower = tech_content.lower()
        
        if any(gui_term in tech_lower for gui_term in ['tkinter', 'pyqt', 'kivy', 'gui', 'desktop']):
            app_type = "gui_app"
        elif any(web_term in tech_lower for web_term in ['flask', 'django', 'fastapi', 'streamlit', 'web']):
            app_type = "web_app"  
        elif any(data_term in tech_lower for data_term in ['pandas', 'numpy', 'matplotlib', 'data science', 'analysis']):
            app_type = "data_analysis"
        elif any(ml_term in tech_lower for ml_term in ['sklearn', 'tensorflow', 'torch', 'machine learning', 'ai']):
            app_type = "machine_learning"
        
        return {
            "tech_analysis": tech_content,
            "detected_app_type": app_type,
            "docker_prep_needed": True
        }
        
    except Exception as e:
        print(f"⚠️ Tech analysis failed: {e}")
        return {
            "tech_analysis": "Basic Python application",
            "detected_app_type": "general", 
            "docker_prep_needed": True
        }

def _prepare_docker_environment_async(state: AgentState, app_type: str):
    """Prepare Docker environment in background while planning continues"""
    try:
        # Import Docker components directly - new architecture
        from msc.agents.docker_agent import docker_agent
        from msc.tools.docker_tools import docker_executor
        
        print(f"🐳 [Background] Preparing Docker environment for {app_type} application...")
        
        # Create a preliminary spec for async building
        sample_code = f"# Sample {app_type} application\nprint('Hello World')"
        spec = docker_agent.analyze_code_and_generate_spec(sample_code, state["user_request"])
        
        # Start async build using the new tools
        image_name = f"msc-session-{docker_executor.session_id}"
        docker_executor.build_image_async(spec.dockerfile_content, image_name)
        
        # Store spec for later use
        state["prepared_docker_spec"] = spec.model_dump() if hasattr(spec, 'model_dump') else spec.__dict__
        
    except Exception as e:
        print(f"⚠️ Background Docker preparation failed: {e}")
        state["prepared_docker_spec"] = None

def _create_project_structure_plan(llm, state: AgentState) -> Dict[str, Any]:
    """Create comprehensive project structure including directories and requirements"""
    print("📁 [Planner] Creating project structure and requirements...")
    
    structure_prompt = ChatPromptTemplate.from_template("""
    Based on this user request and tech stack analysis, create a comprehensive project structure:
    
    User Request: {user_request}
    Tech Analysis: {tech_analysis}
    App Type: {app_type}
    
    Please provide:
    1. Project name (short, descriptive, kebab-case)
    2. Directory structure (folders and their purposes)
    3. Required Python packages (with versions if specific ones needed)
    4. Required system dependencies (if any)
    5. Configuration files needed (requirements.txt, .env, etc.)
    6. Main entry point file
    7. Testing strategy
    
    Create a logical project structure that follows Python best practices.
    Consider:
    - Separation of concerns
    - Testability
    - Maintainability
    - Deployment requirements
    
    Format as JSON with clear explanations for each decision.
    """)
    
    try:
        response = llm.invoke(structure_prompt.format(
            user_request=state["user_request"],
            tech_analysis=state.get("tech_analysis", "General Python application"),
            app_type=state.get("detected_app_type", "general")
        ))
        
        return {
            "project_structure": response.content,
            "structure_planning_success": True
        }
        
    except Exception as e:
        print(f"⚠️ Project structure planning failed: {e}")
        # Fallback structure
        fallback_structure = {
            "project_name": "python-project",
            "directories": {
                "src": "Main source code",
                "tests": "Unit tests",
                "docs": "Documentation"
            },
            "requirements": ["requests", "rich"],
            "main_file": "main.py",
            "config_files": ["requirements.txt"]
        }
        
        return {
            "project_structure": json.dumps(fallback_structure, indent=2),
            "structure_planning_success": False
        }

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
    llm = get_llm("planner", temperature=0.3)
    
    # PHASE 0: Tech Stack Analysis (new phase)
    if not state.get("tech_stack_approved"):
        console.rule("[bold magenta]PLANNER: Analyzing Tech Stack & Architecture[/bold magenta]")
        
        try:
            tech_info = _analyze_tech_stack_and_architecture(llm, state)
            
            console.print("[bold green]📊 Tech Stack Analysis:[/bold green]")
            console.print(tech_info["tech_analysis"])
            console.print(f"\n[bold yellow]🎯 Detected App Type:[/bold yellow] {tech_info['detected_app_type']}")
            
            # Ask for confirmation on tech stack
            tech_approved = user_confirmation_tool(
                "Do you approve this tech stack and architecture analysis?"
            )
            
            if tech_approved.lower() in ["y", "yes", "true", "1"]:
                # Start Docker preparation in background
                if tech_info["docker_prep_needed"]:
                    _prepare_docker_environment_async(state, tech_info["detected_app_type"])
                
                return {
                    "tech_stack_approved": True,
                    "tech_analysis": tech_info["tech_analysis"],
                    "detected_app_type": tech_info["detected_app_type"],
                    "docker_prep_started": tech_info["docker_prep_needed"]
                }
            elif tech_approved.lower() in ["n", "no", "false", "0"]:
                return {
                    "tech_stack_approved": False,
                    "user_feedback_for_tech_replan": "User rejected the tech stack"
                }
            else:
                return {
                    "tech_stack_approved": False,
                    "user_feedback_for_tech_replan": tech_approved
                }
                
        except Exception as e:
            console.log(f"[red]Error in tech stack analysis: {e}[/red]")
            # Fallback: assume general app and continue
            _prepare_docker_environment_async(state, "general")
            return {
                "tech_stack_approved": True,
                "tech_analysis": "General Python application (fallback)",
                "detected_app_type": "general",
                "docker_prep_started": True
            }
    
    # PHASE 1: Project Structure Planning (new phase)
    if not state.get("project_structure_approved"):
        console.rule("[bold magenta]PLANNER: Project Structure & Requirements[/bold magenta]")
        
        try:
            structure_info = _create_project_structure_plan(llm, state)
            
            console.print("[bold green]📁 Proposed Project Structure:[/bold green]")
            console.print(structure_info["project_structure"])
            
            # Ask for confirmation on project structure
            structure_approved = user_confirmation_tool(
                "Do you approve this project structure and requirements?"
            )
            
            if structure_approved.lower() in ["y", "yes", "true", "1"]:
                # Parse project structure to extract key info
                try:
                    import json
                    struct_data = json.loads(structure_info["project_structure"])
                    project_name = struct_data.get("project_name", "untitled-project")
                    main_file = struct_data.get("main_file", "main.py")
                    requirements = struct_data.get("requirements", [])
                except:
                    project_name = "untitled-project"
                    main_file = "main.py"
                    requirements = []
                
                return {
                    "project_structure_approved": True,
                    "project_structure": structure_info["project_structure"],
                    "project_name": project_name,
                    "main_file": main_file,
                    "requirements": requirements
                }
            elif structure_approved.lower() in ["n", "no", "false", "0"]:
                return {
                    "project_structure_approved": False,
                    "user_feedback_for_structure_replan": "User rejected the project structure"
                }
            else:
                return {
                    "project_structure_approved": False,
                    "user_feedback_for_structure_replan": structure_approved
                }
                
        except Exception as e:
            console.log(f"[red]Error in project structure planning: {e}[/red]")
            # Fallback: use simple structure
            return {
                "project_structure_approved": True,
                "project_structure": "Simple Python project structure",
                "project_name": "python-project",
                "main_file": "main.py",
                "requirements": []
            }
    
    # PHASE 2: Overall Software Design
    if not state.get("plan_approved"):
        console.rule("[bold magenta]PLANNER: Software Architecture Design[/bold magenta]")
        
        try:
            if state["enable_got_planning"]:
                design = _run_graph_of_thoughts(llm, state)
            else: # Direct generation
                design = _fallback_direct_design(llm, state)
            
            # Validate the design
            if not design or not design.files or len(design.files) == 0:
                console.log("[red]Error: Invalid design generated[/red]")
                return {"plan_approved": False, "user_feedback_for_replan": "Invalid design generated"}
            
            console.print("[bold green]🏗️ Proposed Software Design:[/bold green]")
            console.print_json(data=design.model_dump())
            
            # Enhanced confirmation with reasoning
            console.print(f"\n[bold cyan]💭 Architecture Reasoning:[/bold cyan]")
            console.print(f"• Project: {state.get('project_name', 'untitled')}")
            console.print(f"• Files to create: {len(design.files)}")
            console.print(f"• Main components: {', '.join([f['name'] for f in design.files])}")
            console.print(f"• Tech stack: {state.get('detected_app_type', 'general')}")
            console.print(f"• Entry point: {state.get('main_file', 'main.py')}")
            
            user_command = user_confirmation_tool("Do you approve this software architecture design?")
            
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

    # PHASE 3: Per-File Generation Strategy
    console.rule("[bold magenta]PLANNER: File Generation Strategy[/bold magenta]")
    
    if not state.get("file_plan_iterator") or len(state["file_plan_iterator"]) == 0:
        console.log("[yellow]All files planned - ready for testing phase[/yellow]")
        return {"files_completed": True, "ready_for_testing": True}
    
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
        
        console.print(f"[bold green]🎯 Proposed Strategy for '{file_name}':[/bold green] [bold yellow]{strategy.chosen_gen_strategy}[/bold yellow]")
        console.print(f"[bold cyan]💭 Strategy Reasoning:[/bold cyan] {getattr(strategy, 'reasoning', 'Standard approach')}")
        
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