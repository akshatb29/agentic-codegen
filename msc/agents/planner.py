# msc/agents/planner.py
import json
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console

from msc.state import AgentState, SoftwareDesign, GenerationStrategy, BrainstormedDesigns, EvaluatedDesigns
from msc.tools import user_confirmation_tool, FilesystemTool, load_prompt, get_llm, search_tech_stack_recommendations, search_architecture_patterns
# Note: docker_manager imported locally to avoid circular imports

console = Console()

def _analyze_tech_stack_and_architecture(llm, state: AgentState) -> Dict[str, Any]:
    """Analyze user request and propose tech stack with detailed reasoning and web search"""
    print("🔍 [Planner] Analyzing tech stack requirements...")
    
    user_request = state["user_request"]
    
    # Ask user if they want web search for tech stack recommendations
    enable_search = user_confirmation_tool(
        "🌐 Would you like me to search for current best practices and tech stack recommendations online? "
    )
    
    search_results = {}
    web_insights = ""
    
    if enable_search.lower() in ["y", "yes", "true", "1"]:
        console.print("🔍 Searching for tech stack recommendations...", style="cyan")
        try:
            # Search for tech stack recommendations
            tech_search = search_tech_stack_recommendations(user_request, "python")
            arch_search = search_architecture_patterns(user_request, "python")
            
            search_results = {
                "tech_recommendations": tech_search.get("recommendations", {}),
                "architecture_patterns": arch_search.get("patterns", {})
            }
            
            # Format web insights for the LLM
            if tech_search.get("search_performed"):
                recommendations = tech_search.get("recommendations", {})
                web_insights = f"""
Web Search Insights:
- Recommended frameworks: {', '.join(recommendations.get('frameworks', []))}
- Suggested libraries: {', '.join(recommendations.get('libraries', []))}
- Architecture patterns: {', '.join(arch_search.get('patterns', {}).get('recommended_patterns', []))}
- Search reasoning: {'; '.join(recommendations.get('reasoning', []))}
"""
            
            console.print("✅ Web search completed", style="green")
        except Exception as e:
            console.print(f"⚠️ Web search failed: {e}", style="yellow")
            web_insights = "Web search was attempted but failed."
    
    tech_analysis_prompt = ChatPromptTemplate.from_template("""
    Analyze this user request and propose the optimal tech stack and architecture with detailed reasoning:
    
    User Request: {user_request}
    
    Web Search Insights (if available): {web_insights}
    
    Please provide a comprehensive analysis with:
    1. Recommended tech stack (libraries, frameworks, tools) with specific versions when important
    2. Architecture pattern (MVC, microservices, layered, etc.) with detailed explanation
    3. Key dependencies and why each is essential for this project
    4. Potential challenges and proposed solutions
    5. Alternative approaches with pros/cons comparison
    6. Scalability considerations
    7. Security implications and best practices
    8. Testing strategy recommendations
    
    Be specific about:
    - Required Python packages with justification
    - System dependencies (if any)
    - Development vs production requirements
    - Performance optimization opportunities
    - Deployment considerations
    - Integration points
    
    IMPORTANT: If the web search provided insights, integrate them into your analysis and 
    explain whether you agree or disagree with the recommendations, providing clear reasoning.
    
    If web search suggests alternatives to your initial recommendations, analyze the trade-offs 
    and explain your final decision.
    
    Format your response as a structured analysis with clear sections and reasoning for each choice.
    """)
    
    try:
        response = llm.invoke(tech_analysis_prompt.format(
            user_request=user_request,
            web_insights=web_insights
        ))
        
        # Extract tech stack info for Docker preparation
        tech_content = response.content
        
        # Detect application type for Docker (enhanced detection)
        app_type = "general"
        tech_lower = tech_content.lower()
        request_lower = user_request.lower()
        
        # More sophisticated app type detection - FIXED ORDER
        if any(ml_term in tech_lower or ml_term in request_lower 
                for ml_term in ['alexnet', 'alex', 'neural', 'network', 'cnn', 'conv', 'sklearn', 'tensorflow', 'torch', 'machine learning', 'ai', 'deep learning']):
            app_type = "machine_learning"
        elif any(data_term in tech_lower or data_term in request_lower 
                for data_term in ['pandas', 'numpy', 'matplotlib', 'data science', 'analysis', 'visualization']):
            app_type = "data_analysis"
        elif any(web_term in tech_lower or web_term in request_lower 
                for web_term in ['flask', 'django', 'fastapi', 'streamlit', 'web', 'api', 'server']):
            app_type = "web_app"  
        elif any(gui_term in tech_lower or gui_term in request_lower 
               for gui_term in ['tkinter', 'pyqt', 'kivy', 'gui', 'desktop', 'window']):
            app_type = "gui_app"
        elif any(game_term in tech_lower or game_term in request_lower 
                for game_term in ['pygame', 'game', 'arcade']):
            app_type = "game"
        
        return {
            "tech_analysis": tech_content,
            "detected_app_type": app_type,
            "docker_prep_needed": True,
            "search_results": search_results,
            "web_search_enabled": enable_search.lower() in ["y", "yes", "true", "1"]
        }
        
    except Exception as e:
        print(f"⚠️ Tech analysis failed: {e}")
        return {
            "tech_analysis": "Basic Python application",
            "detected_app_type": "general", 
            "docker_prep_needed": True,
            "search_results": {},
            "web_search_enabled": False
        }

def _handle_user_feedback_and_reason(llm, user_feedback: str, planner_decision: str, context: str) -> Dict[str, Any]:
    """Handle user feedback by reasoning about the disagreement and providing options"""
    console.print(f"🤔 [Planner] User provided feedback: {user_feedback}", style="yellow")
    console.print("💭 [Planner] Analyzing disagreement and reasoning...", style="cyan")
    
    reasoning_prompt = ChatPromptTemplate.from_template("""
    The user has provided feedback that suggests they disagree with my recommendation. 
    I need to analyze their feedback and either:
    1. Defend my original decision with stronger reasoning
    2. Acknowledge their valid points and propose a compromise
    3. Accept their feedback and revise my recommendation
    
    Original Context: {context}
    My Original Recommendation: {planner_decision}
    User Feedback: {user_feedback}
    
    Please analyze this disagreement and provide:
    1. What specific aspect the user likely disagrees with
    2. Valid points in the user's feedback (if any)
    3. Strengths of my original recommendation that still hold
    4. Potential compromises or middle-ground solutions
    5. A revised recommendation that addresses user concerns while maintaining technical soundness
    6. Clear reasoning for why this revised approach is better
    
    Format your response as a structured analysis that I can present to the user for final decision.
    """)
    
    try:
        response = llm.invoke(reasoning_prompt.format(
            context=context,
            planner_decision=planner_decision,
            user_feedback=user_feedback
        ))
        
        reasoning_content = response.content
        
        console.print("[bold green]🧠 Planner's Reasoning Analysis:[/bold green]")
        console.print(reasoning_content)
        
        # Present options to user
        console.print(f"\n[bold cyan]🎯 Based on your feedback, I see the following options:[/bold cyan]")
        console.print("1. [bold]Stick with original[/bold] - Keep my original recommendation with additional justification")
        console.print("2. [bold]Compromise solution[/bold] - Find middle ground addressing your concerns") 
        console.print("3. [bold]Accept your approach[/bold] - Adopt your suggested direction")
        console.print("4. [bold]Completely re-analyze[/bold] - Start fresh with new approach")
        
        user_choice = user_confirmation_tool(
            "Which approach would you prefer? (1/2/3/4 or explain your preference)"
        )
        
        return {
            "reasoning_analysis": reasoning_content,
            "user_choice": user_choice,
            "feedback_processed": True
        }
        
    except Exception as e:
        console.print(f"❌ Error in reasoning analysis: {e}", style="red")
        return {
            "reasoning_analysis": "Failed to analyze disagreement",
            "user_choice": user_feedback,
            "feedback_processed": False
        }

def _prepare_docker_environment_async(state: AgentState, app_type: str):
    """Docker environment prep - handled automatically by execution.py now"""
    # With new simplified architecture, Docker setup is handled automatically
    # by execution.py when code execution is needed - no async prep required
    print(f"🐳 [Background] Docker environment will be prepared automatically for {app_type} application")
    pass
    """Docker environment prep - handled automatically by execution.py now"""
    # With new simplified architecture, Docker setup is handled automatically
    # by execution.py when code execution is needed - no async prep required
    print(f"🐳 [Background] Docker environment will be prepared automatically for {app_type} application")
    pass

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
    
    response = llm.invoke(structure_prompt.format(
        user_request=state["user_request"],
        tech_analysis=state.get("tech_analysis", "General Python application"),
        app_type=state.get("detected_app_type", "general")
    ))
    
    return {
        "project_structure": response.content,
        "structure_planning_success": True
    }

def _enhance_design_with_requirements(design: SoftwareDesign, state: AgentState) -> SoftwareDesign:
    """Enhance design with intelligent requirements prediction"""
    from msc.agents.requirements_installer import smart_requirements_predictor
    
    user_request = state.get("user_request", "")
    
    # Predict requirements based on user request
    predicted_requirements = smart_requirements_predictor(user_request, "python")
    
    # Detect language from files
    language = "python"  # default
    if design.files:
        for file_info in design.files:
            filename = file_info.get("name", "")
            if filename.endswith(".js") or filename.endswith(".ts"):
                language = "javascript"
                break
            elif filename.endswith(".go"):
                language = "go"
                break
    
    # Detect framework from user request and files
    framework = ""
    request_lower = user_request.lower()
    if "flask" in request_lower:
        framework = "flask"
        if "flask" not in predicted_requirements:
            predicted_requirements.append("flask")
    elif "django" in request_lower:
        framework = "django"
        if "django" not in predicted_requirements:
            predicted_requirements.append("django")
    elif "fastapi" in request_lower:
        framework = "fastapi"
        if "fastapi" not in predicted_requirements:
            predicted_requirements.extend(["fastapi", "uvicorn"])
    
    # Create enhanced design
    enhanced_design = SoftwareDesign(
        thought=design.thought + f" Enhanced with predicted requirements: {predicted_requirements}",
        files=design.files,
        requirements=predicted_requirements,
        language=language,
        framework=framework
    )
    
    console.print(f"📦 Predicted requirements: {predicted_requirements}", style="blue")
    console.print(f"🔍 Detected language: {language}, framework: {framework}", style="dim")
    
    return enhanced_design


def planner_agent(state: AgentState) -> Dict[str, Any]:
    llm = get_llm("planner", temperature=0.3)
    
    # PHASE 0: Tech Stack Analysis (enhanced with web search)
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
                # User provided specific feedback - analyze and reason
                feedback_analysis = _handle_user_feedback_and_reason(
                    llm, tech_approved, tech_info["tech_analysis"], 
                    f"Tech stack analysis for: {state['user_request']}"
                )
                
                if feedback_analysis["user_choice"].startswith("1"):
                    # User chose to stick with original
                    return {
                        "tech_stack_approved": True,
                        "tech_analysis": tech_info["tech_analysis"],
                        "detected_app_type": tech_info["detected_app_type"],
                        "docker_prep_started": True,
                        "planner_reasoning": feedback_analysis["reasoning_analysis"]
                    }
                elif feedback_analysis["user_choice"].startswith("4"):
                    # User wants complete re-analysis
                    return {
                        "tech_stack_approved": False,
                        "user_feedback_for_tech_replan": "Complete re-analysis requested",
                        "reset_analysis": True
                    }
                else:
                    # Need to incorporate feedback
                    return {
                        "tech_stack_approved": False,
                        "user_feedback_for_tech_replan": tech_approved,
                        "feedback_analysis": feedback_analysis
                    }
                
        except Exception as e:
            console.log(f"[red]Error in tech stack analysis: {e}[/red]")
            return {
                "tech_stack_approved": False,
                "error": f"Tech stack analysis failed: {e}"
            }
    
    # PHASE 1: Project Structure Planning (enhanced)
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
                # Handle user feedback with reasoning
                feedback_analysis = _handle_user_feedback_and_reason(
                    llm, structure_approved, structure_info["project_structure"],
                    f"Project structure for: {state['user_request']}"
                )
                
                if feedback_analysis["user_choice"].startswith("1"):
                    # Stick with original but show reasoning
                    try:
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
                        "requirements": requirements,
                        "planner_reasoning": feedback_analysis["reasoning_analysis"]
                    }
                else:
                    return {
                        "project_structure_approved": False,
                        "user_feedback_for_structure_replan": structure_approved,
                        "feedback_analysis": feedback_analysis
                    }
                
        except Exception as e:
            console.log(f"[red]Error in project structure planning: {e}[/red]")
            return {
                "project_structure_approved": False,
                "error": f"Structure planning failed: {e}"
            }
    
    # PHASE 2: Software Design - Create File Plan
    if not state.get("plan_approved"):
        console.rule("[bold magenta]PLANNER: Creating File Plan[/bold magenta]")
        
        try:
            # Generate software design
            prompt = ChatPromptTemplate.from_template(load_prompt("planner_design.txt"))
            chain = prompt | llm.with_structured_output(SoftwareDesign)
            
            design = chain.invoke({
                "user_request": state["user_request"],
                "existing_file_context": json.dumps(state.get("existing_file_context", {}), indent=2),
                "thoughts_history": [], 
                "got_instructions": "", 
                "replan_instructions": ""
            })
            
            # CRITICAL: Check if LLM provided proper filenames
            if not design.files or not design.files[0].get("name") or design.files[0].get("name") == "unknown_file.py":
                console.print("[yellow]⚠️ LLM didn't provide proper filenames, creating intelligent names[/yellow]")
                # Create proper filename based on project
                user_request = state.get("user_request", "")
                if "alexnet" in user_request.lower() or "alex" in user_request.lower():
                    main_filename = "alexnet_clone.py"
                elif "neural" in user_request.lower() or "network" in user_request.lower():
                    main_filename = "neural_network.py"
                elif "cnn" in user_request.lower() or "conv" in user_request.lower():
                    main_filename = "cnn_model.py" 
                else:
                    main_filename = state.get("main_file", "main.py")
                
                # Update the design with proper filename
                design.files[0]["name"] = main_filename
                console.print(f"✅ Fixed filename to: {main_filename}", style="green")
            
            # Enhance with requirements prediction
            enhanced_design = _enhance_design_with_requirements(design, state)
            
            console.print("[bold green]📋 Generated File Plan:[/bold green]")
            console.print(f"• Project: {state.get('project_name', 'untitled')}")
            console.print(f"• Files to create: {len(enhanced_design.files)}")
            console.print(f"• Requirements: {enhanced_design.requirements}")
            console.print(f"• Language: {enhanced_design.language}")
            
            # Ask user for approval
            plan_approved = user_confirmation_tool("Do you approve this file plan?")
            
            if plan_approved.lower() in ["y", "yes", "true", "1"]:
                return {
                    "software_design": enhanced_design.model_dump(), 
                    "plan_approved": True, 
                    "file_plan_iterator": enhanced_design.files.copy()
                }
            else:
                return {
                    "plan_approved": False,
                    "user_feedback_for_replan": plan_approved
                }
                
        except Exception as e:
            console.log(f"[red]Error in planning: {e}[/red]")
            return {"plan_approved": False, "user_feedback_for_replan": f"Planning failed: {e}"}

    # PHASE 3: Per-File Generation Strategy (same as before)
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
        
        # Offer strategy choices to user
        console.print("\n[bold cyan]📋 Available Generation Strategies:[/bold cyan]")
        console.print("1. [bold]NL[/bold] - Natural Language (direct code generation)")
        console.print("2. [bold]Pseudocode[/bold] - Structured pseudocode then code")
        console.print("3. [bold]Symbolic[/bold] - Symbolic reasoning for complex algorithms")
        console.print("4. [bold]Auto[/bold] - Use AI's recommendation")
        
        user_command = user_confirmation_tool("Choose strategy (1/2/3/4 or Y to accept AI choice):")

        # Handle user strategy choice
        chosen_strategy = strategy.chosen_gen_strategy  # Default to AI choice
        
        if user_command.strip() == "1":
            chosen_strategy = "NL"
        elif user_command.strip() == "2":
            chosen_strategy = "Pseudocode"
        elif user_command.strip() == "3":
            chosen_strategy = "Symbolic"
        elif user_command.strip() == "4" or user_command.lower() in ["y", "yes", "true"]:
            chosen_strategy = strategy.chosen_gen_strategy  # Use AI recommendation
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
        
        console.print(f"✅ [bold green]Selected Strategy:[/bold green] [bold yellow]{chosen_strategy}[/bold yellow]")
        
        return {
            "current_file_info": current_file_info, 
            "current_file_name": file_name,
            "current_task_description": current_file_info.get('purpose', 'No description'), 
            "file_path": file_name,
            "chosen_gen_strategy": chosen_strategy, 
            "gen_strategy_approved": True,
            "correction_attempts": 0,
        }
    except Exception as e:
        console.log(f"[red]Error in strategy selection: {e}[/red]")
        return {
            "gen_strategy_approved": False,
            "error": f"Strategy selection failed: {e}"
        }