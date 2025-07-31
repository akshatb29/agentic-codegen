# msc/agents/agentic_planner.py
"""
More Agentic Planner: Agents make autonomous decisions and communicate with each other
"""
import json
from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from dataclasses import dataclass
from enum import Enum
import uuid

from msc.state import AgentState, SoftwareDesign, GenerationStrategy, BrainstormedDesigns, EvaluatedDesigns
from msc.tools import user_confirmation_tool, FilesystemTool, load_prompt, get_llm
from msc.agents.dynamic_problem_solver import solve_planning_error

class AgentDecision(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    DELEGATE = "delegate"
    ESCALATE = "escalate"

@dataclass
class AgentMessage:
    from_agent: str
    to_agent: str
    message_type: str
    content: dict
    timestamp: float
    message_id: str = ""
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())[:8]

class AutonomousAgent:
    """Base class for autonomous agents that can make decisions and communicate"""
    
    def __init__(self, agent_id: str, role: str, autonomy_level: float = 0.7):
        self.agent_id = agent_id
        self.role = role
        self.autonomy_level = autonomy_level  # 0.0 = always ask user, 1.0 = fully autonomous
        self.message_history: List[AgentMessage] = []
        self.decision_history: List[dict] = []
        self.performance_metrics = {"success_rate": 0.5, "user_satisfaction": 0.5}
        
    def should_ask_user(self, decision_confidence: float) -> bool:
        """Decide whether to ask user based on confidence and autonomy level"""
        return decision_confidence < (1.0 - self.autonomy_level)
    
    def learn_from_feedback(self, feedback: str, outcome: str):
        """Update agent's performance based on feedback"""
        if outcome == "success":
            self.performance_metrics["success_rate"] = min(1.0, self.performance_metrics["success_rate"] + 0.1)
        else:
            self.performance_metrics["success_rate"] = max(0.0, self.performance_metrics["success_rate"] - 0.05)
            
        # Adjust autonomy based on performance
        if self.performance_metrics["success_rate"] > 0.8:
            self.autonomy_level = min(0.9, self.autonomy_level + 0.05)
        elif self.performance_metrics["success_rate"] < 0.3:
            self.autonomy_level = max(0.3, self.autonomy_level - 0.1)
    
    def send_message(self, to_agent: str, message_type: str, content: dict) -> AgentMessage:
        """Send message to another agent"""
        message = AgentMessage(
            from_agent=self.agent_id,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            timestamp=__import__('time').time()
        )
        self.message_history.append(message)
        return message

class AgenticPlanner(AutonomousAgent):
    """Autonomous planner agent with decision-making capabilities"""
    
    def __init__(self):
        super().__init__("planner", "Software Architecture Planning", autonomy_level=0.6)
        self._llm = None  # Lazy initialization
    
    @property
    def llm(self):
        """Lazy initialization of LLM"""
        if self._llm is None:
            self._llm = get_llm("planner", temperature=0.3)
        return self._llm
    
    def autonomous_design_decision(self, designs: List[SoftwareDesign], user_request: str) -> dict:
        """Agent makes autonomous design decisions based on criteria"""
        print(f"[Planner Agent]: Evaluating {len(designs)} design option(s) autonomously...")
        
        # Agent's evaluation criteria
        criteria = {
            "complexity": 0.3,
            "maintainability": 0.3, 
            "user_requirements_match": 0.4
        }
        
        best_design = None
        best_score = 0
        confidence = 0
        
        for i, design in enumerate(designs):
            # Simulate agent's evaluation process
            score = self._evaluate_design(design, user_request, criteria)
            print(f"    - Design Option {i+1}: Calculated Score = {score:.2f}/10.0")
            
            if score > best_score:
                best_score = score
                best_design = design
                confidence = min(score / 10.0, 1.0)  # Convert to confidence
        
        decision_type = AgentDecision.APPROVE if confidence > 0.7 else AgentDecision.ESCALATE
        
        return {
            "decision": decision_type,
            "chosen_design": best_design,
            "confidence": confidence,
            "reasoning": f"Selected best design with score {best_score:.2f} based on weighted criteria",
            "needs_user_input": self.should_ask_user(confidence)
        }
    
    def _evaluate_design(self, design: SoftwareDesign, user_request: str, criteria: dict) -> float:
        """Internal method to evaluate a design against criteria"""
        score = 5.0  # Base score
        
        if len(design.files) <= 3 and "simple" in user_request.lower():
            score += criteria["complexity"] * 3
        elif len(design.files) > 5:
            score -= criteria["complexity"] * 2
            
        if all(file.get("purpose") for file in design.files):
            score += criteria["maintainability"] * 2
            
        request_keywords = user_request.lower().split()
        design_text = design.thought.lower() + " ".join([f.get("purpose", "") for f in design.files])
        matches = sum(1 for keyword in request_keywords if keyword in design_text)
        score += criteria["user_requirements_match"] * matches
        
        return min(score, 10.0)
    
    def decompose_task_autonomously(self, user_request: str, complexity_threshold: int = 7) -> dict:
        """Agent decides whether to break down the task further"""
        print(f"[Planner Agent]: Analyzing user request for task complexity...")
        
        complexity_indicators = [
            len(user_request.split()) > 20,
            "and" in user_request.lower(),
            any(word in user_request.lower() for word in ["database", "api", "gui", "web", "server"]),
            any(word in user_request.lower() for word in ["complex", "advanced", "enterprise", "system"])
        ]
        
        complexity_score = sum(complexity_indicators) * 2 + len(user_request.split()) // 10
        
        if complexity_score >= complexity_threshold:
            print(f"    - Task Complexity Score: {complexity_score}/10. Recommending decomposition.")
            return {
                "should_decompose": True,
                "complexity_score": complexity_score,
                "suggested_subtasks": self._suggest_subtasks(user_request),
                "reasoning": "High complexity detected, breaking into subtasks will improve success rate"
            }
        else:
            print(f"    - Task Complexity Score: {complexity_score}/10. Proceeding with a single task approach.")
            return {
                "should_decompose": False,
                "complexity_score": complexity_score,
                "reasoning": "Task complexity is manageable as a single unit"
            }
    
    def _suggest_subtasks(self, user_request: str) -> List[str]:
        """Suggest how to break down a complex task"""
        subtasks = []
        
        if "gui" in user_request.lower() or "interface" in user_request.lower():
            subtasks.append("Design and implement user interface")
        if "database" in user_request.lower() or "data" in user_request.lower():
            subtasks.append("Design and implement data storage layer")
        if "api" in user_request.lower():
            subtasks.append("Implement API endpoints and handlers")
            
        if not subtasks:
            subtasks = [
                "Core functionality implementation",
                "User interface and interaction",
                "Testing and validation"
            ]
        return subtasks

def agentic_planner_agent(state: AgentState) -> Dict[str, Any]:
    """Enhanced planner with agentic capabilities and dynamic agent spawning"""
    planner = AgenticPlanner()
    
    planning_attempts = state.get("planning_attempts", 0)
    max_planning_attempts = 3
    
    if planning_attempts >= max_planning_attempts:
        print(f"WARNING: Maximum planning attempts ({max_planning_attempts}) reached. Spawning a specialized agent to resolve the issue.")
        return solve_planning_error(state)
    
    # PHASE 1: Autonomous Task Analysis
    if not state.get("plan_approved"):
        print("\n" + "=" * 60)
        print("--- AGENTIC PLANNER: AUTONOMOUS PLANNING PHASE ---")
        print("=" * 60)
        print(f"INFO: Planning attempt {planning_attempts + 1} of {max_planning_attempts}.")
        
        task_analysis = planner.decompose_task_autonomously(state["user_request"])
        
        try:
            if state["enable_got_planning"]:
                design = _run_agentic_graph_of_thoughts(planner, state)
            else:
                print("INFO: Using direct design approach (Graph-of-Thoughts disabled).")
                design = _fallback_direct_design(planner.llm, state)
            
            decision_result = planner.autonomous_design_decision([design], state["user_request"])
            
            print("\n[Planner Agent] Decision Summary:")
            print(f"    - Final Decision: {decision_result['decision'].value.upper()}")
            print(f"    - Confidence Level: {decision_result['confidence']:.2f}")
            print(f"    - Reasoning: {decision_result['reasoning']}")
            
            if decision_result["needs_user_input"]:
                print("\n[User Interaction] Agent has low confidence and requires human input...")
                print("\nProposed Software Design:")
                print(json.dumps(design.model_dump(), indent=2))
                user_command = user_confirmation_tool("Agent recommends this design but requires your approval to proceed. Approve?")
            else:
                print(f"\n[Autonomous Action] Agent is proceeding with {decision_result['confidence']:.0%} confidence.")
                user_command = "yes"
            
            if user_command.lower() in ["y", "yes", "true", "1"]:
                if not decision_result["needs_user_input"]:
                    planner.learn_from_feedback("autonomous_approval", "success")
                
                return {
                    "software_design": design.model_dump(), 
                    "plan_approved": True, 
                    "file_plan_iterator": design.files.copy(),
                    "task_analysis": task_analysis,
                    "agent_decision_log": decision_result,
                    "planning_attempts": 0
                }
            else:
                planner.learn_from_feedback(user_command, "failure")
                return {
                    "plan_approved": False, 
                    "user_feedback_for_replan": user_command,
                    "agent_decision_log": decision_result,
                    "planning_attempts": planning_attempts + 1
                }
                
        except Exception as e:
            print(f"ERROR: An exception occurred during the agentic planning phase: {e}")
            return {
                "plan_approved": False, 
                "user_feedback_for_replan": f"Planning error: {str(e)}",
                "planning_attempts": planning_attempts + 1
            }

    # PHASE 2: Autonomous Strategy Selection
    print("\n" + "=" * 60)
    print("--- AGENTIC PLANNER: AUTONOMOUS STRATEGY SELECTION PHASE ---")
    print("=" * 60)
    
    if not state.get("file_plan_iterator") or len(state["file_plan_iterator"]) == 0:
        print("INFO: No more files to process in the current plan.")
        return {"plan_approved": False}
    
    current_file_info = state["file_plan_iterator"][0]
    file_name = current_file_info.get('name', 'unknown_file.py')
    print(f"INFO: Analyzing next file in plan: {file_name}")
    
    autonomous_strategy = planner.select_generation_strategy_autonomously(
        current_file_info, state["software_design"]
    )
    
    print("\n[Planner Agent] Strategy Selection Summary:")
    print(f"    - Chosen Strategy: '{autonomous_strategy['strategy']}'")
    print(f"    - Confidence Level: {autonomous_strategy['confidence']:.2f}")
    print(f"    - Reasoning: {autonomous_strategy['reasoning']}")
    
    if autonomous_strategy["needs_user_input"]:
        print("\n[User Interaction] Agent has low confidence and requires human input for strategy...")
        user_command = user_confirmation_tool(f"Agent suggests using the '{autonomous_strategy['strategy']}' strategy. Approve?")
    else:
        print(f"\n[Autonomous Action] Agent is proceeding with the '{autonomous_strategy['strategy']}' strategy.")
        user_command = "yes"
        
    if user_command.lower() in ["y", "yes", "true", "1"]:
        return {
            "current_file_info": current_file_info, 
            "current_file_name": file_name,
            "current_task_description": current_file_info.get('purpose', 'No description'), 
            "file_path": file_name,
            "chosen_gen_strategy": autonomous_strategy["strategy"], 
            "gen_strategy_approved": True,
            "correction_attempts": 0,
            "autonomous_strategy_log": autonomous_strategy
        }
    else:
        return {
            "gen_strategy_approved": False, 
            "user_feedback_for_rethink": user_command,
            "autonomous_strategy_log": autonomous_strategy
        }

def _run_agentic_graph_of_thoughts(planner: AgenticPlanner, state: AgentState) -> SoftwareDesign:
    """Graph-of-Thoughts with agent communication and autonomous decisions"""
    print("\n--- Starting Agentic Graph-of-Thoughts Process ---")
    
    try:
        # 1. Generation Agent
        print("Step 1: [Generation Agent] Creating multiple design options...")
        gen_prompt = ChatPromptTemplate.from_template("""
        As an autonomous design generation agent, create 2-3 distinct software designs for this request.
        Consider different architectural approaches and complexity levels.
        Request: {request}
        """)
        gen_chain = gen_prompt | planner.llm.with_structured_output(BrainstormedDesigns)
        brainstormed = gen_chain.invoke({"request": state["user_request"]})
        
        if not brainstormed or not brainstormed.designs:
            print("WARNING: Generation Agent failed to produce designs. Using fallback.")
            return _fallback_direct_design(planner.llm, state)
        
        # 2. Evaluation Agent
        print("Step 2: [Evaluation Agent] Analyzing the quality of generated designs...")
        eval_prompt = ChatPromptTemplate.from_template("""
        As an autonomous evaluation agent, analyze these software designs:
        - Rate each design from 1-10 on: complexity, maintainability, user requirements match
        - Identify the best design and explain why
        - Be decisive in your recommendation
        
        Designs: {designs}
        """)
        eval_chain = eval_prompt | planner.llm.with_structured_output(EvaluatedDesigns)
        evaluated = eval_chain.invoke({"designs": brainstormed.model_dump_json(indent=2)})
        
        # 3. Decision Agent (the planner itself)
        if evaluated and 0 <= evaluated.best_design_index < len(brainstormed.designs):
            selected_design = brainstormed.designs[evaluated.best_design_index]
            print(f"Step 3: [Decision Agent] Selected design #{evaluated.best_design_index + 1} based on evaluation.")
            print("--- Graph-of-Thoughts Process Complete ---")
            return selected_design
        else:
            print("WARNING: Decision Agent could not use evaluation. Defaulting to the first design.")
            print("--- Graph-of-Thoughts Process Complete ---")
            return brainstormed.designs[0]
            
    except Exception as e:
        print(f"ERROR: An exception occurred during Graph-of-Thoughts process: {e}")
        print("--- Graph-of-Thoughts Process Aborted ---")
        return _fallback_direct_design(planner.llm, state)

def _fallback_direct_design(llm, state: AgentState) -> SoftwareDesign:
    """Enhanced fallback with better defaults"""
    print("INFO: Executing fallback direct design generation.")
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
        print(f"ERROR: Fallback design generation also failed: {e}")
        print("INFO: Creating an intelligent emergency fallback design.")
        request_lower = state['user_request'].lower()
        
        if any(word in request_lower for word in ['calc', 'calculator', 'math']):
            files = [{"name": "calculator.py", "purpose": "Calculator implementation with basic operations"}]
        elif any(word in request_lower for word in ['web', 'website', 'html']):
            files = [
                {"name": "app.py", "purpose": "Web application main file"},
                {"name": "templates/index.html", "purpose": "Main webpage template"}
            ]
        else:
            files = [{"name": "main.py", "purpose": f"Main implementation for: {state['user_request'][:50]}..."}]
        
        return SoftwareDesign(
            thought=f"Intelligent fallback design for: {state['user_request'][:100]}",
            files=files
        )

# Add the strategy selection method to the AgenticPlanner class
def select_generation_strategy_autonomously(self, file_info: dict, software_design: dict) -> dict:
    """Agent selects generation strategy based on file characteristics"""
    file_name = file_info.get('name', '')
    purpose = file_info.get('purpose', '')
    
    if any(word in purpose.lower() for word in ['complex', 'algorithm', 'logic']):
        strategy = "Pseudocode"
        confidence = 0.8
        reasoning = "Complex logic detected; pseudocode will help structure the implementation."
    elif any(word in purpose.lower() for word in ['proof', 'verify', 'mathematical']):
        strategy = "Symbolic"
        confidence = 0.7
        reasoning = "Mathematical/verification tasks benefit from symbolic reasoning."
    else:
        strategy = "NL"
        confidence = 0.6
        reasoning = "Standard natural language approach is suitable for general implementation."
    
    return {
        "strategy": strategy,
        "confidence": confidence,
        "reasoning": reasoning,
        "needs_user_input": confidence < (1.0 - self.autonomy_level)
    }

# Monkey patch the method to the class
AgenticPlanner.select_generation_strategy_autonomously = select_generation_strategy_autonomously
