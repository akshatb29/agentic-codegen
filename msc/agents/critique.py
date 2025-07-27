# msc/agents/critique.py
import json
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console

from msc.state import AgentState, CritiqueReport
from msc.tools import load_prompt, get_llm

console = Console()

def critique_agent(state: AgentState) -> Dict[str, Any]:
    console.rule("[bold red]CRITIQUE: Reviewing Code[/bold red]")
    llm = get_llm("critique")
    
    # Try to use improved critique prompt first, fallback to original
    try:
        prompt = ChatPromptTemplate.from_template(load_prompt("critique_fixed.txt"))
    except:
        prompt = ChatPromptTemplate.from_template(load_prompt("critique.txt"))
    
    code_to_review = state.get("corrected_code") or state.get("generated_code")
    verifier_report = state.get("verifier_report", {})
    
    # Enhanced critique with execution context
    try:
        response = llm.invoke(prompt.format(
            software_design=json.dumps(state.get("software_design"), indent=2),
            task=state.get("current_task_description", "No task description"),
            code=code_to_review,
            verifier_report=json.dumps(verifier_report, indent=2),
            execution_status="Success" if verifier_report.get("success") else "Failed",
            language=state.get("language", "python")
        ))
        
        # Extract summary from response content
        critique_content = response.content
        console.print(f"Critique Summary: {critique_content[:200]}...", style="yellow")
        
        return {
            "critique_feedback_details": {
                "summary": critique_content,
                "execution_success": verifier_report.get("success", False),
                "has_recommendations": "recommend" in critique_content.lower()
            }
        }
        
    except Exception as e:
        console.print(f"⚠️ Critique failed: {e}", style="red")
        return {
            "critique_feedback_details": {
                "summary": "Basic code review: Focus on fixing execution errors",
                "execution_success": verifier_report.get("success", False),
                "has_recommendations": True
            }
        }