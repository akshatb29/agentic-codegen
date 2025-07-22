# msc/agents/critique.py
import json
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console

from msc.state import AgentState, CritiqueReport
from msc.tools import load_prompt

console = Console()
PLANNER_MODEL = "gemini-1.5-flash-latest"

def critique_agent(state: AgentState) -> Dict[str, Any]:
    console.rule("[bold red]CRITIQUE: Reviewing Code[/bold red]")
    llm = ChatGoogleGenerativeAI(model=PLANNER_MODEL, temperature=0.1)
    prompt = ChatPromptTemplate.from_template(load_prompt("critique.txt"))
    chain = prompt | llm.with_structured_output(CritiqueReport)
    
    code_to_review = state.get("corrected_code") or state.get("generated_code")
    report = chain.invoke({
        "software_design": json.dumps(state.get("software_design"), indent=2),
        "task": state["current_task_description"],
        "code": code_to_review,
        "verifier_report": json.dumps(state.get("verifier_report"), indent=2)
    })
    console.print(f"Critique Summary: {report.summary}")
    return {"critique_feedback_details": report.model_dump()}