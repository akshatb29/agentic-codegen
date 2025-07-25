# msc/agents/corrector.py
import json
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console

from msc.state import AgentState
from msc.tools import load_prompt, get_llm

console = Console()

def corrector_agent(state: AgentState) -> Dict[str, Any]:
    console.rule("[bold orange_red1]CORRECTOR: Patching Code[/bold orange_red1]")
    llm = get_llm("corrector")
    prompt = ChatPromptTemplate.from_template(load_prompt("corrector.txt"))
    chain = prompt | llm
    
    code_to_correct = state.get("corrected_code") or state.get("generated_code")
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