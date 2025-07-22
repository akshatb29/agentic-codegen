# msc/agents/code_generator.py
import json
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console

from msc.state import AgentState
from msc.tools import load_prompt

console = Console()
GENERATOR_MODEL = "gemini-1.5-flash-latest"

def _generate(state: AgentState, context_key: str, context_value: Any) -> Dict[str, Any]:
    console.rule(f"[bold green]GENERATOR: {context_key} -> Code[/bold green]")
    llm = ChatGoogleGenerativeAI(model=GENERATOR_MODEL, temperature=0.2)
    prompt = ChatPromptTemplate.from_template(load_prompt("code_generator.txt"))
    chain = prompt | llm
    
    code = chain.invoke({
        "file_name": state["current_file_name"], "task": state["current_task_description"],
        "software_design": json.dumps(state.get("software_design"), indent=2),
        "existing_file_context": json.dumps(state.get("existing_file_context"), indent=2),
        "generation_context": f"{context_key}:\n{context_value}"
    })
    return {"generated_code": code.content}

def nl_to_code_agent(state: AgentState) -> Dict[str, Any]:
    return _generate(state, "Task Description", state["current_task_description"])

def pseudocode_to_code_agent(state: AgentState) -> Dict[str, Any]:
    return _generate(state, "Pseudocode", state["generated_pseudocode"])

def symbolic_to_code_agent(state: AgentState) -> Dict[str, Any]:
    return _generate(state, "Symbolic Representation", state["generated_symbolic_representation"])