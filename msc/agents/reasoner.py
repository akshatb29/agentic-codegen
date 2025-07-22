# msc/agents/reasoner.py
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console

from msc.state import AgentState
from msc.tools import load_prompt

console = Console()
GENERATOR_MODEL = "gemini-1.5-flash-latest"

def symbolic_reasoner_agent(state: AgentState) -> Dict[str, Any]:
    console.rule("[bold blue]REASONER: Symbolic Representation[/bold blue]")
    llm = ChatGoogleGenerativeAI(model=GENERATOR_MODEL, temperature=0.1)
    prompt = ChatPromptTemplate.from_template(load_prompt("symbolic_reasoner.txt"))
    chain = prompt | llm
    representation = chain.invoke({"task": state["current_task_description"]})
    return {"generated_symbolic_representation": representation.content}
    
def pseudocode_refiner_agent(state: AgentState) -> Dict[str, Any]:
    console.rule("[bold blue]REASONER: Pseudocode Refinement[/bold blue]")
    llm = ChatGoogleGenerativeAI(model=GENERATOR_MODEL, temperature=0.2)
    iterations_remaining = state.get("pseudocode_iterations_remaining", 2)
    console.log(f"Pseudocode Iteration: {3 - iterations_remaining}/2")
    
    prompt = ChatPromptTemplate.from_template(load_prompt("pseudocode_refiner.txt"))
    chain = prompt | llm
    pseudocode = chain.invoke({
        "task": state["current_task_description"],
        "refinement_instructions": f"Refine this pseudocode:\n{state.get('generated_pseudocode')}" if state.get('generated_pseudocode') else ""
    })
    
    return {
        "generated_pseudocode": pseudocode.content,
        "pseudocode_iterations_remaining": iterations_remaining - 1
    }