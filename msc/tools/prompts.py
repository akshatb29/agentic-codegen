# msc/tools/prompts.py
from pathlib import Path
from rich.console import Console

console = Console()
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

def load_prompt(file_name: str) -> str:
    try:
        with open(PROMPTS_DIR / file_name, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        console.log(f"❌ [Prompt Loader] Error: Prompt file '{file_name}' not found.")
        return ""