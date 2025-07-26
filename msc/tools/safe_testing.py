# msc/tools/safe_testing.py
"""
Safe testing framework with user confirmation and process isolation
"""
import subprocess
import tempfile
import os
from pathlib import Path
from rich.console import Console
from rich.prompt import Confirm
from typing import Dict, Any, Optional

console = Console()

class SafeTester:
    """Safe testing framework that asks for permission before execution"""
    
    def __init__(self):
        self.console = Console()
        
    def propose_test(self, description: str, command: str, code: str = None, 
                    expected_outcome: str = None) -> bool:
        """Propose a test to the user and ask for confirmation"""
        console.print("\n" + "="*60, style="blue")
        console.print("🧪 PROPOSED TEST", style="bold blue")
        console.print("="*60, style="blue")
        
        console.print(f"📝 Description: {description}", style="cyan")
        console.print(f"🔧 Command: {command}", style="yellow")
        
        if code:
            console.print("📄 Code to test:", style="green")
            console.print(code[:200] + "..." if len(code) > 200 else code, style="dim")
            
        if expected_outcome:
            console.print(f"🎯 Expected: {expected_outcome}", style="magenta")
            
        console.print("\n⚠️  This will execute code in a separate process", style="red")
        
        return Confirm.ask("🤔 Proceed with this test?", default=False)
    
    def safe_execute_python(self, code: str, description: str = "Python test", 
                          timeout: int = 30) -> Dict[str, Any]:
        """Safely execute Python code in isolated subprocess"""
        
        if not self.propose_test(
            description=description,
            command=f"python3 (in subprocess with {timeout}s timeout)",
            code=code,
            expected_outcome="Safe execution in isolated environment"
        ):
            return {"success": False, "error": "User declined test", "skipped": True}
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            console.print(f"🚀 Executing in isolated subprocess...", style="blue")
            
            result = subprocess.run(
                ['python3', temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tempfile.gettempdir()  # Run in temp directory
            )
            
            os.unlink(temp_file)  # Clean up
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            os.unlink(temp_file)
            return {"success": False, "error": f"Test timed out after {timeout}s"}
        except Exception as e:
            if 'temp_file' in locals():
                os.unlink(temp_file)
            return {"success": False, "error": f"Test failed: {e}"}
    
    def safe_execute_command(self, command: str, description: str = "Shell command",
                           timeout: int = 30, cwd: str = None) -> Dict[str, Any]:
        """Safely execute shell command in subprocess"""
        
        if not self.propose_test(
            description=description,
            command=command,
            expected_outcome="Safe command execution"
        ):
            return {"success": False, "error": "User declined test", "skipped": True}
        
        try:
            console.print(f"🚀 Executing command in subprocess...", style="blue")
            
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd or tempfile.gettempdir()
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Command timed out after {timeout}s"}
        except Exception as e:
            return {"success": False, "error": f"Command failed: {e}"}

# Global instance
safe_tester = SafeTester()
