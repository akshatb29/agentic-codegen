# msc/tools/file_selector.py
try:
    import inquirer
    INQUIRER_AVAILABLE = True
except ImportError:
    INQUIRER_AVAILABLE = False
    
from typing import Dict, List
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm, IntPrompt
from pathlib import Path

from .context_analyzer import ContextAnalyzer

console = Console()

class FileSelector:
    """Enhanced file selection interface with smart suggestions and checkbox selection."""
    
    @staticmethod
    def select_files_interactive(available_files: Dict[str, str], user_request: str = "") -> Dict[str, str]:
        """
        Interactive file selection with smart suggestions and multiple selection modes.
        
        Args:
            available_files: Dict of {file_path: file_content}
            user_request: User's request for context analysis
            
        Returns:
            Dict of selected {file_path: file_content}
        """
        if not available_files:
            console.log("No relevant files found in the current directory.")
            return {}
        
        console.rule("[bold blue]📁 File Context Selection[/bold blue]")
        
        # Quick no-context option
        if Confirm.ask("🚀 [bold yellow]Skip file context?[/bold yellow] (Proceed with no existing file context)", default=False):
            console.log("✅ [bold green]Proceeding with no file context.[/bold green]")
            return {}
        
        # Get smart suggestions if user request is provided
        suggestions = {}
        if user_request.strip():
            suggestions = ContextAnalyzer.get_smart_suggestions(user_request, available_files)
            FileSelector._display_smart_suggestions(suggestions)
        
        # Selection mode choice
        if INQUIRER_AVAILABLE:
            selection_modes = [
                "📋 Multi-select with checkboxes (recommended)",
                "🎯 Use smart suggestions only", 
                "🔢 Numbered selection (fallback)",
                "👆 Select one by one (classic mode)"
            ]
            
            try:
                mode_choice = inquirer.list_input(
                    "Choose selection mode:",
                    choices=selection_modes
                )
            except Exception as e:
                console.print(f"[dim]Inquirer error: {e}. Using fallback mode.[/dim]")
                mode_choice = "🔢 Numbered selection (fallback)"
        else:
            console.print("[dim]📋 Inquirer not available. Using alternative selection modes.[/dim]")
            selection_modes = [
                "🎯 Use smart suggestions only",
                "🔢 Numbered selection",
                "👆 Select one by one (classic mode)"
            ]
            
            console.print("\n[bold]Available selection modes:[/bold]")
            for i, mode in enumerate(selection_modes, 1):
                console.print(f"  {i}. {mode}")
            
            choice_idx = IntPrompt.ask("Choose mode (1-3)", default=1) - 1
            mode_choice = selection_modes[choice_idx] if 0 <= choice_idx < len(selection_modes) else selection_modes[0]
        
        if "smart suggestions" in mode_choice:
            return FileSelector._select_suggested_files(available_files, suggestions)
        elif "checkboxes" in mode_choice and INQUIRER_AVAILABLE:
            return FileSelector._select_files_checkbox(available_files, suggestions)
        elif "numbered" in mode_choice.lower():
            return FileSelector._select_files_numbered(available_files, suggestions)
        else:
            return FileSelector._select_files_classic(available_files)
    
    @staticmethod
    def _display_smart_suggestions(suggestions: Dict[str, List[str]]) -> None:
        """Display smart suggestions in a nice table format."""
        if not any(suggestions.values()):
            return
            
        console.print("\n[bold blue]🧠 Smart Analysis Results:[/bold blue]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Priority", style="dim", width=12)
        table.add_column("Suggested Files", style="cyan")
        
        for priority, files in suggestions.items():
            if files:
                priority_display = priority.replace('_', ' ').title()
                files_display = '\n'.join([f"• {Path(f).name}" for f in files])
                table.add_row(priority_display, files_display)
        
        console.print(table)
        console.print()
    
    @staticmethod
    def _select_suggested_files(available_files: Dict[str, str], suggestions: Dict[str, List[str]]) -> Dict[str, str]:
        """Auto-select based on smart suggestions with user confirmation."""
        suggested_files = []
        for priority_files in suggestions.values():
            suggested_files.extend(priority_files)
        
        if not suggested_files:
            console.log("❌ No smart suggestions found. Falling back to checkbox selection.")
            return FileSelector._select_files_checkbox(available_files, {})
        
        console.print(f"\n[bold green]✨ Smart selection suggests {len(suggested_files)} files:[/bold green]")
        for file_path in suggested_files:
            console.print(f"  • [cyan]{Path(file_path).name}[/cyan] ([dim]{file_path}[/dim])")
        
        if Confirm.ask("\nUse these smart suggestions?", default=True):
            selected = {f: available_files[f] for f in suggested_files if f in available_files}
            console.log(f"✅ [bold green]Selected {len(selected)} files using smart suggestions.[/bold green]")
            return selected
        else:
            return FileSelector._select_files_checkbox(available_files, suggestions)
    
    @staticmethod
    def _select_files_checkbox(available_files: Dict[str, str], suggestions: Dict[str, List[str]]) -> Dict[str, str]:
        """Multi-select files using checkboxes with pre-selection of suggestions."""
        suggested_files = []
        for priority_files in suggestions.values():
            suggested_files.extend(priority_files)
        
        # Prepare choices for inquirer
        choices = []
        for file_path in sorted(available_files.keys()):
            file_name = Path(file_path).name
            # Show relative path and file size info
            display_name = f"{file_name} ({file_path})"
            choices.append((display_name, file_path))
        
        # Pre-select suggested files
        default_selected = [choice[1] for choice in choices if choice[1] in suggested_files]
        
        console.print("\n[bold]Select files to include in context:[/bold]")
        console.print("[dim]💡 Use SPACE to select/deselect, ENTER to confirm[/dim]")
        
        try:
            selected_paths = inquirer.checkbox(
                "Files:",
                choices=[choice[0] for choice in choices],
                default=[choice[0] for choice in choices if choice[1] in default_selected]
            )
            
            # Map back to file paths
            path_mapping = {choice[0]: choice[1] for choice in choices}
            selected_file_paths = [path_mapping[display_name] for display_name in selected_paths]
            
            selected_context = {path: available_files[path] for path in selected_file_paths}
            
            if selected_context:
                console.log(f"✅ [bold green]Selected {len(selected_context)} file(s) for context.[/bold green]")
            else:
                console.log("ℹ️  No files selected for context.")
                
            return selected_context
            
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Selection cancelled. Proceeding with no context.[/bold yellow]")
            return {}
    
    @staticmethod
    def _select_files_numbered(available_files: Dict[str, str], suggestions: Dict[str, List[str]]) -> Dict[str, str]:
        """Multi-select files using numbered interface (fallback for inquirer)."""
        suggested_files = []
        for priority_files in suggestions.values():
            suggested_files.extend(priority_files)
        
        # Display numbered list
        file_list = list(available_files.keys())
        console.print("\n[bold]📝 Available Files:[/bold]")
        console.print("[dim]💡 Files marked with ⭐ are smart suggestions[/dim]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=4)
        table.add_column("File", style="cyan")
        table.add_column("Path", style="dim")
        table.add_column("Suggested", style="green", width=10)
        
        for i, file_path in enumerate(file_list, 1):
            file_name = Path(file_path).name
            is_suggested = "⭐" if file_path in suggested_files else ""
            table.add_row(str(i), file_name, file_path, is_suggested)
        
        console.print(table)
        
        # Get selection
        console.print(f"\n[bold]Select files by number:[/bold]")
        console.print("[dim]Enter numbers separated by spaces (e.g., '1 3 5') or 'all' for all files, or 'suggested' for smart suggestions[/dim]")
        
        while True:
            try:
                selection = input("Selection: ").strip().lower()
                
                if selection == 'all':
                    selected_context = available_files.copy()
                    break
                elif selection == 'suggested':
                    selected_context = {f: available_files[f] for f in suggested_files if f in available_files}
                    break
                elif not selection:
                    console.print("ℹ️  No selection made. Proceeding with no context.")
                    return {}
                else:
                    # Parse number selection
                    numbers = [int(n.strip()) for n in selection.split() if n.strip().isdigit()]
                    selected_paths = [file_list[n-1] for n in numbers if 1 <= n <= len(file_list)]
                    selected_context = {path: available_files[path] for path in selected_paths}
                    break
                    
            except (ValueError, IndexError):
                console.print("[bold red]❌ Invalid selection. Please try again.[/bold red]")
                continue
        
        if selected_context:
            console.log(f"✅ [bold green]Selected {len(selected_context)} file(s) for context.[/bold green]")
        else:
            console.log("ℹ️  No files selected for context.")
        
        return selected_context
    
    @staticmethod
    def _select_files_classic(available_files: Dict[str, str]) -> Dict[str, str]:
        """Classic one-by-one file selection (original method)."""
        selected_context = {}
        console.print("\n[bold]Select files individually:[/bold]")
        
        for file_path, content in available_files.items():
            file_name = Path(file_path).name
            if Confirm.ask(f"Include [cyan]'{file_name}'[/cyan] ([dim]{file_path}[/dim])?", default=False):
                selected_context[file_path] = content
        
        if selected_context:
            console.log(f"✅ [bold green]Selected {len(selected_context)} file(s) for context.[/bold green]")
        else:
            console.log("ℹ️  No files selected for context.")
        
        return selected_context
