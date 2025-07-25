#!/usr/bin/env python3
# msc/tools/llm_config_cli.py
"""
Command-line interface for managing LLM configurations
"""

import argparse
import yaml
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from msc.tools.llm_manager import llm_manager

console = Console()

def list_configurations():
    """List all current LLM configurations"""
    console.print("[bold blue]🔧 Current LLM Configurations[/bold blue]\n")
    
    # Agents table
    table = Table(title="Agent Configurations")
    table.add_column("Agent", style="cyan")
    table.add_column("Provider", style="green")
    table.add_column("Model", style="yellow")
    table.add_column("Temperature", style="magenta")
    table.add_column("Description", style="white")
    
    for agent_name, config in llm_manager.config['agents'].items():
        table.add_row(
            agent_name,
            config.get('provider', 'default'),
            config.get('model', 'default'),
            str(config.get('temperature', 'default')),
            config.get('description', '')
        )
    
    console.print(table)
    
    # Available providers
    console.print("\n[bold green]📦 Available Providers[/bold green]")
    for provider, details in llm_manager.config['providers'].items():
        console.print(f"• {provider}: {', '.join(details['models'])}")

def update_agent_config():
    """Interactive agent configuration update"""
    console.print("[bold yellow]🛠 Update Agent Configuration[/bold yellow]\n")
    
    # List agents
    agents = list(llm_manager.config['agents'].keys())
    console.print("Available agents:")
    for i, agent in enumerate(agents, 1):
        console.print(f"{i}. {agent}")
    
    # Get agent selection
    choice = Prompt.ask("Select agent number", choices=[str(i) for i in range(1, len(agents) + 1)])
    agent_name = agents[int(choice) - 1]
    
    current_config = llm_manager.get_agent_config(agent_name)
    console.print(f"\nCurrent config for [cyan]{agent_name}[/cyan]:")
    console.print(f"Provider: {current_config.provider}")
    console.print(f"Model: {current_config.model}")
    console.print(f"Temperature: {current_config.temperature}")
    
    # Get new configuration
    providers = list(llm_manager.config['providers'].keys())
    new_provider = Prompt.ask(
        "New provider", 
        choices=providers, 
        default=current_config.provider
    )
    
    models = llm_manager.config['providers'][new_provider]['models']
    new_model = Prompt.ask(
        "New model", 
        choices=models, 
        default=current_config.model if current_config.model in models else models[0]
    )
    
    new_temperature = Prompt.ask(
        "New temperature", 
        default=str(current_config.temperature)
    )
    
    # Confirm and update
    if Confirm.ask(f"Update {agent_name} configuration?"):
        llm_manager.update_agent_config(
            agent_name,
            provider=new_provider,
            model=new_model,
            temperature=float(new_temperature)
        )
        console.print(f"[green]✓[/green] Updated {agent_name} configuration")
        
        # Save to file
        save_config()

def save_config():
    """Save current configuration to file"""
    try:
        with open(llm_manager.config_path, 'w') as f:
            yaml.dump(llm_manager.config, f, default_flow_style=False, sort_keys=False)
        console.print(f"[green]✓[/green] Configuration saved to {llm_manager.config_path}")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to save configuration: {e}")

def test_connection():
    """Test LLM connections for all agents"""
    console.print("[bold blue]🔌 Testing LLM Connections[/bold blue]\n")
    
    results = Table()
    results.add_column("Agent", style="cyan")
    results.add_column("Provider", style="green")
    results.add_column("Model", style="yellow")
    results.add_column("Status", style="white")
    
    for agent_name in llm_manager.config['agents'].keys():
        try:
            llm = llm_manager.get_llm_for_agent(agent_name)
            # Try a simple invoke
            response = llm.invoke("Hello")
            status = "[green]✓ Connected[/green]"
        except Exception as e:
            status = f"[red]✗ {str(e)[:50]}...[/red]"
        
        config = llm_manager.get_agent_config(agent_name)
        results.add_row(
            agent_name,
            config.provider,
            config.model,
            status
        )
    
    console.print(results)

def add_provider():
    """Add a new provider configuration"""
    console.print("[bold green]➕ Add New Provider[/bold green]\n")
    
    provider_name = Prompt.ask("Provider name")
    class_path = Prompt.ask("Provider class path (e.g., 'langchain_openai.ChatOpenAI')")
    
    # Get models
    models = []
    console.print("Add models (enter empty line to finish):")
    while True:
        model = Prompt.ask("Model name", default="")
        if not model:
            break
        models.append(model)
    
    # Get default parameters
    temperature = float(Prompt.ask("Default temperature", default="0.1"))
    
    # Add to config
    llm_manager.config['providers'][provider_name] = {
        'class': class_path,
        'models': models,
        'default_params': {'temperature': temperature}
    }
    
    console.print(f"[green]✓[/green] Added provider {provider_name}")
    
    if Confirm.ask("Save configuration?"):
        save_config()

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="LLM Configuration Manager")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    subparsers.add_parser('list', help='List current configurations')
    
    # Update command
    subparsers.add_parser('update', help='Update agent configuration')
    
    # Test command
    subparsers.add_parser('test', help='Test LLM connections')
    
    # Add provider command
    subparsers.add_parser('add-provider', help='Add new provider')
    
    # Reload command
    subparsers.add_parser('reload', help='Reload configuration from file')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_configurations()
    elif args.command == 'update':
        update_agent_config()
    elif args.command == 'test':
        test_connection()
    elif args.command == 'add-provider':
        add_provider()
    elif args.command == 'reload':
        llm_manager.reload_config()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
