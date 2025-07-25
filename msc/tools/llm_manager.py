# msc/tools/llm_manager.py
"""
Centralized LLM Manager for MSC Framework
Handles LLM provider configuration, instantiation, and agent-specific settings.
"""

import os
import yaml
import importlib
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from rich.console import Console

console = Console()

@dataclass
class LLMConfig:
    """Configuration for an LLM instance"""
    provider: str
    model: str
    temperature: float
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    description: str = ""

class LLMManager:
    """Centralized manager for LLM connections and configurations"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the LLM manager with configuration"""
        if config_path is None:
            # Default to config/llm_config.yaml relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "llm_config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._llm_cache = {}  # Cache LLM instances for reuse
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            console.print(f"[green]✓[/green] Loaded LLM configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            console.print(f"[red]✗[/red] Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            console.print(f"[red]✗[/red] Error parsing YAML configuration: {e}")
            raise
    
    def _get_provider_class(self, provider: str):
        """Dynamically import and return the provider class"""
        if provider not in self.config['providers']:
            raise ValueError(f"Unknown provider: {provider}")
        
        provider_config = self.config['providers'][provider]
        class_path = provider_config['class']
        
        # Split module and class name
        module_name, class_name = class_path.rsplit('.', 1)
        
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            console.print(f"[red]✗[/red] Failed to import {class_path}: {e}")
            raise
    
    def _check_api_key(self, provider: str) -> bool:
        """Check if the required API key is available"""
        env_var = self.config['env_vars'].get(provider)
        # If env_var is None (like for Ollama), no API key is required
        if env_var is None:
            return True
        if env_var and not os.getenv(env_var):
            console.print(f"[yellow]⚠[/yellow] API key not found for {provider}. Set {env_var} environment variable.")
            return False
        return True
    
    def get_agent_config(self, agent_name: str) -> LLMConfig:
        """Get configuration for a specific agent"""
        agent_config = self.config['agents'].get(agent_name)
        if not agent_config:
            console.print(f"[yellow]⚠[/yellow] No specific config for agent '{agent_name}', using defaults")
            agent_config = self.config['defaults']
        
        # Merge with defaults
        defaults = self.config['defaults']
        merged_config = {**defaults, **agent_config}
        
        return LLMConfig(
            provider=merged_config['provider'],
            model=merged_config['model'],
            temperature=merged_config['temperature'],
            max_tokens=merged_config.get('max_tokens'),
            timeout=merged_config.get('timeout'),
            description=merged_config.get('description', '')
        )
    
    def get_llm_for_agent(self, agent_name: str, **kwargs):
        """Get LLM instance for a specific agent"""
        config = self.get_agent_config(agent_name)
        
        # Create cache key
        cache_key = f"{agent_name}_{config.provider}_{config.model}_{config.temperature}"
        
        # Return cached instance if available
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]
        
        # Check API key
        if not self._check_api_key(config.provider):
            raise RuntimeError(f"API key not available for provider: {config.provider}")
        
        # Get provider class
        provider_class = self._get_provider_class(config.provider)
        
        # Prepare parameters based on provider type
        provider_config = self.config['providers'][config.provider]
        provider_defaults = provider_config['default_params']
        
        params = {
            **provider_defaults,
            'temperature': config.temperature,
        }
        
        # Provider-specific parameter handling
        if config.provider == "ollama":
            # Ollama uses different parameter names
            params['model'] = config.model
            # Add base_url if specified in config
            if 'base_url' in provider_config:
                params['base_url'] = provider_config['base_url']
        else:
            # Standard LangChain chat models
            params['model'] = config.model
            
            # Inject API key from environment if required
            env_var = self.config['env_vars'].get(config.provider)
            if env_var:
                api_key = os.getenv(env_var)
                if api_key:
                    if config.provider == "google":
                        params['google_api_key'] = api_key
                    elif config.provider == "openai":
                        params['openai_api_key'] = api_key
                    elif config.provider == "anthropic":
                        params['anthropic_api_key'] = api_key
        
        # Add optional parameters
        if config.max_tokens:
            params['max_tokens'] = config.max_tokens
        if config.timeout:
            params['timeout'] = config.timeout
            
        # Override with any kwargs passed
        params.update(kwargs)
        
        try:
            llm = provider_class(**params)
            self._llm_cache[cache_key] = llm
            console.print(f"[green]✓[/green] Created LLM for {agent_name}: {config.provider}/{config.model}")
            return llm
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to create LLM for {agent_name}: {e}")
            raise
    
    def list_available_models(self, provider: Optional[str] = None) -> Dict[str, list]:
        """List all available models by provider"""
        if provider:
            return {provider: self.config['providers'][provider]['models']}
        return {p: config['models'] for p, config in self.config['providers'].items()}
    
    def update_agent_config(self, agent_name: str, **updates):
        """Update configuration for a specific agent"""
        if agent_name not in self.config['agents']:
            self.config['agents'][agent_name] = {}
        
        self.config['agents'][agent_name].update(updates)
        
        # Clear cache for this agent
        cache_keys_to_remove = [k for k in self._llm_cache.keys() if k.startswith(f"{agent_name}_")]
        for key in cache_keys_to_remove:
            del self._llm_cache[key]
        
        console.print(f"[green]✓[/green] Updated configuration for {agent_name}")
    
    def reload_config(self):
        """Reload configuration from file"""
        self.config = self._load_config()
        self._llm_cache.clear()  # Clear all cached instances
        console.print("[green]✓[/green] Configuration reloaded")

# Global instance - can be imported and used across the framework
llm_manager = LLMManager()

# Convenience functions for easy access
def get_llm(agent_name: str, **kwargs):
    """Get LLM instance for an agent - convenience function"""
    return llm_manager.get_llm_for_agent(agent_name, **kwargs)

def get_agent_config(agent_name: str) -> LLMConfig:
    """Get configuration for an agent - convenience function"""
    return llm_manager.get_agent_config(agent_name)
