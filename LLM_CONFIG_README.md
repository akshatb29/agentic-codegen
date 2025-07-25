# LLM Configuration System

This document describes the new centralized LLM configuration system for the MSC (Multi-Stage Coding) framework.

## Overview

The LLM configuration system decouples LLM connections from individual agents, providing:

- **Centralized Configuration**: All LLM settings in one YAML file
- **Provider Flexibility**: Easy switching between Google, OpenAI, Anthropic, Ollama (local)
- **Agent-Specific Settings**: Different models/parameters per agent
- **Runtime Configuration**: Change settings without code modification
- **Connection Caching**: Efficient LLM instance reuse
- **Local Model Support**: Use Ollama for private, offline inference

## Configuration File

The main configuration is in `config/llm_config.yaml`:

```yaml
# Available LLM Providers
providers:
  google:
    class: "langchain_google_genai.ChatGoogleGenerativeAI"
    models: ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest"]
    default_params:
      temperature: 0.1
      
  ollama:
    class: "langchain_community.llms.Ollama"
    base_url: "http://localhost:11434"
    models: ["codellama:latest", "llama3.2:latest"]
    default_params:
      temperature: 0.1

# Agent-specific configurations
agents:
  planner:
    provider: "google"
    model: "gemini-1.5-flash-latest"
    temperature: 0.1
    description: "Strategic planning and architecture design"
    
  code_generator:
    provider: "ollama"  # Use local model for privacy
    model: "codellama:latest"
    temperature: 0.2
    description: "Local code generation with CodeLlama"
```

## Usage

### In Agent Code

Replace direct LLM imports with the centralized manager:

```python
# OLD WAY ❌
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1)

# NEW WAY ✅
from msc.tools import get_llm
llm = get_llm("corrector")  # Uses config for 'corrector' agent
```

### Configuration Management

#### Command Line Interface

```bash
# List all configurations
python -m msc.tools.llm_config_cli list

# Update agent configuration interactively
python -m msc.tools.llm_config_cli update

# Test all connections
python -m msc.tools.llm_config_cli test

# Add new provider
python -m msc.tools.llm_config_cli add-provider
```

#### Programmatic Access

```python
from msc.tools.llm_manager import llm_manager

# Get agent configuration
config = llm_manager.get_agent_config("planner")

# Update configuration
llm_manager.update_agent_config("planner", 
                                provider="openai", 
                                model="gpt-4o")

# Reload from file
llm_manager.reload_config()
```

## Adding New Providers

1. **Add to configuration**:
```yaml
providers:
  new_provider:
    class: "langchain_newprovider.ChatNewProvider"
    models: ["model-1", "model-2"]
    default_params:
      temperature: 0.1
```

2. **Install dependencies**:
```bash
pip install langchain-newprovider
```

3. **Set environment variables** (if required):
```bash
export NEW_PROVIDER_API_KEY="your-key"
```

### Local Models with Ollama

For privacy and offline usage, you can use Ollama:

1. **Install Ollama**: https://ollama.ai/
2. **Pull models**: `ollama pull codellama:latest`
3. **Start service**: `ollama serve`
4. **Configure in YAML**:
```yaml
agents:
  code_generator:
    provider: "ollama"
    model: "codellama:latest"
```

See `OLLAMA_SETUP_GUIDE.md` for detailed instructions.

## Environment Variables

Set these environment variables for each provider:

- **Google**: `GOOGLE_API_KEY`
- **OpenAI**: `OPENAI_API_KEY`
- **Anthropic**: `ANTHROPIC_API_KEY`
- **Ollama**: No API key required (local models)

## Agent Configurations

### Current Agents

| Agent | Purpose | Default Model | Temperature |
|-------|---------|---------------|-------------|
| `planner` | Architecture & strategy | gemini-1.5-flash | 0.1 |
| `code_generator` | Code generation | gemini-1.5-flash | 0.2 |
| `corrector` | Code fixing | gemini-1.5-flash | 0.1 |
| `critique` | Code review | gemini-1.5-flash | 0.1 |
| `reasoner` | Symbolic reasoning | gemini-1.5-flash | 0.1 |
| `docker_agent` | Docker specs | gemini-1.5-flash | 0.3 |

### Adding New Agents

Add to `config/llm_config.yaml`:

```yaml
agents:
  my_new_agent:
    provider: "google"
    model: "gemini-1.5-flash-latest"
    temperature: 0.2
    description: "My custom agent"
```

Then use in code:
```python
llm = get_llm("my_new_agent")
```

## Advanced Features

### Parameter Override

Override configuration at runtime:

```python
# Use different temperature for this call
llm = get_llm("corrector", temperature=0.5)

# Use different model
llm = get_llm("planner", model="gemini-1.5-pro-latest")
```

### Caching

LLM instances are cached for efficiency:
- Same agent + configuration = same instance
- Cache cleared when configuration changes
- Manual cache clearing: `llm_manager._llm_cache.clear()`

### Error Handling

The system handles common errors:
- Missing API keys
- Invalid provider/model combinations
- Import errors for unavailable providers
- Configuration file issues

## Migration Guide

### For Existing Agents

1. **Remove direct imports**:
```python
# Remove this
from langchain_google_genai import ChatGoogleGenerativeAI
GENERATOR_MODEL = "gemini-1.5-flash-latest"
```

2. **Add LLM manager import**:
```python
from msc.tools import get_llm
```

3. **Replace LLM creation**:
```python
# Replace this
llm = ChatGoogleGenerativeAI(model=GENERATOR_MODEL, temperature=0.1)

# With this
llm = get_llm("agent_name")
```

### Configuration Update

Run the test to ensure everything works:
```bash
python test_llm_config.py
```

## Troubleshooting

### Common Issues

1. **"Configuration file not found"**
   - Ensure `config/llm_config.yaml` exists
   - Check file permissions

2. **"API key not found"**
   - Set required environment variables
   - Check variable names in config

3. **"Failed to import provider"**
   - Install required packages: `pip install langchain-provider`
   - Check class path in configuration

4. **"Unknown provider"**
   - Add provider to `providers` section
   - Update environment variables mapping

### Testing

```bash
# Test configuration loading
python test_llm_config.py

# Test specific agent
python -c "from msc.tools import get_llm; print(get_llm('corrector'))"

# Test connection
python -m msc.tools.llm_config_cli test
```

## Benefits

- ✅ **Flexibility**: Switch providers/models without code changes
- ✅ **Consistency**: Centralized configuration management
- ✅ **Efficiency**: Connection caching and reuse
- ✅ **Maintainability**: Clear separation of concerns
- ✅ **Scalability**: Easy addition of new providers/agents
- ✅ **Testing**: Built-in connection testing tools
