# Ollama Setup Guide for MSC Framework

This guide explains how to set up and use Ollama with local models in the MSC framework.

## What is Ollama?

Ollama is a tool that allows you to run large language models locally on your machine. This gives you:

- ✅ **Privacy**: Your code and data never leave your machine
- ✅ **No API costs**: Run models for free after initial setup
- ✅ **Offline capability**: Work without internet connection
- ✅ **Custom models**: Use specialized coding models like CodeLlama
- ✅ **Fast inference**: Local models can be very fast

## Installation

### 1. Install Ollama

**Linux/macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from https://ollama.ai/download

### 2. Start Ollama Service

```bash
ollama serve
```

This starts the Ollama server on `http://localhost:11434`

### 3. Pull Recommended Models

```bash
# For general tasks
ollama pull llama3.2:latest

# For code generation (highly recommended)
ollama pull codellama:latest

# For advanced coding tasks
ollama pull deepseek-coder:latest

# Lightweight but capable
ollama pull phi3:latest

# For reasoning tasks
ollama pull qwen2.5:latest
```

### 4. Verify Installation

```bash
ollama list
```

Should show your downloaded models.

## Configuration

### Using Ollama Configuration Template

Copy the Ollama example configuration:

```bash
cp config/llm_config_ollama_example.yaml config/llm_config.yaml
```

Or create your own configuration with Ollama agents:

```yaml
agents:
  code_generator:
    provider: "ollama"
    model: "codellama:latest"  # Excellent for code generation
    temperature: 0.2
    
  corrector:
    provider: "ollama"
    model: "deepseek-coder:latest"  # Great for code fixes
    temperature: 0.1
    
  planner:
    provider: "ollama"
    model: "llama3.2:latest"  # Good for planning
    temperature: 0.1
```

### Mixed Provider Setup

You can mix Ollama with cloud providers:

```yaml
agents:
  code_generator:
    provider: "ollama"
    model: "codellama:latest"  # Local for privacy
    
  planner:
    provider: "google"
    model: "gemini-1.5-flash-latest"  # Cloud for complex planning
    
  critique:
    provider: "anthropic"
    model: "claude-3-5-sonnet-20241022"  # Cloud for detailed review
```

## Recommended Models by Task

### Code Generation
- **`codellama:latest`** - Specialized for code generation
- **`deepseek-coder:latest`** - Excellent code understanding
- **`starcoder2:latest`** - Good for multiple languages

### General Tasks
- **`llama3.2:latest`** - Latest Llama model, well-balanced
- **`llama3.1:latest`** - Previous version, very capable
- **`qwen2.5:latest`** - Good reasoning capabilities

### Lightweight Options
- **`phi3:latest`** - Small but capable model
- **`mistral:latest`** - Fast and efficient

## Testing Your Setup

Run the Ollama test suite:

```bash
python test_ollama_config.py
```

This will:
- Check if Ollama is running
- Verify configuration loading
- Test LLM creation
- Show available models

## Usage Examples

### Basic Usage
```python
from msc.tools import get_llm

# Use local CodeLlama for code generation
llm = get_llm("code_generator")  # Uses ollama/codellama:latest
```

### Runtime Model Switching
```python
from msc.tools.llm_manager import llm_manager

# Switch to a different local model
llm_manager.update_agent_config("corrector", 
                               provider="ollama", 
                               model="deepseek-coder:latest")
```

### Check Available Models
```bash
# CLI tool
python -m msc.tools.llm_config_cli list

# Or via Ollama directly
ollama list
```

## Performance Tips

### Model Size vs. Performance
- **Small models** (7B params): Fast, good for simple tasks
- **Medium models** (13B params): Balanced performance/speed
- **Large models** (70B+ params): Best quality, slower

### Hardware Requirements
- **Minimum**: 8GB RAM, modern CPU
- **Recommended**: 16GB+ RAM, GPU with VRAM
- **Optimal**: 32GB+ RAM, dedicated GPU

### Optimization
```bash
# Use quantized models for faster inference
ollama pull codellama:7b-code-q4_0

# Monitor resource usage
ollama ps
```

## Troubleshooting

### Common Issues

1. **"Connection refused"**
   ```bash
   # Start Ollama service
   ollama serve
   ```

2. **"Model not found"**
   ```bash
   # Pull the model first
   ollama pull llama3.2:latest
   ```

3. **Slow performance**
   ```bash
   # Use smaller/quantized models
   ollama pull phi3:latest
   ```

4. **Out of memory**
   ```bash
   # Use smaller models or close other applications
   ollama pull codellama:7b
   ```

### Debug Configuration
```python
# Test Ollama connectivity
import requests
response = requests.get("http://localhost:11434/api/tags")
print(response.json())
```

## Model Comparison

| Model | Size | Best For | Speed | Quality |
|-------|------|----------|--------|---------|
| `phi3:latest` | 3.8B | Quick tasks | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| `codellama:latest` | 7B | Code generation | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| `llama3.2:latest` | 3B/1B | General purpose | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| `deepseek-coder:latest` | 6.7B | Code understanding | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| `qwen2.5:latest` | 7B | Reasoning | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## Configuration Examples

### All-Local Setup
```yaml
# Use only local models
defaults:
  provider: "ollama"
  model: "llama3.2:latest"

agents:
  code_generator:
    model: "codellama:latest"
  corrector:
    model: "deepseek-coder:latest"
```

### Hybrid Setup
```yaml
# Mix local and cloud for optimal cost/performance
agents:
  code_generator:
    provider: "ollama"  # Local for privacy
    model: "codellama:latest"
    
  planner:
    provider: "google"  # Cloud for complex reasoning
    model: "gemini-1.5-flash-latest"
```

### Development Setup
```yaml
# Fast models for development
agents:
  code_generator:
    provider: "ollama"
    model: "phi3:latest"  # Fast for quick iterations
    temperature: 0.3
```

## Next Steps

1. **Start with recommended models**: `codellama:latest` and `llama3.2:latest`
2. **Test performance**: Run some coding tasks to see how it performs
3. **Optimize configuration**: Adjust models based on your hardware and needs
4. **Explore specialized models**: Try domain-specific models for your use case

For more information, visit:
- Ollama Documentation: https://ollama.ai/
- Model Library: https://ollama.ai/library
- MSC Framework LLM Config: `LLM_CONFIG_README.md`
