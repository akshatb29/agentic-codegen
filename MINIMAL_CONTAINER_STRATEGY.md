# Minimal Stable Container Strategy

## Philosophy

**Build Once, Install On-Demand**: Create minimal stable Docker containers with only essential system dependencies, then let the coding correction flow handle specific Python package installation dynamically.

## Key Changes

### 1. 🏗️ **Minimal Base Containers**
- **Before**: Pre-installed all suspected packages (pandas, numpy, flask, etc.)
- **After**: Only essential tools (pip, setuptools, wheel) + system dependencies
- **Result**: Faster builds, stable environment, no dependency conflicts

### 2. 🔄 **Dynamic Package Installation**
- **Container Persistence**: Failed executions preserve containers for iterative development
- **Package Tracking**: System tracks what's installed in each container
- **Correction Flow Integration**: Containers ready for AI-driven package installation

### 3. 📦 **System Dependencies Only**
```dockerfile
# GUI apps get:
RUN apt-get install python3-tk python3-dev

# Data/ML apps get:  
RUN apt-get install gcc g++ gfortran libopenblas-dev liblapack-dev

# Web apps get:
RUN apt-get install curl

# All get:
RUN pip install pip setuptools wheel
```

### 4. 🛠️ **Container Methods Added**
```python
# Install packages dynamically
docker_manager.install_packages_in_container(container_name, ["pandas", "numpy"])

# Track what's installed
installed = docker_manager.get_container_installed_packages(container_name)

# Execution results include container info
result = {
    "container_available": True,
    "container_supports_pip_install": True, 
    "installed_packages": ["requests", "pandas"],
    "container_name": "msc-dev-session-1234"
}
```

## Benefits

### 🚀 **Performance**
- **Faster Builds**: No large package downloads during image creation
- **Instant Reuse**: Containers start immediately, packages install only when needed
- **Smaller Images**: Minimal base images with targeted additions

### 🧠 **AI Integration**
- **Error-Driven Installation**: Missing import errors trigger package installation
- **Smart Dependencies**: AI can analyze errors and install correct packages
- **Iterative Development**: Failed runs don't lose progress

### 🔧 **Developer Experience**
- **Quick Setup**: No waiting for pre-built environments
- **Flexible**: Same container adapts to different requirements
- **Persistent**: Development state maintained across runs

## Workflow Example

### Initial Run
```bash
🤖 Analyzing task requirements for minimal Docker environment...
🎯 App Type: gui_app
📦 Generated Image: msc-gui-1234
🔨 Building Docker image: msc-gui-1234  # Only tkinter system deps
✅ Successfully built image: msc-gui-1234
🚀 Creating new development container: msc-dev-session-5678
🏃 Executing in container: msc-dev-session-5678
❌ [Docker] Container execution failed.
💡 Container ready for correction flow - packages can be installed dynamically
```

### Correction Flow (External AI Agent)
```python
# AI detects "ModuleNotFoundError: No module named 'pandas'"
# AI calls: docker_manager.install_packages_in_container("msc-dev-session-5678", ["pandas"])
# AI re-runs corrected code in same container
```

### Subsequent Runs
```bash
🔍 Found 1 existing development containers:
  1. 🟢 msc-dev-session-5678 (msc-gui-1234) - running
Reuse existing container? (1, Enter for new, 'n' for new): 1
♻️ Using existing container: msc-dev-session-5678
🏃 Executing in container: msc-dev-session-5678
✅ [Docker] Code executed successfully in container.
```

## Configuration

### Minimal Dockerfile Template
```dockerfile
FROM python:3.11-slim

# Security: non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Essential compilers
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# App-specific system deps (GUI/ML/Web)
[SYSTEM_PACKAGES_IF_NEEDED]

# Working environment
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN chown -R appuser:appuser /app
USER appuser

# User installs enabled
ENV PIP_USER=1
ENV PATH="/home/appuser/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1
```

### Container Metadata
```json
{
  "msc-dev-session-5678": {
    "container_id": "abc123...",
    "image_name": "msc-gui-1234", 
    "created_at": 1753206789,
    "session_files": ["script1.py", "script2.py"],
    "installed_packages": ["pandas", "numpy", "matplotlib"]
  }
}
```

## Integration Points

### For Correction Agents
```python
# Check if container execution failed but container is available
if not result["success"] and result.get("container_available"):
    # Analyze error for missing packages
    missing_packages = analyze_import_errors(result["stderr"])
    
    # Install missing packages
    if missing_packages:
        success = docker_manager.install_packages_in_container(
            result["container_name"], 
            missing_packages
        )
        
        # Re-run code in same container
        if success:
            result = docker_manager.execute_in_container(
                result["container_name"], 
                corrected_code
            )
```

### For Session Management
- Containers persist across keyboard interrupts
- User choice to save/remove containers at session end
- Automatic cleanup of session files and temporary scripts
- Container state preserved for multi-session development

## Result

✅ **Stable Base Environment**: Never changes after build
✅ **Fast Iteration**: No rebuild cycles for dependency changes  
✅ **AI-Friendly**: Clear error signals for package installation
✅ **Resource Efficient**: Minimal storage and memory usage
✅ **Developer-Focused**: Seamless multi-session development
