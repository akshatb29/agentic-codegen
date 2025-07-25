# 🏗️ MSC Framework - Architecture Diagram

## System Overview

```mermaid
graph TB
    %% Entry Point
    A[User Request] --> B[File Context Selection]
    B --> C[Agentic Planner]
    
    %% Planning Phase (Multi-Agent)
    subgraph "Planning Phase"
        C --> D[Tech Stack Analysis]
        D --> E[Project Structure]
        E --> F[Software Design]
        F --> G[Generation Strategy]
    end
    
    %% Code Generation (Strategy-Based)
    subgraph "Generation Strategies"
        G --> H{Strategy Router}
        H -->|NL| I[Natural Language → Code]
        H -->|Pseudocode| J[Pseudocode Refinement → Code]
        H -->|Symbolic| K[Symbolic Reasoning → Code]
    end
    
    %% Verification & Quality Loop
    subgraph "Quality Assurance"
        I --> L[Code Verifier]
        J --> L
        K --> L
        L --> M[Critique Agent]
        M --> N{Quality Check}
        N -->|Pass| O[Docker Execution]
        N -->|Fail| P[Corrector Agent]
        P --> L
    end
    
    %% Testing & Deployment
    subgraph "Testing Phase"
        O --> Q[Containerized Testing]
        Q --> R[Integration Tests]
        R --> S[Final Validation]
    end
    
    %% Output
    S --> T[Generated Application]
    
    %% State Management (Central)
    subgraph "State Management"
        U[AgentState]
        V[File Context]
        W[Execution Environment]
    end
    
    %% Tools Layer
    subgraph "Tools & Utilities"
        X[LLM Manager]
        Y[Docker Tools]
        Z[File System]
        AA[Context Analyzer]
    end
    
    %% Connections to state and tools
    C -.-> U
    L -.-> U
    M -.-> U
    O -.-> Y
    Q -.-> Y
    G -.-> X
    D -.-> AA
```

## Data Flow Architecture

```mermaid
graph LR
    %% Input Layer
    A[User Input] --> B[Context Analysis]
    B --> C[AgentState Initialization]
    
    %% Processing Layer
    C --> D[Multi-Agent Processing]
    
    subgraph "Agent Communication"
        D --> E[Agent 1: Planning]
        D --> F[Agent 2: Generation]
        D --> G[Agent 3: Verification]
        E <--> F
        F <--> G
        G <--> E
    end
    
    %% State Management
    subgraph "Centralized State"
        H[AgentState TypedDict]
        I[File Context]
        J[Generation Artifacts]
        K[Verification Results]
    end
    
    E --> H
    F --> J
    G --> K
    
    %% Output Layer
    H --> L[Code Files]
    J --> L
    K --> M[Test Results]
    L --> N[Final Application]
    M --> N
```

## Component Interaction Map

```mermaid
graph TD
    %% Core Components
    A[Main Entry Point] --> B[Graph Builder]
    B --> C[LangGraph Workflow]
    
    %% Agent Layer
    subgraph "Agent Ecosystem"
        D[Agentic Planner]
        E[Code Generators]
        F[Reasoners]
        G[Verifiers]
        H[Critics]
        I[Correctors]
        J[Docker Agents]
    end
    
    %% Tools Layer
    subgraph "Tool Ecosystem"
        K[LLM Manager]
        L[File System Tools]
        M[Docker Manager]
        N[Context Analyzer]
        O[User Interaction]
    end
    
    %% Configuration Layer
    subgraph "Configuration"
        P[LLM Config]
        Q[Prompts]
        R[Docker Specs]
    end
    
    %% Data Layer
    subgraph "Data Management"
        S[AgentState]
        T[File Context]
        U[Execution Results]
    end
    
    %% Connections
    C --> D
    C --> E
    C --> F
    C --> G
    C --> H
    C --> I
    C --> J
    
    D --> K
    E --> K
    F --> K
    G --> L
    H --> L
    I --> L
    J --> M
    
    K --> P
    L --> Q
    M --> R
    
    D --> S
    E --> S
    F --> S
    G --> T
    H --> T
    I --> U
    J --> U
```

## Agentic vs Workflow Balance

### Current Implementation: **60% Agentic / 40% Workflow**

#### Agentic Characteristics (60%):
- ✅ **Autonomous Decision Making**: Agents choose strategies independently
- ✅ **Multi-Agent Collaboration**: Graph-of-Thoughts with agent communication
- ✅ **Dynamic Strategy Selection**: Context-aware generation approach
- ✅ **Self-Correction Loops**: Automatic error detection and fixing
- ✅ **Adaptive Behavior**: Complexity-based task decomposition

#### Workflow Characteristics (40%):
- ⚪ **Fixed Graph Structure**: Predefined node sequences
- ⚪ **Centralized State**: AgentState as communication hub
- ⚪ **Human Approval Gates**: User confirmation at key points
- ⚪ **Predetermined Paths**: Limited dynamic workflow modification

## Key Features

### 🎯 **Multi-Strategy Code Generation**
- **Natural Language**: Direct prompt-to-code
- **Pseudocode**: Iterative refinement approach
- **Symbolic**: Formal logic and mathematical reasoning

### 🔄 **Quality Assurance Pipeline**
- **Verification**: Syntax and execution testing
- **Critique**: Quality analysis and suggestions
- **Correction**: Automated error fixing
- **Docker Testing**: Containerized validation

### 🤖 **Autonomous Capabilities**
- **Task Decomposition**: Auto-break complex requests
- **Tech Stack Analysis**: Smart technology detection
- **Project Structure**: Automatic architecture planning
- **Strategy Selection**: Context-aware approach choice

### 🐳 **Docker Integration**
- **Multi-App Support**: GUI, Web, Data Analysis, ML
- **Environment Isolation**: Safe code execution
- **Container Management**: Automated setup and cleanup
- **Testing Pipeline**: Comprehensive validation

## Technology Stack

### **Core Framework**
- **LangGraph**: Workflow orchestration
- **LangChain**: LLM integration and chaining
- **Pydantic**: Data validation and modeling

### **LLM Providers**
- **Google Gemini**: Primary generation model
- **Ollama**: Local model support
- **Multi-Provider**: Configurable LLM backends

### **Execution Environment**
- **Docker**: Containerized execution
- **Python**: Core runtime environment
- **Rich**: Enhanced CLI interface

### **File Management**
- **Smart Context**: Intelligent file selection
- **Directory Analysis**: Automatic structure detection
- **Version Control**: Git integration ready
