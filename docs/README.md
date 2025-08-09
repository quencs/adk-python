# Google ADK Documentation

Welcome to the comprehensive documentation for the Google Agent Development Kit (ADK)! This documentation provides detailed information about all public APIs, functions, and components, along with practical examples and usage instructions.

## Documentation Overview

This documentation is organized into several specialized guides to help you understand and use Google ADK effectively:

### 📚 Core Documentation

#### [**User Guide**](user-guide.md)
**Start here if you're new to Google ADK!**

A comprehensive getting-started guide that covers:
- Installation and setup instructions
- Core concepts and terminology
- Building your first agent step-by-step
- Working with tools and multi-agent systems
- Best practices and troubleshooting

Perfect for developers new to agent development or those transitioning from other frameworks.

#### [**API Reference**](api-reference.md)
**Complete reference for all public APIs**

Detailed documentation for all classes, functions, and interfaces:
- Core classes (`Agent`, `Runner`)
- Tools and toolsets (`BaseTool`, built-in tools)
- Models and LLM integration (`Gemini`, `BaseLlm`)
- Sessions and state management
- Type aliases and callback functions

Essential for developers who need detailed API information while building applications.

#### [**Tools Guide**](tools-guide.md)
**Comprehensive guide to ADK's tool ecosystem**

Everything you need to know about tools:
- Built-in tools (Google Search, Enterprise Search, etc.)
- Creating custom tools and toolsets
- Tool configuration and best practices
- Integration patterns with APIs and databases
- Advanced tool development techniques

Perfect for developers building custom tools or integrating external services.

#### [**Examples**](examples.md)
**Practical examples and usage patterns**

Real-world examples covering:
- Basic agent examples with progressive complexity
- Multi-agent systems and coordination patterns
- Tool integration (REST APIs, databases)
- Advanced patterns (streaming, state management)
- Production deployment examples
- Complete application examples

Ideal for learning through practical examples and understanding implementation patterns.

#### [**Utilities Guide**](utilities-guide.md)
**Helper functions and utility classes**

Documentation for ADK's utility modules:
- Model name utilities (parsing, validation)
- Feature decorators (experimental, WIP features)
- Instruction utilities (template processing)
- Variant utilities (Google LLM variants)
- Usage examples and composition patterns

Useful for advanced developers who want to leverage ADK's utility functions.

---

## Quick Start

### Installation

```bash
# Install Google ADK
pip install google-adk

# For development version with latest features
pip install git+https://github.com/google/adk-python.git@main
```

### Hello World Example

```python
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService

# Create a simple agent
agent = Agent(
    name="hello_agent",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant. Be friendly and concise."
)

# Set up and run
session_service = InMemorySessionService()
runner = Runner(
    app_name="hello_world",
    agent=agent,
    session_service=session_service
)

# Interact with the agent
for event in runner.run(
    user_id="user123",
    session_id="session456",
    new_message="Hello! What can you help me with?"
):
    if event.type == "model_response":
        print(f"Agent: {event.data.get('text', '')}")
```

For more detailed examples, see the [User Guide](user-guide.md) and [Examples](examples.md).

---

## Architecture Overview

Google ADK is built around four core concepts:

### 🤖 Agents
The primary building blocks that encapsulate:
- **Model**: The LLM (e.g., Gemini) that powers the agent
- **Instructions**: Behavioral guidelines and personality
- **Tools**: Capabilities and external integrations
- **Context**: Conversation state and memory

### 🛠️ Tools
Extensible capabilities that provide:
- External API access
- Database operations
- Custom business logic
- Inter-agent communication

### 💾 Sessions
State management for:
- Conversation history
- User and session identification
- Context preservation
- State persistence

### 🏃 Runners
Execution orchestration including:
- Message processing
- Event generation
- Service coordination
- Error handling and recovery

---

## Key Features

### 🔧 **Code-First Development**
Define agents, tools, and workflows directly in Python with full type safety and IDE support.

```python
from google.adk import Agent
from google.adk.tools import google_search, FunctionTool

def calculate_tip(bill: float, percentage: float = 15.0) -> dict:
    """Calculate tip amount."""
    tip = bill * (percentage / 100)
    return {"tip": tip, "total": bill + tip}

agent = Agent(
    name="helpful_assistant",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant with search and calculation capabilities.",
    tools=[google_search, FunctionTool(func=calculate_tip)]
)
```

### 🌐 **Rich Tool Ecosystem**
Extensive collection of pre-built tools plus easy custom tool creation:

```python
from google.adk.tools import (
    google_search,           # Web search
    enterprise_web_search,   # Enterprise search
    url_context,            # Web page analysis
    load_memory,            # Memory management
    AgentTool,              # Inter-agent communication
    VertexAiSearchTool      # Custom search
)
```

### 🤝 **Multi-Agent Systems**
Build complex workflows with specialized agents:

```python
# Specialized agents
researcher = Agent(name="researcher", tools=[google_search], ...)
writer = Agent(name="writer", ...)
editor = Agent(name="editor", ...)

# Coordinator agent
coordinator = Agent(
    name="content_team",
    tools=[AgentTool(agent=researcher), AgentTool(agent=writer), AgentTool(agent=editor)]
)
```

### 🚀 **Production Ready**
Deploy anywhere with built-in support for:
- FastAPI web services
- Docker containerization
- Cloud Run deployment
- Vertex AI integration
- Horizontal scaling

### 🔄 **Model Agnostic**
While optimized for Gemini, ADK supports various LLMs:

```python
# Gemini models (recommended)
agent1 = Agent(model="gemini-2.0-flash", ...)
agent2 = Agent(model="gemini-1.5-pro", ...)

# Custom model integration
from google.adk.models import BaseLlm
custom_model = CustomLLM()
agent3 = Agent(model=custom_model, ...)
```

---

## Common Use Cases

### 💬 **Conversational AI**
Build sophisticated chatbots and virtual assistants with memory, tools, and personality.

### 🔍 **Research and Analysis**
Create agents that can search, analyze, and synthesize information from multiple sources.

### 🏢 **Enterprise Automation**
Automate business processes with agents that integrate with internal systems and APIs.

### 📝 **Content Creation**
Build multi-agent workflows for research, writing, editing, and publishing content.

### 🎯 **Specialized Assistants**
Create domain-specific assistants for coding, data analysis, customer support, etc.

### 🌐 **API Integration**
Connect AI agents to external services, databases, and enterprise systems.

---

## Getting Help

### 📖 **Documentation Navigation**

1. **New to ADK?** → Start with the [User Guide](user-guide.md)
2. **Building tools?** → Check the [Tools Guide](tools-guide.md)
3. **Need API details?** → See the [API Reference](api-reference.md)
4. **Want examples?** → Browse the [Examples](examples.md)
5. **Using utilities?** → Read the [Utilities Guide](utilities-guide.md)

### 🔗 **External Resources**

- **[Official Documentation](https://google.github.io/adk-docs)** - Complete documentation website
- **[Sample Projects](https://github.com/google/adk-samples)** - Example applications and templates
- **[ADK Web Interface](https://github.com/google/adk-web)** - Browser-based development UI
- **[Community Forum](https://www.reddit.com/r/agentdevelopmentkit/)** - Community discussions and support

### 🐛 **Troubleshooting**

Common issues and solutions are covered in the [User Guide's Troubleshooting section](user-guide.md#troubleshooting).

For bugs and feature requests:
- **Python ADK**: [GitHub Issues](https://github.com/google/adk-python/issues)
- **Documentation**: [Documentation Issues](https://github.com/google/adk-docs/issues)

---

## Version Information

This documentation covers **Google ADK Python** and is compatible with:
- **Python**: 3.9, 3.10, 3.11, 3.12, 3.13
- **Google Cloud**: All regions supporting Gemini
- **Vertex AI**: Agent Engine integration
- **Deployment**: Cloud Run, containers, local development

---

## Contributing

We welcome contributions to both ADK and its documentation! See:
- [Contributing Guidelines](https://google.github.io/adk-docs/contributing-guide/)
- [Code Contributing Guidelines](https://github.com/google/adk-python/blob/main/CONTRIBUTING.md)

---

## License

Google ADK is licensed under the Apache 2.0 License. See the [LICENSE](https://github.com/google/adk-python/blob/main/LICENSE) file for details.

---

**Happy Agent Building!** 🚀

*This documentation was generated to provide comprehensive coverage of Google ADK's public APIs, functions, and components. For the most up-to-date information, visit the [official documentation](https://google.github.io/adk-docs).*