# Google ADK User Guide

Welcome to the Google Agent Development Kit (ADK)! This guide will help you get started with building AI agents, from basic concepts to advanced multi-agent systems.

## Table of Contents

- [What is Google ADK?](#what-is-google-adk)
- [Installation and Setup](#installation-and-setup)
- [Core Concepts](#core-concepts)
- [Your First Agent](#your-first-agent)
- [Working with Tools](#working-with-tools)
- [Multi-Agent Systems](#multi-agent-systems)
- [Advanced Features](#advanced-features)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## What is Google ADK?

The Google Agent Development Kit (ADK) is a flexible and modular framework for developing and deploying AI agents. It's designed to make agent development feel more like software development, providing:

### Key Benefits

- **Code-First Development**: Define agents, tools, and workflows directly in Python
- **Model Agnostic**: Works with various LLMs, optimized for Gemini
- **Rich Tool Ecosystem**: Pre-built tools and easy custom tool creation  
- **Multi-Agent Support**: Build complex systems with multiple specialized agents
- **Production Ready**: Deploy to Cloud Run, Vertex AI, or any container platform

### When to Use ADK

- Building conversational AI applications
- Creating specialized AI assistants
- Developing multi-agent workflows
- Integrating AI with existing systems
- Rapid prototyping of AI solutions

---

## Installation and Setup

### Requirements

- Python 3.9 or higher
- Google Cloud account (for Gemini models)
- Optional: Docker for containerized deployment

### Installation

```bash
# Install the latest stable version
pip install google-adk

# Or install from source for latest features
pip install git+https://github.com/google/adk-python.git@main
```

### Authentication Setup

For Google Cloud services (recommended for production):

```bash
# Install Google Cloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth application-default login

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

Alternatively, use service account credentials:

```python
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/credentials.json"
```

### Verify Installation

```python
from google.adk import Agent
print("Google ADK installed successfully!")
```

---

## Core Concepts

### Agents

Agents are the primary building blocks in ADK. They encapsulate:
- **Model**: The LLM (e.g., Gemini) that powers the agent
- **Instructions**: How the agent should behave
- **Tools**: Capabilities the agent can use
- **Context**: Information about the current conversation

### Tools

Tools extend agent capabilities by providing access to:
- External APIs and services
- Database operations
- File system access
- Custom business logic
- Other agents

### Sessions

Sessions manage conversation state and history:
- User and session identification
- Message history
- Context preservation
- State management

### Runners

Runners orchestrate agent execution:
- Message processing
- Event generation
- Service coordination
- Error handling

---

## Your First Agent

Let's build a simple assistant to understand the basics.

### Step 1: Basic Agent

```python
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService

# Create a simple agent
assistant = Agent(
    name="my_first_agent",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant. Be friendly and concise."
)

# Set up session management
session_service = InMemorySessionService()

# Create a runner to execute the agent
runner = Runner(
    app_name="my_first_app",
    agent=assistant,
    session_service=session_service
)

# Test the agent
for event in runner.run(
    user_id="user123",
    session_id="session456", 
    new_message="Hello! What can you help me with?"
):
    if event.type == "model_response":
        print(f"Assistant: {event.data.get('text', '')}")
```

### Step 2: Understanding Events

The runner generates events during execution:

```python
for event in runner.run(
    user_id="user123",
    session_id="session456",
    new_message="Tell me a joke"
):
    print(f"Event Type: {event.type}")
    print(f"Event Data: {event.data}")
    
    # Common event types:
    if event.type == "model_request":
        print("Sending request to model...")
    elif event.type == "model_response":
        print(f"Model responded: {event.data.get('text', '')}")
    elif event.type == "tool_call":
        print(f"Tool called: {event.data.get('tool_name', '')}")
```

### Step 3: Adding Personality

```python
# Create an agent with more personality
creative_agent = Agent(
    name="creative_assistant",
    model="gemini-2.0-flash",
    instruction="""You are a creative writing assistant named Alex. You:
    - Love helping people with creative projects
    - Speak enthusiastically about storytelling
    - Provide specific, actionable advice
    - Always encourage creativity
    - Use emojis occasionally to be friendly ✨"""
)

runner = Runner(
    app_name="creative_app",
    agent=creative_agent,
    session_service=session_service
)

for event in runner.run(
    user_id="writer",
    session_id="writing_session",
    new_message="I'm stuck on my story. The main character needs motivation."
):
    if event.type == "model_response":
        print(f"Alex: {event.data.get('text', '')}")
```

---

## Working with Tools

Tools are what make agents truly powerful. Let's explore different types.

### Built-in Tools

#### Google Search

```python
from google.adk import Agent, Runner
from google.adk.tools import google_search
from google.adk.sessions import InMemorySessionService

# Create a research agent with Google Search
research_agent = Agent(
    name="researcher",
    model="gemini-2.0-flash",
    instruction="You are a research assistant. Use web search to find accurate, current information. Always cite your sources.",
    tools=[google_search]
)

session_service = InMemorySessionService()
runner = Runner(
    app_name="research_app",
    agent=research_agent,
    session_service=session_service
)

# Test with a research question
for event in runner.run(
    user_id="student",
    session_id="research_session",
    new_message="What are the latest developments in renewable energy?"
):
    if event.type == "model_response":
        print(f"Researcher: {event.data.get('text', '')}")
```

### Custom Function Tools

Turn any Python function into a tool:

```python
from google.adk.tools import FunctionTool
import random

def roll_dice(sides: int = 6, count: int = 1) -> dict:
    """Roll dice and return the results.
    
    Args:
        sides: Number of sides on each die (default: 6)
        count: Number of dice to roll (default: 1)
    
    Returns:
        Dictionary with roll results
    """
    if sides < 2:
        return {"error": "Dice must have at least 2 sides"}
    if count < 1 or count > 10:
        return {"error": "Can only roll 1-10 dice at once"}
    
    rolls = [random.randint(1, sides) for _ in range(count)]
    return {
        "rolls": rolls,
        "total": sum(rolls),
        "sides": sides,
        "count": count
    }

def calculate_tip(bill: float, percentage: float = 15.0) -> dict:
    """Calculate tip and total bill amount.
    
    Args:
        bill: Original bill amount
        percentage: Tip percentage (default: 15.0)
    
    Returns:
        Dictionary with tip calculations
    """
    if bill < 0:
        return {"error": "Bill amount cannot be negative"}
    
    tip = bill * (percentage / 100)
    total = bill + tip
    
    return {
        "original_bill": round(bill, 2),
        "tip_percentage": percentage,
        "tip_amount": round(tip, 2),
        "total_amount": round(total, 2)
    }

# Create tools from functions
dice_tool = FunctionTool(func=roll_dice)
tip_tool = FunctionTool(func=calculate_tip)

# Create a utility agent
utility_agent = Agent(
    name="utility_bot",
    model="gemini-2.0-flash",
    instruction="You are a helpful utility bot. You can roll dice for games and calculate tips for restaurants. Be friendly and explain what you're doing.",
    tools=[dice_tool, tip_tool]
)

# Test the utility agent
runner = Runner(
    app_name="utility_app",
    agent=utility_agent,
    session_service=InMemorySessionService()
)

for event in runner.run(
    user_id="gamer",
    session_id="game_session",
    new_message="Roll 2 six-sided dice for my board game, then calculate a 20% tip on a $45 bill"
):
    if event.type == "model_response":
        print(f"Utility Bot: {event.data.get('text', '')}")
```

### Custom Tool Classes

For more complex logic, create custom tool classes:

```python
from google.adk.tools import BaseTool, ToolContext
from typing import Any, Optional
from google.genai import types
import requests

class WeatherTool(BaseTool):
    """Custom tool for weather information."""
    
    def __init__(self, api_key: str):
        super().__init__(
            name="weather_lookup",
            description="Get current weather information for any city"
        )
        self.api_key = api_key
    
    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        city = args.get("city")
        if not city:
            return {"error": "City name is required"}
        
        # For demo purposes, return simulated data
        # In production, you'd call a real weather API
        weather_data = {
            "city": city,
            "temperature": "22°C (72°F)",
            "condition": "Partly cloudy",
            "humidity": "65%",
            "wind": "8 km/h NW",
            "feels_like": "24°C (75°F)"
        }
        
        return weather_data
    
    def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
        """Define the tool's parameters for the LLM."""
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "city": types.Schema(
                        type=types.Type.STRING,
                        description="The city name to get weather for"
                    )
                },
                required=["city"]
            )
        )

# Use the custom tool
weather_tool = WeatherTool(api_key="your_api_key_here")

weather_agent = Agent(
    name="weather_bot",
    model="gemini-2.0-flash",
    instruction="You are a weather assistant. Provide current weather information for any city requested. Be helpful and include relevant details.",
    tools=[weather_tool]
)

runner = Runner(
    app_name="weather_app",
    agent=weather_agent,
    session_service=InMemorySessionService()
)

for event in runner.run(
    user_id="traveler",
    session_id="weather_session",
    new_message="What's the weather like in Tokyo right now?"
):
    if event.type == "model_response":
        print(f"Weather Bot: {event.data.get('text', '')}")
```

---

## Multi-Agent Systems

ADK excels at building systems where multiple specialized agents work together.

### Basic Multi-Agent Setup

```python
from google.adk import Agent, Runner
from google.adk.tools import google_search, AgentTool
from google.adk.sessions import InMemorySessionService

# Create specialized agents
researcher = Agent(
    name="researcher",
    model="gemini-2.0-flash",
    instruction="""You are a research specialist. Your job is to:
    - Find accurate, up-to-date information using web search
    - Organize findings clearly with bullet points
    - Include relevant sources and statistics
    - Focus on facts and data""",
    tools=[google_search]
)

writer = Agent(
    name="writer",
    model="gemini-2.0-flash",
    instruction="""You are a content writer. Your job is to:
    - Take research data and create engaging articles
    - Use clear, accessible language
    - Structure content with headers and sections
    - Make complex topics easy to understand"""
)

# Create tools from the specialized agents
research_tool = AgentTool(agent=researcher)
writing_tool = AgentTool(agent=writer)

# Coordinator agent that uses the specialists
coordinator = Agent(
    name="content_coordinator",
    model="gemini-2.0-flash",
    instruction="""You coordinate content creation. Follow this process:
    
    1. First, use the research tool to gather information on the topic
    2. Then, use the writing tool to create engaging content from the research
    3. Present the final content to the user
    
    Always follow this sequence and explain what you're doing at each step.""",
    tools=[research_tool, writing_tool]
)

# Run the multi-agent system
runner = Runner(
    app_name="content_creation",
    agent=coordinator,
    session_service=InMemorySessionService()
)

for event in runner.run(
    user_id="content_creator",
    session_id="creation_session",
    new_message="Create an article about electric vehicles and their environmental impact"
):
    if event.type == "model_response":
        print(f"Coordinator: {event.data.get('text', '')}")
```

### Hierarchical Agent Structure

```python
# Create a hierarchical system with sub-agents
greeter = Agent(
    name="greeter",
    model="gemini-2.0-flash",
    instruction="You handle greetings and introductions. Be warm and welcoming. Ask users what they need help with."
)

support_agent = Agent(
    name="support_specialist",
    model="gemini-2.0-flash", 
    instruction="You handle customer support issues. Be helpful and solution-focused. Escalate complex issues when needed."
)

sales_agent = Agent(
    name="sales_specialist",
    model="gemini-2.0-flash",
    instruction="You handle sales inquiries. Be informative about products and pricing. Help customers make decisions."
)

# Main agent with sub-agents
main_agent = Agent(
    name="customer_service",
    model="gemini-2.0-flash",
    instruction="""You are a customer service coordinator. You have access to specialized team members:
    - Greeter: For welcoming new customers
    - Support: For technical issues and problems
    - Sales: For product information and purchases
    
    Route customers to the appropriate specialist based on their needs.""",
    sub_agents=[greeter, support_agent, sales_agent]
)

runner = Runner(
    app_name="customer_service",
    agent=main_agent,
    session_service=InMemorySessionService()
)

# Test the hierarchical system
test_messages = [
    "Hi there! I'm new here.",
    "I'm having trouble with my account login.",
    "I want to know about your premium plans."
]

for i, message in enumerate(test_messages, 1):
    print(f"\n=== Test {i}: {message} ===")
    for event in runner.run(
        user_id=f"customer_{i}",
        session_id=f"session_{i}",
        new_message=message
    ):
        if event.type == "model_response":
            print(f"Response: {event.data.get('text', '')}")
```

---

## Advanced Features

### Memory and State Management

```python
from google.adk import Agent, Runner
from google.adk.tools import FunctionTool, load_memory, preload_memory
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService

# Create a stateful agent that remembers user preferences
def save_preference(key: str, value: str) -> dict:
    """Save a user preference."""
    # In production, save to database
    print(f"Saving: {key} = {value}")
    return {"saved": True, "key": key, "value": value}

def get_preference(key: str) -> dict:
    """Get a user preference."""
    # Simulated preferences
    prefs = {"theme": "dark", "language": "en"}
    return {"key": key, "value": prefs.get(key, "not_set")}

stateful_agent = Agent(
    name="personal_assistant",
    model="gemini-2.0-flash",
    instruction="""You are a personal assistant that remembers user preferences.
    You can save and retrieve preferences, and you maintain conversation context.
    Always personalize responses based on saved preferences.""",
    tools=[
        FunctionTool(func=save_preference),
        FunctionTool(func=get_preference),
        load_memory,
        preload_memory
    ]
)

# Set up with memory service
memory_service = InMemoryMemoryService()
runner = Runner(
    app_name="personal_app",
    agent=stateful_agent,
    session_service=InMemorySessionService(),
    memory_service=memory_service
)

# Test conversation with memory
print("=== Setting preferences ===")
for event in runner.run(
    user_id="john",
    session_id="personal_session",
    new_message="Hi! Please save my theme preference as 'dark' and language as 'english'"
):
    if event.type == "model_response":
        print(f"Assistant: {event.data.get('text', '')}")

print("\n=== Using saved preferences ===")
for event in runner.run(
    user_id="john", 
    session_id="personal_session",
    new_message="What's my theme preference?"
):
    if event.type == "model_response":
        print(f"Assistant: {event.data.get('text', '')}")
```

### Callbacks and Monitoring

```python
from google.adk import Agent, Runner
from google.adk.tools import FunctionTool
from google.adk.sessions import InMemorySessionService

# Callback functions for monitoring
async def before_model_callback(context, request):
    """Called before each model request."""
    print(f"🤖 Sending request to model: {request.model}")
    return None

async def after_model_callback(context, response):
    """Called after each model response."""
    print(f"✅ Model responded with {len(response.text)} characters")
    return None

async def before_tool_callback(tool, args, context):
    """Called before each tool execution."""
    print(f"🔧 Executing tool: {tool.name} with args: {args}")
    return None

async def after_tool_callback(tool, args, context, result):
    """Called after each tool execution."""
    print(f"✅ Tool {tool.name} completed successfully")
    return None

def simple_calculator(operation: str, a: float, b: float) -> dict:
    """Perform basic math operations."""
    if operation == "add":
        return {"result": a + b}
    elif operation == "subtract":
        return {"result": a - b}
    elif operation == "multiply":
        return {"result": a * b}
    elif operation == "divide":
        return {"result": a / b if b != 0 else "Error: Division by zero"}
    else:
        return {"error": "Unknown operation"}

# Create agent with callbacks
monitored_agent = Agent(
    name="calculator",
    model="gemini-2.0-flash",
    instruction="You are a calculator bot. Use the calculator tool for math operations.",
    tools=[FunctionTool(func=simple_calculator)],
    before_model_callbacks=before_model_callback,
    after_model_callbacks=after_model_callback,
    before_tool_callbacks=before_tool_callback,
    after_tool_callbacks=after_tool_callback
)

runner = Runner(
    app_name="monitored_app",
    agent=monitored_agent,
    session_service=InMemorySessionService()
)

for event in runner.run(
    user_id="math_user",
    session_id="calc_session",
    new_message="Calculate 15 + 27 and then multiply the result by 3"
):
    if event.type == "model_response":
        print(f"Calculator: {event.data.get('text', '')}")
```

---

## Best Practices

### 1. Agent Design

**Keep Agents Focused**
```python
# Good: Specialized agent
search_agent = Agent(
    name="web_searcher",
    model="gemini-2.0-flash",
    instruction="You only search the web. Provide search results clearly and cite sources.",
    tools=[google_search]
)

# Avoid: Jack-of-all-trades agents
```

**Clear Instructions**
```python
# Good: Specific, actionable instructions
agent = Agent(
    name="customer_support",
    model="gemini-2.0-flash",
    instruction="""You are a customer support agent. Follow these steps:
    1. Greet the customer warmly
    2. Listen to their issue carefully
    3. Search the knowledge base for solutions
    4. Provide clear, step-by-step help
    5. Ask for confirmation that the issue is resolved
    
    Always be patient and professional."""
)
```

### 2. Tool Development

**Robust Error Handling**
```python
def safe_division(a: float, b: float) -> dict:
    """Safely divide two numbers."""
    try:
        if b == 0:
            return {"error": "Cannot divide by zero", "a": a, "b": b}
        result = a / b
        return {"result": result, "a": a, "b": b}
    except Exception as e:
        return {"error": f"Calculation error: {str(e)}", "a": a, "b": b}
```

**Input Validation**
```python
def validate_email(email: str) -> dict:
    """Validate an email address."""
    import re
    
    if not email or not isinstance(email, str):
        return {"valid": False, "error": "Email is required"}
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    is_valid = re.match(pattern, email) is not None
    
    return {
        "valid": is_valid,
        "email": email,
        "message": "Valid email" if is_valid else "Invalid email format"
    }
```

### 3. Performance Optimization

**Use Async Operations**
```python
import asyncio
import aiohttp

class AsyncAPITool(BaseTool):
    """Async tool for better performance."""
    
    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.example.com/data") as response:
                return await response.json()
```

**Efficient Tool Selection**
```python
# Good: Specific tools for specific tasks
math_agent = Agent(
    name="mathematician",
    model="gemini-2.0-flash",
    instruction="You solve math problems using the calculator tool.",
    tools=[calculator_tool]  # Only what's needed
)
```

### 4. Testing Strategies

**Unit Test Your Tools**
```python
import pytest
from your_tools import calculator_tool

def test_calculator_addition():
    result = calculator_tool.run_sync(
        args={"operation": "add", "a": 5, "b": 3},
        tool_context=mock_context
    )
    assert result["result"] == 8

def test_calculator_division_by_zero():
    result = calculator_tool.run_sync(
        args={"operation": "divide", "a": 10, "b": 0},
        tool_context=mock_context
    )
    assert "error" in result
```

**Integration Testing**
```python
def test_agent_with_tools():
    """Test agent behavior with tools."""
    agent = Agent(
        name="test_agent",
        model="gemini-2.0-flash",
        instruction="You are a test agent.",
        tools=[calculator_tool]
    )
    
    runner = Runner(
        app_name="test_app",
        agent=agent,
        session_service=InMemorySessionService()
    )
    
    # Test the interaction
    events = list(runner.run(
        user_id="test",
        session_id="test",
        new_message="Calculate 5 + 3"
    ))
    
    # Verify expected behavior
    assert any(event.type == "model_response" for event in events)
```

---

## Troubleshooting

### Common Issues

**1. Import Errors**
```python
# Problem: ModuleNotFoundError
# Solution: Check installation
pip list | grep google-adk

# Reinstall if needed
pip install --upgrade google-adk
```

**2. Authentication Issues**
```python
# Problem: Authentication failed
# Solution: Check credentials
import os
print(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))

# Or use gcloud auth
# gcloud auth application-default login
```

**3. Tool Not Being Called**
```python
# Problem: Agent doesn't use tools
# Common causes:
# - Missing tool registration
# - Unclear tool description
# - Conflicting instructions

# Solution: Clear tool descriptions
tool = FunctionTool(func=my_function)
print(tool.description)  # Should be clear and specific
```

**4. Memory/Session Issues**
```python
# Problem: Agent doesn't remember context
# Solution: Use proper session management
session_service = InMemorySessionService()  # For development
# or DatabaseSessionService()  # For production

# Ensure same user_id and session_id for continuity
```

### Debugging Tips

**Enable Logging**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed execution logs
```

**Event Monitoring**
```python
# Monitor all events for debugging
for event in runner.run(...):
    print(f"Event: {event.type} | Data: {event.data}")
```

**Tool Testing**
```python
# Test tools independently
from google.adk.tools import ToolContext

context = ToolContext(session_id="test", user_id="test")
result = await my_tool.run_async(
    args={"param": "value"},
    tool_context=context
)
print(result)
```

---

## Next Steps

Now that you understand the basics of Google ADK:

1. **Explore the [API Reference](api-reference.md)** for detailed documentation
2. **Check out [Examples](examples.md)** for more complex use cases
3. **Read the [Tools Guide](tools-guide.md)** for advanced tool development
4. **Join the community** for support and discussions

### Additional Resources

- [Google ADK Documentation](https://google.github.io/adk-docs)
- [Sample Projects](https://github.com/google/adk-samples)
- [ADK Web Interface](https://github.com/google/adk-web)
- [Community Forum](https://www.reddit.com/r/agentdevelopmentkit/)

Happy agent building! 🚀