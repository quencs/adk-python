# Google ADK Tools Guide

This guide provides comprehensive documentation for all tools available in the Google Agent Development Kit (ADK), including built-in tools, custom tool creation, and integration patterns.

## Table of Contents

- [Overview](#overview)
- [Built-in Tools](#built-in-tools)
  - [Search Tools](#search-tools)
  - [Utility Tools](#utility-tools)
  - [Integration Tools](#integration-tools)
  - [Memory and Artifact Tools](#memory-and-artifact-tools)
- [Custom Tool Development](#custom-tool-development)
- [Tool Configuration](#tool-configuration)
- [Best Practices](#best-practices)

---

## Overview

Tools in ADK extend agent capabilities by providing access to external services, APIs, functions, and data sources. The framework supports multiple tool types:

1. **Built-in Tools**: Pre-implemented tools for common tasks
2. **Function Tools**: Python functions wrapped as tools
3. **Custom Tools**: Tools extending `BaseTool` for complex logic
4. **OpenAPI Tools**: REST API endpoints exposed as tools
5. **Agent Tools**: Other agents used as tools
6. **Toolsets**: Collections of related tools

---

## Built-in Tools

### Search Tools

#### Google Search

**Import**: `from google.adk.tools import google_search`

Built-in Google Search integration for Gemini models.

```python
from google.adk import Agent
from google.adk.tools import google_search

agent = Agent(
    name="search_agent",
    model="gemini-2.0-flash",
    instruction="Search the web to answer user questions.",
    tools=[google_search]
)
```

**Features:**
- Automatic integration with Gemini 2.0+ models
- No API keys required
- Built-in result formatting
- Optimized for conversational search

**Limitations:**
- Only works with Gemini models
- Cannot be combined with other tools in Gemini 1.x models

#### Enterprise Search

**Import**: `from google.adk.tools import enterprise_web_search`

Search within enterprise content using Google Enterprise Search.

```python
from google.adk.tools import enterprise_web_search

enterprise_agent = Agent(
    name="enterprise_assistant",
    model="gemini-2.0-flash",
    instruction="Help users find information from company resources.",
    tools=[enterprise_web_search]
)
```

**Use Cases:**
- Internal knowledge base search
- Company document retrieval
- Enterprise content discovery

#### Vertex AI Search

**Class**: `google.adk.tools.VertexAiSearchTool`

Custom search using Vertex AI Search service.

```python
from google.adk.tools import VertexAiSearchTool

search_tool = VertexAiSearchTool(
    project_id="your-project",
    location="us-central1",
    data_store_id="your-datastore"
)

agent = Agent(
    name="custom_search_agent",
    model="gemini-2.0-flash",
    tools=[search_tool]
)
```

### Utility Tools

#### Function Tool

**Class**: `google.adk.tools.FunctionTool`

Wraps Python functions as tools for agent use.

```python
from google.adk.tools import FunctionTool

def calculate_tip(bill_amount: float, tip_percentage: float = 15.0) -> dict:
    """Calculate tip amount and total bill.
    
    Args:
        bill_amount: The original bill amount
        tip_percentage: Tip percentage (default: 15.0)
    
    Returns:
        Dictionary with tip amount and total
    """
    tip_amount = bill_amount * (tip_percentage / 100)
    total = bill_amount + tip_amount
    return {
        "tip_amount": round(tip_amount, 2),
        "total": round(total, 2),
        "tip_percentage": tip_percentage
    }

# Create tool from function
tip_calculator = FunctionTool(func=calculate_tip)

# Use in agent
agent = Agent(
    name="tip_calculator",
    model="gemini-2.0-flash",
    instruction="Help users calculate tips and totals.",
    tools=[tip_calculator]
)
```

**Function Requirements:**
- Type hints for parameters (recommended)
- Docstring describing functionality
- JSON-serializable return values
- No side effects that affect tool context

#### URL Context Tool

**Import**: `from google.adk.tools import url_context`

Extracts and analyzes content from web URLs.

```python
from google.adk.tools import url_context

web_agent = Agent(
    name="web_analyzer",
    model="gemini-2.0-flash",
    instruction="Analyze web pages and answer questions about their content.",
    tools=[url_context]
)
```

**Capabilities:**
- HTML content extraction
- Text summarization
- Link analysis
- Metadata extraction

#### Exit Loop Tool

**Import**: `from google.adk.tools import exit_loop`

Provides a way to exit loops in agent workflows.

```python
from google.adk.tools import exit_loop
from google.adk.agents import LoopAgent

loop_agent = LoopAgent(
    name="task_loop",
    model="gemini-2.0-flash",
    instruction="Process tasks in a loop. Use exit_loop when done.",
    tools=[exit_loop]
)
```

#### User Choice Tool

**Import**: `from google.adk.tools import get_user_choice`

Allows agents to present choices to users and get their selection.

```python
from google.adk.tools import get_user_choice

interactive_agent = Agent(
    name="choice_agent",
    model="gemini-2.0-flash",
    instruction="Present options to users and get their preferences.",
    tools=[get_user_choice]
)
```

### Integration Tools

#### Agent Tool

**Class**: `google.adk.tools.AgentTool`

Enables agents to invoke other agents as tools.

```python
from google.adk.tools import AgentTool

# Create specialized agents
researcher = Agent(
    name="researcher",
    model="gemini-2.0-flash",
    instruction="Research topics and gather information.",
    tools=[google_search]
)

writer = Agent(
    name="writer",
    model="gemini-2.0-flash",
    instruction="Write clear, engaging content based on research."
)

# Create tools from agents
research_tool = AgentTool(agent=researcher)
writing_tool = AgentTool(agent=writer)

# Coordinator agent using other agents as tools
coordinator = Agent(
    name="coordinator",
    model="gemini-2.0-flash",
    instruction="Coordinate research and writing tasks.",
    tools=[research_tool, writing_tool]
)
```

**Benefits:**
- Modular agent composition
- Reusable specialized agents
- Clear separation of concerns
- Hierarchical agent structures

#### OpenAPI Tool

**Package**: `google.adk.tools.openapi_tool`

Automatically generates tools from OpenAPI specifications.

```python
from google.adk.tools.openapi_tool import create_openapi_tools

# Create tools from OpenAPI spec
api_tools = create_openapi_tools(
    openapi_spec_url="https://api.example.com/openapi.json",
    base_url="https://api.example.com",
    headers={"Authorization": "Bearer your-token"}
)

agent = Agent(
    name="api_agent",
    model="gemini-2.0-flash",
    instruction="Interact with the API to fulfill user requests.",
    tools=api_tools
)
```

#### BigQuery Tool

**Package**: `google.adk.tools.bigquery`

Execute SQL queries against BigQuery datasets.

```python
from google.adk.tools.bigquery import BigQueryTool

bq_tool = BigQueryTool(
    project_id="your-project",
    credentials_path="path/to/credentials.json"
)

data_agent = Agent(
    name="data_analyst",
    model="gemini-2.0-flash",
    instruction="Answer questions by querying BigQuery datasets.",
    tools=[bq_tool]
)
```

#### MCP (Model Context Protocol) Tools

**Class**: `google.adk.tools.MCPToolset` (Python 3.10+)

Integrate MCP-compatible tools and services.

```python
from google.adk.tools import MCPToolset

mcp_toolset = MCPToolset(
    server_config={
        "command": "python",
        "args": ["-m", "mcp_server"],
        "env": {}
    }
)

agent = Agent(
    name="mcp_agent",
    model="gemini-2.0-flash",
    instruction="Use MCP tools to assist users.",
    tools=[mcp_toolset]
)
```

### Memory and Artifact Tools

#### Load Memory Tool

**Import**: `from google.adk.tools import load_memory`

Loads conversation memory into the current context.

```python
from google.adk.tools import load_memory

memory_agent = Agent(
    name="memory_agent",
    model="gemini-2.0-flash",
    instruction="Use conversation history to provide contextual responses.",
    tools=[load_memory]
)
```

#### Preload Memory Tool

**Import**: `from google.adk.tools import preload_memory`

Preloads memory at the start of conversations.

```python
from google.adk.tools import preload_memory

contextual_agent = Agent(
    name="contextual_agent",
    model="gemini-2.0-flash",
    instruction="Always maintain conversation context.",
    tools=[preload_memory]
)
```

#### Load Artifacts Tool

**Import**: `from google.adk.tools import load_artifacts`

Loads and manages artifacts (files, documents, data).

```python
from google.adk.tools import load_artifacts

document_agent = Agent(
    name="document_agent",
    model="gemini-2.0-flash",
    instruction="Work with documents and files provided by users.",
    tools=[load_artifacts]
)
```

---

## Custom Tool Development

### Creating a Basic Custom Tool

```python
from google.adk.tools import BaseTool, ToolContext
from typing import Any, Optional
from google.genai import types

class WeatherTool(BaseTool):
    """Custom weather information tool."""
    
    def __init__(self, api_key: str):
        super().__init__(
            name="weather_tool",
            description="Get current weather information for a given location",
        )
        self.api_key = api_key
    
    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        location = args.get("location")
        if not location:
            return {"error": "Location is required"}
        
        # Simulate API call
        weather_data = {
            "location": location,
            "temperature": "22°C",
            "condition": "Sunny",
            "humidity": "45%"
        }
        
        return weather_data
    
    def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "location": types.Schema(
                        type=types.Type.STRING,
                        description="The location to get weather for"
                    )
                },
                required=["location"]
            )
        )

# Usage
weather_tool = WeatherTool(api_key="your-api-key")

agent = Agent(
    name="weather_agent",
    model="gemini-2.0-flash",
    instruction="Provide weather information using the weather tool.",
    tools=[weather_tool]
)
```

### Advanced Custom Tool with Streaming

```python
from google.adk.tools import BaseTool, ToolContext
from typing import Any, AsyncGenerator

class StreamingDataTool(BaseTool):
    """Tool that streams data progressively."""
    
    def __init__(self):
        super().__init__(
            name="streaming_data",
            description="Streams large datasets progressively",
            is_long_running=True
        )
    
    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        query = args.get("query", "")
        
        # Simulate streaming data
        async def stream_data():
            for i in range(5):
                yield f"Data chunk {i+1} for query: {query}\n"
                await asyncio.sleep(0.5)  # Simulate processing time
        
        result = ""
        async for chunk in stream_data():
            result += chunk
        
        return {"data": result, "total_chunks": 5}
```

### Custom Tool with Authentication

```python
from google.adk.tools import BaseAuthenticatedTool
from google.adk.auth import AuthToolArguments

class AuthenticatedAPITool(BaseAuthenticatedTool):
    """Tool requiring authentication."""
    
    def __init__(self):
        super().__init__(
            name="authenticated_api",
            description="Access authenticated API endpoints",
            auth_args=AuthToolArguments(
                service_name="example_api",
                scopes=["https://www.example.com/auth/api"]
            )
        )
    
    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        # Access authenticated resources
        credentials = self.get_credentials(tool_context)
        
        # Make authenticated API call
        response = await self.make_authenticated_request(
            url="https://api.example.com/data",
            headers={"Authorization": f"Bearer {credentials.token}"},
            params=args
        )
        
        return response
```

---

## Tool Configuration

### Tool Args Configuration

```python
from google.adk.tools.tool_configs import ToolConfig, ToolArgsConfig

# Configure tool with specific arguments
tool_config = ToolConfig(
    name="custom_tool",
    config=ToolArgsConfig(
        custom_arg1="value1",
        custom_arg2="value2"
    )
)

# Use in agent configuration
agent = Agent(
    name="configured_agent",
    model="gemini-2.0-flash",
    tools=[tool_config]
)
```

### Toolset Creation

```python
from google.adk.tools import BaseToolset, BaseTool

class CalculationToolset(BaseToolset):
    """Toolset for mathematical calculations."""
    
    def __init__(self):
        super().__init__(
            name="calculation_toolset",
            description="Mathematical calculation tools"
        )
    
    async def get_tools(self, ctx) -> list[BaseTool]:
        return [
            FunctionTool(func=self.add),
            FunctionTool(func=self.multiply),
            FunctionTool(func=self.divide)
        ]
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
    
    def divide(self, a: float, b: float) -> float:
        """Divide two numbers."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

# Usage
calc_toolset = CalculationToolset()

agent = Agent(
    name="math_agent",
    model="gemini-2.0-flash",
    instruction="Perform mathematical calculations.",
    tools=[calc_toolset]
)
```

---

## Best Practices

### Tool Design

1. **Single Responsibility**: Each tool should have one clear purpose
2. **Descriptive Names**: Use clear, descriptive names and descriptions
3. **Error Handling**: Implement robust error handling and validation
4. **Type Safety**: Use type hints and validate inputs
5. **Documentation**: Provide comprehensive docstrings

### Performance Optimization

1. **Async Operations**: Use async/await for I/O operations
2. **Caching**: Implement caching for expensive operations
3. **Resource Management**: Properly manage connections and resources
4. **Timeouts**: Set appropriate timeouts for external calls

### Security Considerations

1. **Input Validation**: Validate all inputs thoroughly
2. **Authentication**: Use proper authentication mechanisms
3. **Rate Limiting**: Implement rate limiting for API calls
4. **Data Sanitization**: Sanitize outputs to prevent injection attacks

### Testing Tools

```python
import pytest
from google.adk.tools import ToolContext
from your_module import WeatherTool

@pytest.mark.asyncio
async def test_weather_tool():
    tool = WeatherTool(api_key="test-key")
    context = ToolContext(session_id="test", user_id="test-user")
    
    result = await tool.run_async(
        args={"location": "New York"},
        tool_context=context
    )
    
    assert "location" in result
    assert result["location"] == "New York"
    assert "temperature" in result
```

### Tool Composition

```python
# Combine multiple tools effectively
comprehensive_agent = Agent(
    name="comprehensive_assistant",
    model="gemini-2.0-flash",
    instruction="Help users with various tasks using available tools.",
    tools=[
        google_search,           # Web search
        url_context,            # Web page analysis
        FunctionTool(func=calculate_tip),  # Custom calculations
        load_memory,            # Memory management
        weather_tool,           # Custom weather info
    ]
)
```

---

This tools guide provides comprehensive documentation for using and creating tools in the Google ADK framework. Tools are the primary way to extend agent capabilities and integrate with external services and data sources.