# Google ADK API Reference

This document provides comprehensive API reference documentation for the Google Agent Development Kit (ADK).

## Table of Contents

- [Core Classes](#core-classes)
  - [Agent](#agent)
  - [Runner](#runner)
- [Tools](#tools)
  - [BaseTool](#basetool)
  - [Built-in Tools](#built-in-tools)
- [Models](#models)
  - [BaseLlm](#basellm)
  - [Gemini](#gemini)
- [Sessions](#sessions)
  - [Session](#session)
  - [BaseSessionService](#basesessionservice)
- [Examples](#examples)
- [Utilities](#utilities)

---

## Core Classes

### Agent

**Class**: `google.adk.agents.LlmAgent`

The main agent class for creating LLM-powered agents.

#### Constructor

```python
Agent(
    name: str,
    model: Union[str, BaseLlm] = '',
    instruction: Union[str, InstructionProvider] = '',
    global_instruction: Union[str, InstructionProvider] = '',
    tools: list[ToolUnion] = [],
    generate_content_config: Optional[types.GenerateContentConfig] = None,
    disallow_transfer_to_parent: bool = False,
    disallow_transfer_to_peers: bool = False,
    include_contents: Literal['default', 'none'] = 'default',
    input_schema: Optional[type[BaseModel]] = None,
    output_schema: Optional[type[BaseModel]] = None,
    output_key: Optional[str] = None,
    planner: Optional[BasePlanner] = None,
    flows: list[BaseLlmFlow] = [],
    code_executor: Optional[BaseCodeExecutor] = None,
    before_model_callbacks: Optional[BeforeModelCallback] = None,
    after_model_callbacks: Optional[AfterModelCallback] = None,
    before_tool_callbacks: Optional[BeforeToolCallback] = None,
    after_tool_callbacks: Optional[AfterToolCallback] = None,
    sub_agents: list[BaseAgent] = [],
    **kwargs
)
```

#### Parameters

- **name** (`str`): A unique identifier for the agent
- **model** (`Union[str, BaseLlm]`): The LLM model to use (e.g., "gemini-2.0-flash")
- **instruction** (`Union[str, InstructionProvider]`): Instructions guiding the agent's behavior
- **global_instruction** (`Union[str, InstructionProvider]`): Instructions for all agents in the agent tree
- **tools** (`list[ToolUnion]`): List of tools available to the agent
- **generate_content_config** (`Optional[types.GenerateContentConfig]`): Additional content generation configurations
- **disallow_transfer_to_parent** (`bool`): Prevents LLM-controlled transfer to parent agent
- **disallow_transfer_to_peers** (`bool`): Prevents LLM-controlled transfer to peer agents
- **include_contents** (`Literal['default', 'none']`): Controls content inclusion in model requests
- **input_schema** (`Optional[type[BaseModel]]`): Schema for input validation when used as a tool
- **output_schema** (`Optional[type[BaseModel]]`): Schema for output validation
- **output_key** (`Optional[str]`): Key in session state to store agent output
- **planner** (`Optional[BasePlanner]`): Planning component for advanced reasoning
- **flows** (`list[BaseLlmFlow]`): Flow configurations for the agent
- **code_executor** (`Optional[BaseCodeExecutor]`): Code execution component
- **before_model_callbacks** (`Optional[BeforeModelCallback]`): Callbacks executed before model invocation
- **after_model_callbacks** (`Optional[AfterModelCallback]`): Callbacks executed after model invocation
- **before_tool_callbacks** (`Optional[BeforeToolCallback]`): Callbacks executed before tool invocation
- **after_tool_callbacks** (`Optional[AfterToolCallback]`): Callbacks executed after tool invocation
- **sub_agents** (`list[BaseAgent]`): Child agents for multi-agent systems

#### Example Usage

```python
from google.adk import Agent
from google.adk.tools import google_search

# Simple agent with Google Search
search_agent = Agent(
    name="search_assistant",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant that can search the web.",
    tools=[google_search]
)

# Multi-agent system
coordinator = Agent(
    name="coordinator",
    model="gemini-2.0-flash",
    instruction="Coordinate tasks between specialized agents.",
    sub_agents=[
        Agent(name="researcher", model="gemini-2.0-flash", tools=[google_search]),
        Agent(name="writer", model="gemini-2.0-flash", instruction="Write clear summaries")
    ]
)
```

---

### Runner

**Class**: `google.adk.runners.Runner`

The Runner class manages agent execution within sessions, handling message processing, event generation, and service interactions.

#### Constructor

```python
Runner(
    *,
    app_name: str,
    agent: BaseAgent,
    plugins: Optional[List[BasePlugin]] = None,
    artifact_service: Optional[BaseArtifactService] = None,
    session_service: BaseSessionService,
    memory_service: Optional[BaseMemoryService] = None,
    credential_service: Optional[BaseCredentialService] = None,
)
```

#### Parameters

- **app_name** (`str`): Application name for the runner
- **agent** (`BaseAgent`): Root agent to execute
- **plugins** (`Optional[List[BasePlugin]]`): List of plugins for extended functionality
- **artifact_service** (`Optional[BaseArtifactService]`): Service for artifact storage and management
- **session_service** (`BaseSessionService`): Service for session management
- **memory_service** (`Optional[BaseMemoryService]`): Service for memory management
- **credential_service** (`Optional[BaseCredentialService]`): Service for credential management

#### Methods

##### `run()`

```python
def run(
    self,
    *,
    user_id: str,
    session_id: str,
    new_message: types.Content,
    run_config: RunConfig = RunConfig(),
) -> Generator[Event, None, None]
```

Synchronous method to run the agent (for local testing).

**Parameters:**
- **user_id** (`str`): User identifier
- **session_id** (`str`): Session identifier
- **new_message** (`types.Content`): New message to process
- **run_config** (`RunConfig`): Configuration for the run

**Returns:** Generator yielding execution events

##### `run_async()`

```python
async def run_async(
    self,
    *,
    user_id: str,
    session_id: str,
    new_message: types.Content,
    state_delta: Optional[dict[str, Any]] = None,
    run_config: RunConfig = RunConfig(),
) -> AsyncGenerator[Event, None]
```

Asynchronous method to run the agent (recommended for production).

**Parameters:**
- **user_id** (`str`): User identifier
- **session_id** (`str`): Session identifier
- **new_message** (`types.Content`): New message to process
- **state_delta** (`Optional[dict[str, Any]]`): State changes to apply
- **run_config** (`RunConfig`): Configuration for the run

**Returns:** AsyncGenerator yielding execution events

#### Example Usage

```python
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService

# Create agent and runner
agent = Agent(name="assistant", model="gemini-2.0-flash")
session_service = InMemorySessionService()

runner = Runner(
    app_name="my_app",
    agent=agent,
    session_service=session_service
)

# Run the agent
for event in runner.run(
    user_id="user123",
    session_id="session456",
    new_message="Hello, how can you help me?"
):
    print(f"Event: {event.type} - {event.data}")
```

---

## Tools

### BaseTool

**Class**: `google.adk.tools.BaseTool`

Abstract base class for all tools in the ADK framework.

#### Constructor

```python
BaseTool(
    *,
    name: str,
    description: str,
    is_long_running: bool = False,
    custom_metadata: Optional[dict[str, Any]] = None,
)
```

#### Parameters

- **name** (`str`): Unique name for the tool
- **description** (`str`): Description of the tool's functionality
- **is_long_running** (`bool`): Whether the tool performs long-running operations
- **custom_metadata** (`Optional[dict[str, Any]]`): Additional tool-specific metadata

#### Abstract Methods

##### `run_async()`

```python
async def run_async(
    self, *, args: dict[str, Any], tool_context: ToolContext
) -> Any
```

Main execution method for the tool.

**Parameters:**
- **args** (`dict[str, Any]`): Tool arguments
- **tool_context** (`ToolContext`): Context containing session information

**Returns:** Tool execution result

#### Example Implementation

```python
from google.adk.tools import BaseTool
from google.adk.tools import ToolContext

class CustomCalculatorTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Performs basic mathematical calculations",
        )
    
    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        operation = args.get("operation")
        a = args.get("a")
        b = args.get("b")
        
        if operation == "add":
            return {"result": a + b}
        elif operation == "multiply":
            return {"result": a * b}
        else:
            return {"error": "Unsupported operation"}
```

### Built-in Tools

#### Google Search

```python
from google.adk.tools import google_search

# Use in agent
agent = Agent(
    name="search_agent",
    model="gemini-2.0-flash",
    tools=[google_search]
)
```

Performs web searches using Google Search.

#### Enterprise Search

```python
from google.adk.tools import enterprise_web_search

# Use in agent
agent = Agent(
    name="enterprise_agent",
    model="gemini-2.0-flash",
    tools=[enterprise_web_search]
)
```

Searches within enterprise content using Google Enterprise Search.

#### URL Context

```python
from google.adk.tools import url_context

# Use in agent
agent = Agent(
    name="web_agent",
    model="gemini-2.0-flash",
    tools=[url_context]
)
```

Extracts and analyzes content from web URLs.

#### Function Tool

```python
from google.adk.tools import FunctionTool

def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 75°F"

weather_tool = FunctionTool(func=get_weather)

agent = Agent(
    name="weather_agent",
    model="gemini-2.0-flash",
    tools=[weather_tool]
)
```

Wraps Python functions as tools for agent use.

#### Agent Tool

```python
from google.adk.tools import AgentTool

sub_agent = Agent(name="specialist", model="gemini-2.0-flash")
agent_tool = AgentTool(agent=sub_agent)

main_agent = Agent(
    name="main_agent",
    model="gemini-2.0-flash",
    tools=[agent_tool]
)
```

Allows agents to invoke other agents as tools.

---

## Models

### BaseLlm

**Class**: `google.adk.models.BaseLlm`

Abstract base class for all language models.

### Gemini

**Class**: `google.adk.models.Gemini`

Google's Gemini model implementation.

#### Supported Models

```python
from google.adk.models import Gemini

# Get list of supported models
supported_models = Gemini.supported_models()
print(supported_models)
```

#### Example Usage

```python
from google.adk.models import Gemini

# Create Gemini model instance
gemini = Gemini(model_name="gemini-2.0-flash")

# Use with agent
agent = Agent(
    name="gemini_agent",
    model=gemini,
    instruction="You are a helpful assistant."
)
```

---

## Sessions

### Session

**Class**: `google.adk.sessions.Session`

Represents a conversation session between a user and agents.

#### Attributes

- **session_id** (`str`): Unique session identifier
- **user_id** (`str`): User identifier
- **messages** (`list`): Conversation history
- **state** (`dict`): Session state data

### BaseSessionService

**Class**: `google.adk.sessions.BaseSessionService`

Abstract base class for session management services.

#### Key Methods

##### `get_session()`

```python
async def get_session(
    self, *, app_name: str, user_id: str, session_id: str
) -> Session
```

Retrieves or creates a session.

##### `save_session()`

```python
async def save_session(self, *, session: Session) -> None
```

Persists session data.

#### Implementations

- **InMemorySessionService**: In-memory session storage
- **DatabaseSessionService**: Database-backed session storage
- **VertexAiSessionService**: Vertex AI-powered session storage

#### Example Usage

```python
from google.adk.sessions import InMemorySessionService

session_service = InMemorySessionService()

# Use with runner
runner = Runner(
    app_name="my_app",
    agent=agent,
    session_service=session_service
)
```

---

## Examples

### Example

**Class**: `google.adk.examples.Example`

Represents a training or evaluation example for agents.

### BaseExampleProvider

**Class**: `google.adk.examples.BaseExampleProvider`

Abstract base class for providing examples to agents.

### VertexAiExampleStore

**Class**: `google.adk.examples.VertexAiExampleStore`

Vertex AI-backed example storage service.

---

## Utilities

### Feature Decorators

```python
from google.adk.utils.feature_decorator import experimental

@experimental
class MyExperimentalFeature:
    pass

@experimental
def my_experimental_function():
    pass
```

Mark features as experimental or work-in-progress.

### Model Name Utilities

```python
from google.adk.utils.model_name_utils import extract_model_name, is_gemini_model

model_name = extract_model_name("projects/my-project/locations/us/models/gemini-2.0-flash")
is_gemini = is_gemini_model("gemini-2.0-flash")
```

Utilities for working with model names and identification.

### Variant Utils

```python
from google.adk.utils.variant_utils import get_google_llm_variant, GoogleLLMVariant

variant = get_google_llm_variant()
```

Utilities for working with Google LLM variants.

---

## Type Aliases and Callbacks

### Callback Types

```python
from google.adk.agents.llm_agent import (
    BeforeModelCallback,
    AfterModelCallback,
    BeforeToolCallback,
    AfterToolCallback,
    InstructionProvider
)
```

Type aliases for various callback functions used throughout the framework.

### Tool Types

```python
from google.adk.agents.llm_agent import ToolUnion
```

Union type for tools that can be functions, BaseTool instances, or BaseToolset instances.

---

## Error Handling

The ADK framework includes custom exception classes for better error handling:

```python
from google.adk.errors import ADKError

try:
    # ADK operations
    pass
except ADKError as e:
    print(f"ADK Error: {e}")
```

---

This API reference provides comprehensive coverage of the Google ADK's public interfaces. For more detailed examples and tutorials, refer to the [User Guide](user-guide.md) and [Examples](examples.md) documentation.