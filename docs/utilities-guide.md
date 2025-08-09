# Google ADK Utilities Guide

This guide documents the utility functions and helper classes available in the Google Agent Development Kit (ADK).

## Table of Contents

- [Overview](#overview)
- [Model Utilities](#model-utilities)
- [Feature Decorators](#feature-decorators)
- [Instruction Utilities](#instruction-utilities)
- [Variant Utilities](#variant-utilities)
- [Usage Examples](#usage-examples)

---

## Overview

ADK provides several utility modules to help with common tasks in agent development:

- **Model Name Utilities**: Functions for parsing and validating model names
- **Feature Decorators**: Decorators for marking experimental features
- **Instruction Utilities**: Functions for processing agent instructions and session state
- **Variant Utilities**: Utilities for working with different Google LLM variants

---

## Model Utilities

**Module**: `google.adk.utils.model_name_utils`

Utilities for model name validation and parsing.

### `extract_model_name(model_string: str) -> str`

Extract the actual model name from either simple or path-based format.

```python
from google.adk.utils.model_name_utils import extract_model_name

# Simple model name
model_name = extract_model_name("gemini-2.0-flash")
print(model_name)  # Output: "gemini-2.0-flash"

# Path-based model name
path_model = "projects/my-project/locations/us-central1/publishers/google/models/gemini-2.0-flash-001"
model_name = extract_model_name(path_model)
print(model_name)  # Output: "gemini-2.0-flash-001"
```

**Parameters:**
- `model_string`: Either a simple model name like "gemini-2.0-flash" or a path-based model name

**Returns:**
- The extracted model name

### `is_gemini_model(model_string: Optional[str]) -> bool`

Check if the model is a Gemini model.

```python
from google.adk.utils.model_name_utils import is_gemini_model

# Test various model names
print(is_gemini_model("gemini-2.0-flash"))  # True
print(is_gemini_model("gemini-1.5-pro"))    # True
print(is_gemini_model("gpt-4"))             # False
print(is_gemini_model(None))                # False
```

**Parameters:**
- `model_string`: Model name to check (can be None)

**Returns:**
- `True` if it's a Gemini model, `False` otherwise

### `is_gemini_1_model(model_string: Optional[str]) -> bool`

Check if the model is a Gemini 1.x model.

```python
from google.adk.utils.model_name_utils import is_gemini_1_model

print(is_gemini_1_model("gemini-1.5-pro"))     # True
print(is_gemini_1_model("gemini-1.0-pro"))     # True
print(is_gemini_1_model("gemini-2.0-flash"))   # False
```

**Parameters:**
- `model_string`: Model name to check

**Returns:**
- `True` if it's a Gemini 1.x model, `False` otherwise

### `is_gemini_2_model(model_string: Optional[str]) -> bool`

Check if the model is a Gemini 2.x model.

```python
from google.adk.utils.model_name_utils import is_gemini_2_model

print(is_gemini_2_model("gemini-2.0-flash"))   # True
print(is_gemini_2_model("gemini-2.5-pro"))     # True
print(is_gemini_2_model("gemini-1.5-pro"))     # False
```

**Parameters:**
- `model_string`: Model name to check

**Returns:**
- `True` if it's a Gemini 2.x model, `False` otherwise

### Complete Model Utilities Example

```python
from google.adk.utils.model_name_utils import (
    extract_model_name,
    is_gemini_model,
    is_gemini_1_model,
    is_gemini_2_model
)

def analyze_model(model_string: str):
    """Analyze a model string and provide information about it."""
    model_name = extract_model_name(model_string)
    
    print(f"Original: {model_string}")
    print(f"Extracted: {model_name}")
    print(f"Is Gemini: {is_gemini_model(model_string)}")
    print(f"Is Gemini 1.x: {is_gemini_1_model(model_string)}")
    print(f"Is Gemini 2.x: {is_gemini_2_model(model_string)}")
    print("---")

# Test with different model formats
models = [
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "projects/my-project/locations/us/publishers/google/models/gemini-2.0-flash-001",
    "gpt-4",
    "claude-3"
]

for model in models:
    analyze_model(model)
```

---

## Feature Decorators

**Module**: `google.adk.utils.feature_decorator`

Decorators for marking features as experimental or work-in-progress.

### `@experimental`

Mark classes, functions, or methods as experimental.

```python
from google.adk.utils.feature_decorator import experimental

@experimental
class ExperimentalFeature:
    """This is an experimental feature that may change in future versions."""
    
    def __init__(self):
        self.name = "experimental"
    
    @experimental
    def experimental_method(self):
        """This method is experimental."""
        return "experimental result"

@experimental
def experimental_function():
    """This function is experimental and may be removed or changed."""
    return "experimental output"

# Usage
feature = ExperimentalFeature()
result = feature.experimental_method()
output = experimental_function()
```

**Purpose:**
- Warn users that the feature is experimental
- Indicate that the API may change in future versions
- Help track which features are stable vs. experimental

### `@working_in_progress`

Mark features as work-in-progress (WIP).

```python
from google.adk.utils.feature_decorator import working_in_progress

@working_in_progress
class WIPFeature:
    """This feature is still being developed."""
    pass

@working_in_progress
def wip_function():
    """This function is work-in-progress."""
    pass
```

### Custom Feature Decorators

You can also create custom feature decorators:

```python
from google.adk.utils.feature_decorator import _make_feature_decorator

# Create a custom decorator for beta features
beta = _make_feature_decorator(
    flag_name="beta",
    warning_message="This feature is in beta and may have limitations."
)

@beta
class BetaFeature:
    """This is a beta feature."""
    pass
```

---

## Instruction Utilities

**Module**: `google.adk.utils.instructions_utils`

Utilities for processing agent instructions and session state injection.

### `inject_session_state(instruction: str, session_state: dict) -> str`

Inject session state variables into instruction templates.

```python
from google.adk.utils.instructions_utils import inject_session_state

# Template with placeholders
instruction_template = """
You are a helpful assistant named {{agent_name}}.
The current user is {{user_name}} and their preferred language is {{language}}.
Today's date is {{current_date}}.
"""

# Session state data
session_state = {
    "agent_name": "Alex",
    "user_name": "John",
    "language": "English",
    "current_date": "2024-01-15"
}

# Inject state into instruction
final_instruction = inject_session_state(instruction_template, session_state)
print(final_instruction)
# Output:
# You are a helpful assistant named Alex.
# The current user is John and their preferred language is English.
# Today's date is 2024-01-15.
```

**Parameters:**
- `instruction`: Instruction template with `{{variable}}` placeholders
- `session_state`: Dictionary of state variables

**Returns:**
- Instruction with placeholders replaced by state values

### Advanced Instruction Processing

```python
from google.adk.utils.instructions_utils import inject_session_state

# Complex instruction template
instruction = """
Hello {{user_name}}! I'm {{assistant_name}}, your {{assistant_type}} assistant.

Current Context:
- Session ID: {{session_id}}
- User Preferences: {{preferences}}
- Current Task: {{current_task}}

{{#if has_history}}
I remember our previous conversations.
{{/if}}

How can I help you today?
"""

# Rich session state
session_state = {
    "user_name": "Sarah",
    "assistant_name": "Ada",
    "assistant_type": "AI coding",
    "session_id": "sess_12345",
    "preferences": "Dark theme, Python focus",
    "current_task": "Code review",
    "has_history": True
}

personalized_instruction = inject_session_state(instruction, session_state)
```

---

## Variant Utilities

**Module**: `google.adk.utils.variant_utils`

Utilities for working with different Google LLM variants.

### `GoogleLLMVariant`

Enum for Google LLM variants.

```python
from google.adk.utils.variant_utils import GoogleLLMVariant

# Available variants
print(GoogleLLMVariant.VERTEX_AI.value)   # "VERTEX_AI"
print(GoogleLLMVariant.GEMINI_API.value)  # "GEMINI_API"
```

**Variants:**
- `VERTEX_AI`: For using credentials from Google Vertex AI
- `GEMINI_API`: For using API Key from Google AI Studio

### `get_google_llm_variant() -> str`

Get the current Google LLM variant based on environment variables.

```python
import os
from google.adk.utils.variant_utils import get_google_llm_variant, GoogleLLMVariant

# Default behavior (returns GEMINI_API)
variant = get_google_llm_variant()
print(variant)  # GoogleLLMVariant.GEMINI_API

# Set environment variable to use Vertex AI
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
variant = get_google_llm_variant()
print(variant)  # GoogleLLMVariant.VERTEX_AI

# Alternative ways to enable Vertex AI
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
variant = get_google_llm_variant()
print(variant)  # GoogleLLMVariant.VERTEX_AI
```

**Environment Variable:**
- `GOOGLE_GENAI_USE_VERTEXAI`: Set to "true" or "1" to use Vertex AI variant

**Returns:**
- `GoogleLLMVariant.VERTEX_AI` if environment variable is set
- `GoogleLLMVariant.GEMINI_API` otherwise (default)

---

## Usage Examples

### Model-Aware Agent Creation

```python
from google.adk import Agent
from google.adk.utils.model_name_utils import is_gemini_2_model
from google.adk.tools import google_search

def create_smart_agent(model_name: str) -> Agent:
    """Create an agent with model-specific optimizations."""
    
    # Base configuration
    agent_config = {
        "name": "smart_agent",
        "model": model_name,
        "instruction": "You are a helpful assistant."
    }
    
    # Add Google Search for Gemini 2.x models
    if is_gemini_2_model(model_name):
        agent_config["tools"] = [google_search]
        agent_config["instruction"] += " You can search the web for current information."
    
    return Agent(**agent_config)

# Usage
agent1 = create_smart_agent("gemini-2.0-flash")    # Will have search capability
agent2 = create_smart_agent("gemini-1.5-pro")     # Will not have search
```

### Environment-Aware Configuration

```python
import os
from google.adk.utils.variant_utils import get_google_llm_variant, GoogleLLMVariant

def configure_model_for_environment():
    """Configure model settings based on environment."""
    
    variant = get_google_llm_variant()
    
    if variant == GoogleLLMVariant.VERTEX_AI:
        print("Using Vertex AI configuration")
        # Configure for Vertex AI
        model_config = {
            "project_id": os.environ.get("GOOGLE_CLOUD_PROJECT"),
            "location": "us-central1"
        }
    else:
        print("Using Gemini API configuration")
        # Configure for Gemini API
        model_config = {
            "api_key": os.environ.get("GOOGLE_API_KEY")
        }
    
    return model_config

# Usage
config = configure_model_for_environment()
```

### Dynamic Instruction Generation

```python
from google.adk import Agent
from google.adk.utils.instructions_utils import inject_session_state
import datetime

def create_personalized_agent(user_profile: dict) -> Agent:
    """Create an agent with personalized instructions."""
    
    instruction_template = """
You are {{assistant_name}}, a personalized AI assistant.

User Profile:
- Name: {{user_name}}
- Preferred Language: {{language}}
- Experience Level: {{experience_level}}
- Interests: {{interests}}

Current Context:
- Date: {{current_date}}
- Time: {{current_time}}

Adapt your responses to the user's experience level and interests.
Be {{communication_style}} in your communication.
"""
    
    # Prepare session state
    now = datetime.datetime.now()
    session_state = {
        "assistant_name": "Alex",
        "current_date": now.strftime("%Y-%m-%d"),
        "current_time": now.strftime("%H:%M"),
        **user_profile  # Merge user profile data
    }
    
    # Generate personalized instruction
    personalized_instruction = inject_session_state(instruction_template, session_state)
    
    return Agent(
        name="personalized_assistant",
        model="gemini-2.0-flash",
        instruction=personalized_instruction
    )

# Usage
user_profile = {
    "user_name": "Alice",
    "language": "English",
    "experience_level": "beginner",
    "interests": "programming, AI, books",
    "communication_style": "friendly and encouraging"
}

agent = create_personalized_agent(user_profile)
```

### Experimental Feature Usage

```python
from google.adk.utils.feature_decorator import experimental
from google.adk import Agent

@experimental
class AdvancedReasoningAgent(Agent):
    """An experimental agent with advanced reasoning capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reasoning_mode = "experimental"
    
    @experimental
    def advanced_reasoning(self, problem: str) -> str:
        """Experimental advanced reasoning method."""
        # This would contain experimental reasoning logic
        return f"Advanced reasoning applied to: {problem}"

# Usage with awareness of experimental nature
agent = AdvancedReasoningAgent(
    name="experimental_reasoner",
    model="gemini-2.0-flash",
    instruction="You are an experimental reasoning agent."
)

# The decorators will warn about experimental features
result = agent.advanced_reasoning("Complex problem")
```

### Utility Function Composition

```python
from google.adk.utils.model_name_utils import extract_model_name, is_gemini_model
from google.adk.utils.variant_utils import get_google_llm_variant
from google.adk.utils.instructions_utils import inject_session_state

def smart_agent_factory(model_string: str, user_context: dict) -> dict:
    """Factory function that uses multiple utilities to create optimal agent config."""
    
    # Extract and validate model
    model_name = extract_model_name(model_string)
    is_gemini = is_gemini_model(model_string)
    
    # Get variant configuration
    variant = get_google_llm_variant()
    
    # Prepare instruction with context
    instruction_template = """
You are an AI assistant running on {{model_name}}.
Variant: {{variant}}
User: {{user_name}}
Capabilities: {{capabilities}}
"""
    
    session_state = {
        "model_name": model_name,
        "variant": variant,
        "user_name": user_context.get("name", "User"),
        "capabilities": "Standard AI assistant" + (" with Gemini features" if is_gemini else "")
    }
    
    instruction = inject_session_state(instruction_template, session_state)
    
    return {
        "name": f"smart_agent_{model_name.replace('-', '_')}",
        "model": model_string,
        "instruction": instruction,
        "metadata": {
            "variant": variant,
            "is_gemini": is_gemini,
            "extracted_model": model_name
        }
    }

# Usage
user_context = {"name": "John", "preferences": "detailed explanations"}
config = smart_agent_factory("gemini-2.0-flash", user_context)
print(config)
```

---

This utilities guide provides comprehensive documentation for all the helper functions and utilities available in Google ADK. These utilities help with common tasks like model validation, feature management, instruction processing, and environment configuration.