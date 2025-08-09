# Google ADK Examples and Usage Guide

This guide provides practical examples and usage patterns for building applications with the Google Agent Development Kit (ADK).

## Table of Contents

- [Getting Started](#getting-started)
- [Basic Agent Examples](#basic-agent-examples)
- [Multi-Agent Systems](#multi-agent-systems)
- [Tool Integration](#tool-integration)
- [Advanced Patterns](#advanced-patterns)
- [Production Deployment](#production-deployment)
- [Complete Applications](#complete-applications)

---

## Getting Started

### Installation and Setup

```bash
# Install Google ADK
pip install google-adk

# For development version
pip install git+https://github.com/google/adk-python.git@main
```

### Basic Environment Setup

```python
import os
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService

# Set up environment (if needed)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/credentials.json"
```

---

## Basic Agent Examples

### Simple Assistant

```python
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService

# Create a basic assistant
assistant = Agent(
    name="helpful_assistant",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant. Be concise and friendly.",
)

# Set up runner
session_service = InMemorySessionService()
runner = Runner(
    app_name="simple_assistant",
    agent=assistant,
    session_service=session_service
)

# Run the assistant
for event in runner.run(
    user_id="user123",
    session_id="session456",
    new_message="Hello! Can you help me with Python programming?"
):
    if event.type == "model_response":
        print(f"Assistant: {event.data.get('text', '')}")
```

### Search-Enabled Agent

```python
from google.adk import Agent, Runner
from google.adk.tools import google_search
from google.adk.sessions import InMemorySessionService

# Create search agent
search_agent = Agent(
    name="search_assistant",
    model="gemini-2.0-flash",
    instruction="You are a research assistant. Use web search to find current information and provide accurate, up-to-date answers.",
    tools=[google_search]
)

# Setup and run
session_service = InMemorySessionService()
runner = Runner(
    app_name="search_app",
    agent=search_agent,
    session_service=session_service
)

# Example interaction
for event in runner.run(
    user_id="researcher",
    session_id="research_session",
    new_message="What are the latest developments in quantum computing?"
):
    if event.type == "model_response":
        print(f"Research Assistant: {event.data.get('text', '')}")
```

### Custom Function Tool Agent

```python
from google.adk import Agent, Runner
from google.adk.tools import FunctionTool
from google.adk.sessions import InMemorySessionService
import datetime
import json

def get_current_time(timezone: str = "UTC") -> dict:
    """Get the current time in the specified timezone.
    
    Args:
        timezone: Timezone name (e.g., 'UTC', 'US/Eastern')
    
    Returns:
        Dictionary with current time information
    """
    now = datetime.datetime.now()
    return {
        "current_time": now.isoformat(),
        "timezone": timezone,
        "timestamp": now.timestamp(),
        "formatted": now.strftime("%Y-%m-%d %H:%M:%S")
    }

def calculate_age(birth_year: int, current_year: int = None) -> dict:
    """Calculate age from birth year.
    
    Args:
        birth_year: Year of birth
        current_year: Current year (defaults to current year)
    
    Returns:
        Dictionary with age information
    """
    if current_year is None:
        current_year = datetime.datetime.now().year
    
    age = current_year - birth_year
    return {
        "age": age,
        "birth_year": birth_year,
        "current_year": current_year
    }

# Create tools from functions
time_tool = FunctionTool(func=get_current_time)
age_tool = FunctionTool(func=calculate_age)

# Create agent with custom tools
utility_agent = Agent(
    name="utility_assistant",
    model="gemini-2.0-flash",
    instruction="You are a utility assistant with access to time and age calculation tools. Help users with these calculations.",
    tools=[time_tool, age_tool]
)

# Example usage
session_service = InMemorySessionService()
runner = Runner(
    app_name="utility_app",
    agent=utility_agent,
    session_service=session_service
)

for event in runner.run(
    user_id="user",
    session_id="util_session",
    new_message="What time is it and calculate the age of someone born in 1990?"
):
    if event.type == "model_response":
        print(f"Utility Assistant: {event.data.get('text', '')}")
```

---

## Multi-Agent Systems

### Research and Writing Pipeline

```python
from google.adk import Agent, Runner
from google.adk.tools import google_search, AgentTool
from google.adk.sessions import InMemorySessionService

# Create specialized agents
researcher = Agent(
    name="researcher",
    model="gemini-2.0-flash",
    instruction="""You are a research specialist. Your job is to:
    1. Search for current, accurate information on the given topic
    2. Gather facts, statistics, and key points
    3. Organize findings in a clear, structured format
    4. Provide sources and context for the information""",
    tools=[google_search]
)

writer = Agent(
    name="writer",
    model="gemini-2.0-flash",
    instruction="""You are a professional writer. Your job is to:
    1. Take research findings and create engaging content
    2. Structure information logically with clear headings
    3. Write in a clear, accessible style
    4. Include relevant examples and context
    5. Ensure the content flows well and is engaging"""
)

editor = Agent(
    name="editor",
    model="gemini-2.0-flash",
    instruction="""You are an editor. Your job is to:
    1. Review written content for clarity and accuracy
    2. Check for grammatical errors and flow
    3. Suggest improvements for readability
    4. Ensure the content meets the original requirements
    5. Provide a polished final version"""
)

# Create tools from agents
research_tool = AgentTool(agent=researcher)
writing_tool = AgentTool(agent=writer)
editing_tool = AgentTool(agent=editor)

# Coordinator agent
coordinator = Agent(
    name="content_coordinator",
    model="gemini-2.0-flash",
    instruction="""You coordinate content creation. Follow this process:
    1. Use the research tool to gather information on the topic
    2. Use the writing tool to create content based on the research
    3. Use the editing tool to polish the final content
    4. Present the final polished content to the user""",
    tools=[research_tool, writing_tool, editing_tool]
)

# Example usage
session_service = InMemorySessionService()
runner = Runner(
    app_name="content_pipeline",
    agent=coordinator,
    session_service=session_service
)

for event in runner.run(
    user_id="content_creator",
    session_id="creation_session",
    new_message="Create a comprehensive article about sustainable energy solutions"
):
    if event.type == "model_response":
        print(f"Coordinator: {event.data.get('text', '')}")
```

### Customer Support System

```python
from google.adk import Agent, Runner
from google.adk.tools import AgentTool, FunctionTool
from google.adk.sessions import InMemorySessionService

# Simulated knowledge base
KNOWLEDGE_BASE = {
    "password_reset": "To reset your password, go to login page and click 'Forgot Password'",
    "billing": "For billing questions, contact billing@company.com or call 1-800-BILLING",
    "technical_support": "Technical issues can be reported at support@company.com",
    "account_status": "Check account status in your user dashboard under Account Settings"
}

def search_knowledge_base(query: str) -> dict:
    """Search the knowledge base for relevant information.
    
    Args:
        query: Search query
    
    Returns:
        Dictionary with search results
    """
    results = []
    query_lower = query.lower()
    
    for topic, content in KNOWLEDGE_BASE.items():
        if any(word in topic.lower() or word in content.lower() 
               for word in query_lower.split()):
            results.append({"topic": topic, "content": content})
    
    return {"results": results, "query": query}

def escalate_to_human(issue: str, priority: str = "normal") -> dict:
    """Escalate issue to human support.
    
    Args:
        issue: Description of the issue
        priority: Priority level (low, normal, high, urgent)
    
    Returns:
        Escalation ticket information
    """
    ticket_id = f"TICKET-{hash(issue) % 10000:04d}"
    return {
        "ticket_id": ticket_id,
        "issue": issue,
        "priority": priority,
        "status": "escalated",
        "message": f"Your issue has been escalated to human support. Ticket ID: {ticket_id}"
    }

# Create specialized agents
knowledge_agent = Agent(
    name="knowledge_assistant",
    model="gemini-2.0-flash",
    instruction="Search the knowledge base and provide helpful information to users.",
    tools=[FunctionTool(func=search_knowledge_base)]
)

escalation_agent = Agent(
    name="escalation_handler",
    model="gemini-2.0-flash",
    instruction="Handle escalations to human support when issues cannot be resolved automatically.",
    tools=[FunctionTool(func=escalate_to_human)]
)

# Main support agent
support_agent = Agent(
    name="support_coordinator",
    model="gemini-2.0-flash",
    instruction="""You are a customer support coordinator. Follow this process:
    1. Understand the customer's issue clearly
    2. Try to resolve using the knowledge base first
    3. If the issue cannot be resolved, escalate to human support
    4. Always be helpful, patient, and professional
    5. Provide clear next steps to the customer""",
    tools=[
        AgentTool(agent=knowledge_agent),
        AgentTool(agent=escalation_agent)
    ]
)

# Example usage
session_service = InMemorySessionService()
runner = Runner(
    app_name="customer_support",
    agent=support_agent,
    session_service=session_service
)

for event in runner.run(
    user_id="customer123",
    session_id="support_session",
    new_message="I can't reset my password and I need urgent help with my billing"
):
    if event.type == "model_response":
        print(f"Support: {event.data.get('text', '')}")
```

---

## Tool Integration

### REST API Integration

```python
from google.adk import Agent, Runner
from google.adk.tools import BaseTool, ToolContext
from google.adk.sessions import InMemorySessionService
import aiohttp
import json
from typing import Any, Optional
from google.genai import types

class WeatherAPITool(BaseTool):
    """Tool for fetching weather data from an API."""
    
    def __init__(self, api_key: str):
        super().__init__(
            name="weather_api",
            description="Get current weather information for any city"
        )
        self.api_key = api_key
    
    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        city = args.get("city")
        if not city:
            return {"error": "City name is required"}
        
        # Simulate API call (replace with real API)
        async with aiohttp.ClientSession() as session:
            # In a real implementation, you would make an actual API call
            # url = f"https://api.weather.com/v1/current?city={city}&key={self.api_key}"
            # async with session.get(url) as response:
            #     data = await response.json()
            #     return data
            
            # Simulated response
            return {
                "city": city,
                "temperature": "22°C",
                "condition": "Sunny",
                "humidity": "45%",
                "wind_speed": "10 km/h"
            }
    
    def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
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

# Create weather agent
weather_tool = WeatherAPITool(api_key="your-api-key")

weather_agent = Agent(
    name="weather_assistant",
    model="gemini-2.0-flash",
    instruction="You are a weather assistant. Use the weather API to provide current weather information for any city requested.",
    tools=[weather_tool]
)

# Example usage
session_service = InMemorySessionService()
runner = Runner(
    app_name="weather_app",
    agent=weather_agent,
    session_service=session_service
)

for event in runner.run(
    user_id="user",
    session_id="weather_session",
    new_message="What's the weather like in Tokyo?"
):
    if event.type == "model_response":
        print(f"Weather Assistant: {event.data.get('text', '')}")
```

### Database Integration

```python
from google.adk import Agent, Runner
from google.adk.tools import FunctionTool
from google.adk.sessions import InMemorySessionService
import sqlite3
import json
from typing import List, Dict, Any

# Set up a simple database
def setup_database():
    """Set up a simple user database for demonstration."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            age INTEGER,
            city TEXT
        )
    """)
    
    # Insert sample data
    users = [
        (1, "Alice Johnson", "alice@example.com", 30, "New York"),
        (2, "Bob Smith", "bob@example.com", 25, "Los Angeles"),
        (3, "Carol Davis", "carol@example.com", 35, "Chicago"),
        (4, "David Wilson", "david@example.com", 28, "Houston")
    ]
    
    cursor.executemany("INSERT INTO users VALUES (?, ?, ?, ?, ?)", users)
    conn.commit()
    return conn

# Database connection
db_conn = setup_database()

def search_users(name: str = None, city: str = None, min_age: int = None) -> List[Dict[str, Any]]:
    """Search users in the database.
    
    Args:
        name: Name to search for (partial match)
        city: City to filter by
        min_age: Minimum age filter
    
    Returns:
        List of matching users
    """
    cursor = db_conn.cursor()
    query = "SELECT id, name, email, age, city FROM users WHERE 1=1"
    params = []
    
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    
    if city:
        query += " AND city = ?"
        params.append(city)
    
    if min_age is not None:
        query += " AND age >= ?"
        params.append(min_age)
    
    cursor.execute(query, params)
    results = cursor.fetchall()
    
    return [
        {
            "id": row[0],
            "name": row[1],
            "email": row[2],
            "age": row[3],
            "city": row[4]
        }
        for row in results
    ]

def get_user_count_by_city() -> Dict[str, int]:
    """Get count of users by city.
    
    Returns:
        Dictionary with city names and user counts
    """
    cursor = db_conn.cursor()
    cursor.execute("SELECT city, COUNT(*) FROM users GROUP BY city")
    results = cursor.fetchall()
    
    return {city: count for city, count in results}

# Create database tools
search_tool = FunctionTool(func=search_users)
stats_tool = FunctionTool(func=get_user_count_by_city)

# Create database agent
db_agent = Agent(
    name="database_assistant",
    model="gemini-2.0-flash",
    instruction="""You are a database assistant. You can help users search for users and get statistics from the user database. 
    Use the search_users function to find specific users and get_user_count_by_city to get statistics.""",
    tools=[search_tool, stats_tool]
)

# Example usage
session_service = InMemorySessionService()
runner = Runner(
    app_name="database_app",
    agent=db_agent,
    session_service=session_service
)

for event in runner.run(
    user_id="admin",
    session_id="db_session",
    new_message="Find all users in New York and show me user statistics by city"
):
    if event.type == "model_response":
        print(f"Database Assistant: {event.data.get('text', '')}")
```

---

## Advanced Patterns

### Async Operations and Streaming

```python
import asyncio
from google.adk import Agent, Runner
from google.adk.tools import BaseTool, ToolContext
from google.adk.sessions import InMemorySessionService
from typing import Any, AsyncGenerator

class StreamingDataTool(BaseTool):
    """Tool that demonstrates streaming data processing."""
    
    def __init__(self):
        super().__init__(
            name="streaming_processor",
            description="Process data in streaming fashion",
            is_long_running=True
        )
    
    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        data_type = args.get("data_type", "numbers")
        count = args.get("count", 10)
        
        results = []
        
        # Simulate streaming processing
        for i in range(count):
            if data_type == "numbers":
                item = {"index": i, "value": i * i, "timestamp": i}
            else:
                item = {"index": i, "message": f"Processed item {i}", "timestamp": i}
            
            results.append(item)
            
            # Simulate processing delay
            await asyncio.sleep(0.1)
            
            # In a real streaming scenario, you might yield results progressively
            # For this example, we collect all results
        
        return {
            "data_type": data_type,
            "total_processed": len(results),
            "results": results[:5],  # Return first 5 for brevity
            "summary": f"Processed {len(results)} items of type {data_type}"
        }

# Callback functions for monitoring
async def before_tool_callback(tool, args, tool_context):
    """Called before tool execution."""
    print(f"Starting tool: {tool.name} with args: {args}")
    return None

async def after_tool_callback(tool, args, tool_context, result):
    """Called after tool execution."""
    print(f"Completed tool: {tool.name}")
    return None

# Create streaming agent with callbacks
streaming_agent = Agent(
    name="streaming_assistant",
    model="gemini-2.0-flash",
    instruction="You are a streaming data processor. Use the streaming tool to process data requests.",
    tools=[StreamingDataTool()],
    before_tool_callbacks=before_tool_callback,
    after_tool_callbacks=after_tool_callback
)

async def run_streaming_example():
    """Run the streaming example asynchronously."""
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="streaming_app",
        agent=streaming_agent,
        session_service=session_service
    )
    
    async for event in runner.run_async(
        user_id="stream_user",
        session_id="stream_session",
        new_message="Process 20 numbers in streaming fashion"
    ):
        if event.type == "model_response":
            print(f"Streaming Assistant: {event.data.get('text', '')}")

# Run the async example
# asyncio.run(run_streaming_example())
```

### State Management and Memory

```python
from google.adk import Agent, Runner
from google.adk.tools import FunctionTool, load_memory, preload_memory
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService

# State management functions
def save_user_preference(key: str, value: str, user_context: dict = None) -> dict:
    """Save a user preference.
    
    Args:
        key: Preference key
        value: Preference value
        user_context: Current user context
    
    Returns:
        Confirmation of saved preference
    """
    # In a real implementation, this would save to a database
    print(f"Saving preference: {key} = {value}")
    return {
        "saved": True,
        "key": key,
        "value": value,
        "message": f"Preference '{key}' saved as '{value}'"
    }

def get_user_preference(key: str, user_context: dict = None) -> dict:
    """Get a user preference.
    
    Args:
        key: Preference key to retrieve
        user_context: Current user context
    
    Returns:
        The preference value or default
    """
    # Simulated preferences store
    preferences = {
        "theme": "dark",
        "language": "en",
        "notifications": "enabled"
    }
    
    value = preferences.get(key, "not_set")
    return {
        "key": key,
        "value": value,
        "found": key in preferences
    }

# Create stateful agent
stateful_agent = Agent(
    name="stateful_assistant",
    model="gemini-2.0-flash",
    instruction="""You are a stateful assistant that remembers user preferences and conversation history. 
    You can save and retrieve user preferences, and you maintain context across conversations.
    Always use the conversation memory to provide personalized responses.""",
    tools=[
        FunctionTool(func=save_user_preference),
        FunctionTool(func=get_user_preference),
        load_memory,
        preload_memory
    ]
)

# Set up with memory service
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

runner = Runner(
    app_name="stateful_app",
    agent=stateful_agent,
    session_service=session_service,
    memory_service=memory_service
)

# Example conversation with state
print("=== First interaction ===")
for event in runner.run(
    user_id="stateful_user",
    session_id="persistent_session",
    new_message="Hi! I prefer dark theme and want notifications enabled. Please save these preferences."
):
    if event.type == "model_response":
        print(f"Assistant: {event.data.get('text', '')}")

print("\n=== Second interaction (same session) ===")
for event in runner.run(
    user_id="stateful_user",
    session_id="persistent_session",
    new_message="What were my theme preferences again?"
):
    if event.type == "model_response":
        print(f"Assistant: {event.data.get('text', '')}")
```

---

## Production Deployment

### FastAPI Web Service

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.adk import Agent, Runner
from google.adk.tools import google_search
from google.adk.sessions import InMemorySessionService
import asyncio
import uuid
from typing import Optional

app = FastAPI(title="ADK Agent Service", version="1.0.0")

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    user_id: str
    session_id: str

# Initialize agent and runner
agent = Agent(
    name="web_assistant",
    model="gemini-2.0-flash",
    instruction="You are a helpful web assistant. Provide clear, concise responses.",
    tools=[google_search]
)

session_service = InMemorySessionService()
runner = Runner(
    app_name="web_service",
    agent=agent,
    session_service=session_service
)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for interacting with the agent."""
    try:
        user_id = request.user_id or str(uuid.uuid4())
        session_id = request.session_id or str(uuid.uuid4())
        
        response_text = ""
        
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=request.message
        ):
            if event.type == "model_response":
                response_text = event.data.get("text", "")
                break
        
        return ChatResponse(
            response=response_text,
            user_id=user_id,
            session_id=session_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "adk-agent-service"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  adk-agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json
    volumes:
      - ./credentials.json:/app/credentials.json:ro
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
```

---

## Complete Applications

### Personal Assistant Bot

```python
from google.adk import Agent, Runner
from google.adk.tools import (
    google_search, url_context, FunctionTool,
    load_memory, preload_memory
)
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
import datetime
import json

# Personal assistant functions
def create_reminder(task: str, date: str, time: str = "09:00") -> dict:
    """Create a reminder for a task.
    
    Args:
        task: Task description
        date: Date in YYYY-MM-DD format
        time: Time in HH:MM format
    
    Returns:
        Reminder confirmation
    """
    reminder_id = f"REM_{hash(f'{task}{date}{time}') % 10000:04d}"
    return {
        "reminder_id": reminder_id,
        "task": task,
        "date": date,
        "time": time,
        "status": "created",
        "message": f"Reminder created: '{task}' on {date} at {time}"
    }

def get_calendar_events(date: str = None) -> dict:
    """Get calendar events for a specific date.
    
    Args:
        date: Date in YYYY-MM-DD format (defaults to today)
    
    Returns:
        List of events for the date
    """
    if not date:
        date = datetime.date.today().isoformat()
    
    # Simulated calendar events
    events = {
        "2024-01-15": [
            {"time": "09:00", "title": "Team Meeting", "duration": "1h"},
            {"time": "14:00", "title": "Project Review", "duration": "30m"}
        ],
        "2024-01-16": [
            {"time": "10:00", "title": "Client Call", "duration": "45m"}
        ]
    }
    
    return {
        "date": date,
        "events": events.get(date, []),
        "total_events": len(events.get(date, []))
    }

def calculate_distance(origin: str, destination: str) -> dict:
    """Calculate distance between two locations.
    
    Args:
        origin: Starting location
        destination: Ending location
    
    Returns:
        Distance and travel time information
    """
    # Simulated distance calculation
    return {
        "origin": origin,
        "destination": destination,
        "distance": "25.3 km",
        "driving_time": "32 minutes",
        "walking_time": "4 hours 15 minutes",
        "public_transport": "45 minutes"
    }

# Create personal assistant
personal_assistant = Agent(
    name="personal_assistant",
    model="gemini-2.0-flash",
    instruction="""You are a comprehensive personal assistant. You can help with:
    
    1. Information research and web searches
    2. Creating reminders and managing tasks
    3. Checking calendar events and scheduling
    4. Calculating distances and travel times
    5. Analyzing web content and URLs
    6. Maintaining conversation context and memory
    
    Always be proactive, helpful, and personalized in your responses. 
    Use the available tools to provide accurate and useful information.""",
    tools=[
        google_search,
        url_context,
        FunctionTool(func=create_reminder),
        FunctionTool(func=get_calendar_events),
        FunctionTool(func=calculate_distance),
        load_memory,
        preload_memory
    ]
)

# Set up services
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

runner = Runner(
    app_name="personal_assistant",
    agent=personal_assistant,
    session_service=session_service,
    memory_service=memory_service
)

# Example interactions
def run_personal_assistant_demo():
    """Run a demo of the personal assistant."""
    
    interactions = [
        "Hi! I'm planning a trip to Tokyo next week. Can you help me research the best places to visit?",
        "Great! Can you also create a reminder for me to book flights on January 20th at 2 PM?",
        "What's the distance between Tokyo Station and Shibuya?",
        "Check my calendar for January 15th and 16th.",
        "Can you search for current weather in Tokyo?"
    ]
    
    user_id = "demo_user"
    session_id = "demo_session"
    
    for i, message in enumerate(interactions, 1):
        print(f"\n=== Interaction {i} ===")
        print(f"User: {message}")
        
        for event in runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=message
        ):
            if event.type == "model_response":
                print(f"Assistant: {event.data.get('text', '')}")

# Uncomment to run the demo
# run_personal_assistant_demo()
```

---

This examples guide demonstrates the flexibility and power of the Google ADK framework. From simple assistants to complex multi-agent systems, ADK provides the tools and patterns needed to build sophisticated AI applications.