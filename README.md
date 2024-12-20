# Flux: Advanced AI Agent Framework

Flux is a powerful Python framework built from the ground up with asynchronous execution at its core. It provides a modern, async-first approach to building AI agents, featuring non-blocking LLM integration, concurrent vector operations, and real-time monitoring capabilities.

## 🌟 Key Features

### 🤖 Configurable Agents
- Built on asyncio for non-blocking, concurrent operations
- Flexible agent configuration with different logging modes
- Tool integration with React-style reasoning
- Robust message handling and state management
- Vector store integration for context management
- Parallel processing of multiple agent tasks

### 🧠 LLM Integration
- Support for multiple LLM providers (Gemini, Pulse)
- Asynchronous inference with automatic batching
- Non-blocking embedding generation
- Built-in embedding generation and management
- Smart token handling and context management

### 🛠️ Utilities
- Concurrent file and metadata operations
- Efficient serialization with msgpack and orjson
- Text summarization and truncation capabilities
- Shared tokenizer functionality
- Thread-safe resource management
- Comprehensive logging and monitoring

## 📦 Installation

```bash
pip install flux-agents
```

Or with torch support for local embeddings:
```bash
pip install flux-agents[torch]
```

## 🚀 Quick Start

```python
from agents import Agent, Message, AgentConfig, Logging
import asyncio

# Create an agent with basic logging
agent = Agent(
    name="Test Agent",
    config=AgentConfig(
        task_prompt="You must respond like a pirate",
        logging=Logging.ENABLED
    )
)

# Example of concurrent agent operations
async def main():
    # Create multiple messages
    messages = [
        Message(content="What is the weather?"),
        Message(content="Tell me a joke"),
        Message(content="Write a poem")
    ]
    
    # Process messages concurrently
    responses = await asyncio.gather(*[
        agent(msg) for msg in messages
    ])
    
    for msg, response in zip(messages, responses):
        print(f"Q: {msg.content}\nA: {response.content}\n")

# Run the example
asyncio.run(main())
```

## 📊 Core Components

### Agent System
- Base agent implementation with configurable behaviors
- React-style agents for tool integration
- Vector chat capabilities
- Concurrent message processing
- Thread-safe state management

### Monitoring
- Comprehensive logging system
- Real-time monitoring with async logging
- Support for multiple logging backends
- Integration with LangFuse
- Detailed agent activity tracking

### Storage
- Concurrent vector operations with HNSW
- Asynchronous file I/O
- File management system
- Serialization utilities
- State persistence

## 🔧 Dependencies

Core dependencies:
```
msgpack>=1.0.5
orjson>=3.9.10
google-generativeai
hnswlib
aiofiles
aiohttp
plotly
pandas
numpy
```

Optional dependencies:
```
sentence-transformers[torch]>=2.5.0
torch
transformers
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

[License details here]

## 👥 Authors

- Christian de Frondeville
- Arijit Nukala
- Gubi Ganguly

## 📚 Documentation

Full documentation is available at [GitHub Pages URL] 