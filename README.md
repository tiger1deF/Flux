# Flux: Advanced AI Agent Framework

Flux is a powerful Python framework for building configurable AI agents with advanced capabilities including LLM integration, vector storage, and comprehensive monitoring.

## 🌟 Key Features

### 🤖 Configurable Agents
- Flexible agent configuration with different logging modes
- Tool integration with React-style reasoning
- Robust message handling and state management
- Vector store integration for context management
- Async-first design for optimal performance

### 🧠 LLM Integration
- Support for multiple LLM providers (Gemini, Pulse)
- Both async and sync inference modes
- Built-in embedding generation and management
- Smart token handling and context management

### 🛠️ Utilities
- Efficient serialization with msgpack and orjson
- Text summarization and truncation capabilities
- Shared tokenizer functionality
- Advanced file and metadata management
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

# Create an agent with basic logging
agent = Agent(
    name="Test Agent",
    config=AgentConfig(
        task_prompt="You must respond like a pirate",
        logging=Logging.ENABLED
    )
)

# Send a message
response = await agent(Message(content="Hello, world!"))
```

## 📊 Core Components

### Agent System
- Base agent implementation with configurable behaviors
- React-style agents for tool integration
- Vector chat capabilities
- Message handling and state management

### Monitoring
- Comprehensive logging system
- Support for multiple logging backends
- Integration with LangFuse
- Detailed agent activity tracking

### Storage
- Vector store integration
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