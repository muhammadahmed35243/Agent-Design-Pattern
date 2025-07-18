# Agent-Design-Pattern

A sophisticated multi-agent system that simulates intelligent behavior through coordinated planning, reflection, and tool-usage agents. This project demonstrates advanced routing decisions using Large Language Models (Grok API), classifier-based routing, and dynamic tool invocation with real-time data integration.

## Overview

Agent-Design-Pattern represents a modular approach to multi-agent AI systems, designed to handle complex tasks through intelligent agent coordination. The system leverages state-of-the-art LLMs for decision-making while maintaining flexibility through classifier-based routing and comprehensive tool integration.

## Key Features

**Multi-Agent Architecture**
- Coordinated Planning, Reflection, and Tool-User agents
- Intelligent task decomposition and execution
- Seamless inter-agent communication

**Advanced Routing System**
- LLM-based decision making via Grok API
- Classifier-based routing with trained ML models
- Dynamic agent selection based on query context

**Robust Infrastructure**
- Persistent memory for dialogue history
- Interactive Streamlit web interface
- Comprehensive tool integration (Search, Calculator, SerpAPI)
- Modular and extensible architecture

## System Architecture

### Agent Specifications

| Agent | Purpose | Capabilities |
|-------|---------|--------------|
| **PlanningAgent** | Strategic Planning | Breaks down complex goals into actionable steps and creates structured execution plans |
| **ReflectionAgent** | Analysis & Insights | Provides feedback, conducts analysis, and offers strategic insights |
| **ToolUserAgent** | External Operations | Executes calculations, web searches, and external tool-based operations |

### Technology Stack

**Core Technologies**
- Python 3.11+
- Streamlit (User Interface)
- Grok API (LLM Decision Making)
- SerpAPI (Real-time Search)
- Scikit-learn + Sentence-Transformers (Classification)
- OpenAI-Compatible Groq LLMs

**Supporting Libraries**
- dotenv, requests, JSON, datetime

### API Integration

| Service | Implementation | Purpose |
|---------|---------------|---------|
| **Grok API** | `chat_with_grok` function | LLM-based agent selection and reasoning |
| **SerpAPI** | ToolUserAgent integration | Web search and real-time information retrieval |
| **Custom Classifier** | `agent_classifier` model | Predictive agent routing based on query analysis |

## Project Structure

```
AGENT-DESIGN-PATTERN/
├── .venv/                          # Virtual environment
├── data/                           # Data storage
│   ├── agent_logs.json            # Agent interaction logs
│   ├── agent_routing_dataset.csv  # Training data for classifier
│   └── planning_steps.json        # Planning agent outputs
├── logs/                          # System logs
│   └── agent_logs.json
├── model/                         # ML models
│   └── simple_agent_classifier.pkl
├── notebook/                      # Development notebooks
│   └── train_agent_classifier.ipynb
├── src/                           # Core source code
│   ├── base_agent.py              # Base agent class
│   ├── call_groq.py               # Grok API integration
│   ├── multi_agent.py             # Multi-agent coordinator
│   ├── planning_agent.py          # Planning agent implementation
│   ├── reflection_agent.py        # Reflection agent implementation
│   └── tool_user_agent.py         # Tool user agent implementation
├── tools/                         # External tools
│   ├── calculator.py              # Mathematical operations
│   └── search.py                  # Search functionality
├── .env                           # Environment variables
├── main.py                        # Application entry point
├── README.md                      # Documentation
└── requirements.txt               # Dependencies
```

## Installation & Setup

### Prerequisites
- Python 3.11 or higher
- Valid API keys for Grok and SerpAPI

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/muhammadahmed35243/Agent-Design-Pattern.git
   cd Agent-Design-Pattern
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   GROK_API=your_groq_api_key
   SERPAPI_API_KEY=your_serpapi_key
   ```

4. **Launch the Application**
   ```bash
   streamlit run main.py
   ```

## Usage Examples

The system intelligently routes queries to the most appropriate agent based on content analysis:

**Planning Tasks**
- "Break this project into actionable steps"
- "Create a roadmap for implementing this feature"

**Reflection & Analysis**
- "Why is my model underperforming?"
- "Analyze the effectiveness of this strategy"

**Tool-Based Operations**
- "What's the capital of Sweden?"
- "Calculate the compound interest for this investment"

## Contributing

We welcome contributions to Agent-Design-Pattern. Please ensure your code follows the established patterns and includes appropriate documentation.

## License

This project is open source and available under the [MIT License](LICENSE).

## Author

**Muhammad Ahmed**

*Studying AI Architecture, Multi-Agent Systems, and Intelligent Automation*

---

**Agent-Design-Pattern** - Advancing the frontier of multi-agent AI systems through intelligent coordination and modular design.
