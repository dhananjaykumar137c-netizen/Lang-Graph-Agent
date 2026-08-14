# LangGraph Customer Support Agent - Setup Guide

## 📋 Requirements

### Python Dependencies (`requirements.txt`)
```txt
langgraph>=0.0.26
google-generativeai>=0.3.0
python-dotenv>=1.0.0
pydantic>=2.0.0
asyncio
typing-extensions>=4.8.0
dataclasses
```

### Environment Variables (`.env`)
```bash
# Gemini AI API Key (required)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: MCP Server URLs (if using real MCP servers)
MCP_COMMON_URL=http://localhost:8001
MCP_ATLAS_URL=http://localhost:8002
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd langgraph-customer-support
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API
1. Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create `.env` file:
```bash
echo "GEMINI_API_KEY=your_actual_api_key" > .env
```

### 5. Run the Demo
```bash
python langgraph_agent.py
```

## 📁 Project Structure
```
langgraph-customer-support/
│
├── langgraph_agent.py        # Main agent implementation
├── agent_config.yaml         # Workflow configuration
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create this)
├── README.md                 # Project documentation
│
├── tests/                    # Test cases
│   ├── test_stages.py
│   └── test_integration.py
│
├── logs/                     # Execution logs
│   └── agent_logs.txt
│
└── examples/                 # Example queries
    ├── order_status.json
    ├── refund_request.json
    └── technical_support.json
```

## 🧪 Testing the Agent

### Basic Test
```python
# test_basic.py
import asyncio
from langgraph_agent import run_customer_support_agent

async def test_order_status():
    input_data = {
        "customer_name": "Jane Doe",
        "email": "jane@example.com",
        "query": "Where is my order #123?",
        "priority": "high",
        "ticket_id": "T002"
    }
    result = await run_customer_support_agent(input_data)
    print(f"Result: {result}")

asyncio.run(test_order_status())
```

### Complex Test (Refund with Escalation)
```python
# test_refund.py
async def test_refund_escalation():
    input_data = {
        "customer_name": "John Smith",
        "email": "",  # Missing email triggers clarification
        "query": "My order #456 arrived damaged. I want a full refund immediately!",
        "priority": "urgent",
        "ticket_id": "T003"
    }
    result = await run_customer_support_agent(input_data)
    assert result['escalation_required'] == True
    print("Escalation test passed!")
```

## 🔄 Workflow Stages Explained

### Deterministic Stages (Sequential)
- **INTAKE**: Accepts initial payload
- **UNDERSTAND**: Parses query using AI
- **PREPARE**: Normalizes and enriches data
- **WAIT**: Processes customer responses
- **RETRIEVE**: Searches knowledge base
- **UPDATE**: Updates ticket status
- **CREATE**: Generates response
- **DO**: Executes API calls
- **COMPLETE**: Outputs final result

### Non-Deterministic Stage
- **DECIDE**: AI evaluates multiple solutions and scores them

### Human Interaction Stage
- **ASK**: Requests clarification from customer

## 🔧 Customization

### Adding New Abilities
```python
# In MCPClient class
async def _execute_custom_ability(self, ability_name: str, params: Dict) -> Dict:
    if ability_name == "your_custom_ability":
        # Your logic here
        return {"result": "processed"}
```

### Modifying Stage Logic
```python
async def custom_stage(state: WorkflowState) -> WorkflowState:
    # Your stage logic
    state['custom_field'] = "value"
    return state

# Add to workflow
workflow.add_node("custom", custom_stage)
workflow.add_edge("previous_stage", "custom")
```

### Changing AI Model
```python
# Switch to different Gemini model
gemini_model = genai.GenerativeModel('gemini-pro-vision')

# Or use a different AI provider
# Replace gemini calls with your preferred AI API
```

## 📊 Monitoring & Logging

### Enable Detailed Logging
```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/agent_debug.log'),
        logging.StreamHandler()
    ]
)
```

### Performance Metrics
```python
# Add to stage_11_complete
metrics = {
    "total_stages": len(state['stage_history']),
    "processing_time": state['processing_time'],
    "ai_calls": state.get('ai_call_count', 0),
    "external_api_calls": len(state.get('executed_actions', []))
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Gemini API Error**
   - Check API key is valid
   - Ensure you have API quota remaining
   - Try with smaller queries

2. **Async Errors**
   ```python
   # Windows users might need:
   asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
   ```

3. **State Persistence Issues**
   - Ensure all state fields are properly initialized
   - Check for None values before accessing nested fields

## 📈 Performance Optimization

### Caching KB Results
```python
from functools import lru_cache

@lru_cache(maxsize=100)
async def cached_kb_search(query: str):
    return await atlas_client.call_ability("knowledge_base_search", {"query": query})
```

### Parallel Processing
```python
# Execute independent stages in parallel
results = await asyncio.gather(
    common_client.call_ability("parse_request_text", params1),
    atlas_client.call_ability("extract_entities", params2)
)
```

## 🚢 Deployment

### Docker Setup
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "langgraph_agent.py"]
```

### Environment Configuration
```yaml
# docker-compose.yml
version: '3.8'
services:
  agent:
    build: .
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    ports:
      - "8000:8000"
```

## 📝 Submission Checklist

Before submitting:
- [ ] Code runs without errors
- [ ] All 11 stages execute correctly
- [ ] State persists across stages
- [ ] Gemini AI integration works
- [ ] Logs show clear execution flow
- [ ] Configuration file included
- [ ] README with setup instructions
- [ ] Example test cases provided
- [ ] Demo video recorded

## 🎥 Demo Video Script

1. **Introduction** (30 seconds)
   - Your name
   - Brief overview of the solution

2. **Architecture** (1 minute)
   - Show the 11-stage workflow
   - Explain deterministic vs non-deterministic stages
   - Show MCP client integration

3. **Live Demo** (2-3 minutes)
   - Run basic order status query
   - Show state persistence across stages
   - Demonstrate escalation scenario
   - Display final output

4. **Code Walkthrough** (2 minutes)
   - Show key implementation details
   - Explain Gemini AI integration
   - Highlight state management

5. **Conclusion** (30 seconds)
   - Summary of features
   - Future improvements possible

## 📧 Contact & Support

For questions about this implementation:
- Review the Claude conversation in `Customer service claude convo.txt`
- Check the original task document
- Test with different query types

Good luck with your submission! 🎉