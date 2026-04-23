# kiso-llm

A Python library that provides a unified interface for calling LLM providers. Designed for JSON-formatted responses with built-in retry logic, token tracking, and conversation history persistence.

## Supported Providers

| llm_choice | Provider |
|---|---|
| `claude` | Anthropic API (direct) |
| `claude_azure` | Claude via Azure Databricks |

**Note:**
- While the codebase may include implementations for GPT, Gemini, and Llama for comparison purposes, these providers have not been continuously tested or maintained and may not work as expected. Only Claude-based providers (claude, claude_azure) are actively maintained and will continue to be updated alongside new model releases.
- The default model is currently `claude-opus-4-7` (Claude Opus 4.7).


## Setup

```bash
pip install git+https://github.com/momo-trip/kiso-llm.git
```


### Dependencies

```
openai
anthropic
google-generativeai
replicate
```

## Usage

### Basic Call

```python
from llm_api import ask_llm, LLMInterface

# Configure LLMInterface
interface = LLMInterface(
    llm_choice="claude_azure",
    llm_model="databricks-claude-sonnet-4",
    api_key="your-api-key",
    azure_endpoint="https://your-endpoint.azuredatabricks.net/serving-endpoints",
    temperature=0,
    output_max=8192,
    timeout=60,
    history_path="./history.json",
    token_path="./token.json",
    database_dir="./db",
    chat_dir="./chat",
    count_path="./count.json",
    exp_data={},
)

# Start a new conversation
response = ask_llm("Your prompt here", memory_type="init", llm_interface=interface)

# Continue the conversation (carries over history)
response = ask_llm("Follow-up prompt", memory_type="continue", llm_interface=interface)
```

### LLMInterface Parameters

| Parameter | Description |
|---|---|
| `llm_choice` | Provider to use (`claude_azure`, `claude`, etc.) |
| `llm_model` | Model name (e.g., `databricks-claude-sonnet-4`) |
| `api_key` | API key.
| `azure_endpoint` | Azure/Databricks endpoint URL. |
| `temperature` | Generation temperature (0 = deterministic, higher = more random) |
| `output_max` | Maximum output tokens |
| `timeout` | API timeout in seconds |
| `history_path` | File path to persist conversation history (JSON) |
| `token_path` | File path to record token usage |
| `database_dir` | Directory for prompt/response logs |
| `chat_dir` | Directory for chat logs |
| `count_path` | File path for prompt count tracking |


### memory_type

| Value | Behavior |
|---|---|
| `"init"` | Clears history and starts a new conversation |
| `"continue"` | Loads history from `history_path` and continues the conversation |

### Return Value

Returns a JSON object (Python dict/list) extracted from the LLM response. Internally, `extract_json_response` parses the response string into structured JSON.

## Key Features

### Automatic Retry

Automatically retries on rate limits and server errors with exponential backoff. Error handling is implemented per provider.

### JSON Response Auto-Repair

When the LLM response is not valid JSON, the library automatically requests corrections (escape fixes, format adjustments) and retries until valid JSON is obtained.

### Long Response Splitting

When a response exceeds `output_max`, the library instructs the LLM to split the output into smaller segments, using an `ongoing` key to track continuation state.

### Conversation History Management

Conversation history is automatically saved to `history_path` as a JSON file. When the context window limit is reached, history is automatically trimmed (GPT: ~100k tokens, Claude: ~180k tokens).

### Token Usage Tracking

Input and output token counts are recorded to `token_path`. Prompts and responses are also logged to `database_dir` / `chat_dir`.

## Directory Structure

```
kiso-llm/
├── llm_api/           # Main package
├── pyproject.toml     # Package configuration
├── requirements.txt   # Dependencies
└── README.md
```
