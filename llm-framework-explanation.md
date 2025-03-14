# Local LLM Framework Explanation

## Core Components

### 1. Data Structures
- `Message`: A simple dataclass representing a conversation message with a role (system/user/assistant) and content.
- `LLMResponse`: Structured output from an LLM containing the generated text, model name, token usage stats, and the raw response.

### 2. Provider Classes

#### Base Class: `LocalLLMProvider`
This abstract class defines the interface that all LLM providers must implement:
- `generate_text()`: For simple text completion from a prompt
- `generate_chat()`: For generating responses in a conversation format
- `_count_tokens()`: A basic method to estimate token counts

#### Concrete Providers:

1. **`LlamaProvider`**:
   - Integrates with the llama.cpp binary to run LLaMA models locally
   - Takes a path to the model weights file
   - Converts prompts and chat messages into a format suitable for LLaMA
   - Uses subprocess to call the llama.cpp executable

2. **`OllamaProvider`**:
   - Connects to an Ollama server running locally (typically on port 11434)
   - Takes a model name (like "llama2" or "mistral") rather than a file path
   - Uses HTTP requests to communicate with the Ollama API
   - Supports both direct text generation and chat completions

### 3. Client Interface: `LLMClient`

This is the main class users interact with:
- Takes a provider instance in its constructor
- Offers simplified methods for chat and text completion
- Handles conversion between dictionary and Message objects
- Provides utilities like saving responses to markdown files

## How It Works

1. **Setup**: The user creates a provider (LlamaProvider or OllamaProvider) and passes it to an LLMClient.

2. **For LlamaProvider**:
   - The framework finds the llama.cpp executable and verifies the model path
   - When generating text, it constructs a command-line call to llama.cpp
   - It processes the output to extract the generated text

3. **For OllamaProvider**:
   - The framework verifies that the Ollama server is running
   - It formats requests as JSON and sends them to the Ollama API
   - For chat completion, it converts the messages to Ollama's expected format

4. **Usage Patterns**:
   - For simple prompts: `client.prompt("Tell me about AI")`
   - For conversations: `client.chat([Message(role="user", content="Hello")])`
   - Results include both the generated text and metadata like token counts

## Key Features

- **Local Execution**: Runs models on your own hardware without sending data to external services
- **Multiple Model Support**: Works with both LLaMA models via llama.cpp and any model supported by Ollama
- **Consistent Interface**: The same client code works regardless of which provider you use
- **Flexible Configuration**: Supports passing additional parameters to the underlying models
- **Logging**: Built-in logging to track operations and diagnose issues
