# Local LLM Framework Setup Guide

This guide will walk you through setting up and using a Python framework for accessing local large language models (LLMs).

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Option 1: Using Ollama (Recommended)](#option-1-using-ollama-recommended)
  - [Option 2: Using llama.cpp](#option-2-using-llamacpp)
- [Framework Files](#framework-files)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## Prerequisites

- Python 3.8 or higher
- Basic knowledge of Python programming
- At least 8GB RAM (16GB+ recommended for larger models)
- (Optional) GPU with CUDA support for faster inference

## Installation

### Option 1: Using Ollama (Recommended)

Ollama provides an easy way to run local LLMs with minimal setup.

1. **Install Ollama**:
   - Visit [ollama.ai](https://ollama.ai/) and download the installer for your platform
   - Available for macOS, Linux, and Windows
   - Follow the installation instructions

2. **Pull a model**:
   After installing Ollama, open a terminal and pull a model:
   ```bash
   ollama pull llama3       # The latest Llama 3 model
   # Alternative models:
   # ollama pull mistral     # Mistral 7B
   # ollama pull phi         # Microsoft Phi-2
   # ollama pull gemma       # Google Gemma
   ```

3. **Install Python dependencies**:
   ```bash
   pip install requests
   ```

### Option 2: Using llama.cpp

For more advanced users who want direct control over model loading and inference.

1. **Install llama.cpp**:
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   make
   ```

2. **Download a model**:
   - Download a GGUF format model from [HuggingFace](https://huggingface.co/)
   - Popular options:
     - [Llama-3](https://huggingface.co/meta-llama)
     - [Mistral-7B](https://huggingface.co/mistralai)
     - [Phi-2](https://huggingface.co/microsoft/phi-2)
   - Make sure to download the GGUF format

3. **Make the llama.cpp executable available**:
   - Either add the llama.cpp directory to your PATH
   - Or specify the full path to the executable when using the framework

## Framework Files

Create two Python files:

### 1. `local_llm_framework.py`

This file contains the core framework for interacting with local LLMs. The full code is provided in the downloaded file.

### 2. `test_local_llm.py`

This file contains examples of how to use the framework. The full code is provided in the downloaded file.

## Usage Examples

### Basic Usage with Ollama

```python
from local_llm_framework import OllamaProvider, LLMClient, Message

# Create provider and client
provider = OllamaProvider(model_name="llama3")
client = LLMClient(provider)

# Simple prompt
response = client.prompt("Explain quantum computing in simple terms")
print(response.text)

# Save response to markdown
client.save_response_to_markdown(
    response,
    "quantum_computing.md",
    "Quantum Computing Explained"
)

# Chat conversation
messages = [
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="How do I learn Python?")
]
response = client.chat(messages)
print(response.text)
```

### Advanced Usage with llama.cpp

```python
from local_llm_framework import LlamaProvider, LLMClient

# Create provider with a specific model path
provider = LlamaProvider(
    model_path="/path/to/model.gguf",
    llama_cpp_path="/path/to/llama.cpp/main"  # Optional, will auto-detect if possible
)
client = LLMClient(provider)

# Generate text with specific parameters
response = client.prompt(
    "Write a short story about robots",
    max_tokens=2000,
    temperature=0.8,
    top_p=0.95
)
print(response.text)
```

## Troubleshooting

### Ollama Issues

- **Ollama not running**: Make sure the Ollama service is running in the background.
- **Model not found**: Check available models with `ollama list`.
- **Connection refused**: Ensure Ollama is running and listening on the default port (11434).

### llama.cpp Issues

- **Model not loading**: Ensure the model path is correct and the file exists.
- **Executable not found**: Specify the full path to the llama.cpp executable.
- **Out of memory**: Try using a smaller model or enable GPU acceleration.

### General Issues

- **ImportError**: Run `pip install requests` to install required dependencies.
- **High CPU usage**: This is normal for CPU-based inference. Consider using GPU acceleration.
- **Slow responses**: Adjust the model size or run on more powerful hardware.

## Advanced Configuration

### GPU Acceleration

For Ollama:
```python
provider = OllamaProvider(model_name="llama3")
response = client.prompt("Hello", gpu=True)  # Enable GPU
```

For llama.cpp:
```bash
# Build with CUDA support
cd llama.cpp
make LLAMA_CUBLAS=1
```

### Custom Formatting

```python
# Custom save function
def save_to_html(response, filename):
    with open(filename, "w") as f:
        f.write(f"<html><body><h1>AI Response</h1><p>{response.text}</p></body></html>")

# Use with client
response = client.prompt("Hello")
save_to_html(response, "output.html")
```

### Batch Processing

```python
prompts = [
    "Write a poem about spring",
    "Explain how a car engine works",
    "Describe quantum computing"
]

results = []
for prompt in prompts:
    results.append(client.prompt(prompt))
    
# Save all results
for i, response in enumerate(results):
    client.save_response_to_markdown(response, f"output_{i}.md")
```
