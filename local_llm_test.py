# -----------------------------------------------------------------------------
# File: local_llm_test.py
# Author: Pat Joyce
# Email: joyce.pat@gmail.com
# Created: March 14, 2025
# 
# Copyright (c) 2025 Pat Joyce
# 
# Description:
# This file contains ....
# 
# License:
# This code is licensed ....
# -----------------------------------------------------------------------------

from local_llm_framework import OllamaProvider, LlamaProvider, LLMClient, Message
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_ollama():
    """Test using the Ollama provider."""
    print("\n==== Testing Ollama Provider ====")
    
    # Common Ollama models include: llama3, mistral, phi, gemma
    model_name = "tinyllama:latest"  # Replace with any model you have in Ollama
    
    try:
        provider = OllamaProvider(model_name=model_name)
        
        if not provider.check_server_availability():
            print("Error: Ollama server is not available. Please make sure it's running.")
            return
        else:
            print(f"Ollama server available. Using model '{model_name}'.")

        client = LLMClient(provider)
        
        # Simple prompt test
        print(f"\nTesting simple prompt with {model_name}...")
        prompt = "Explain what a transformer model is in simple terms."
        
        response = client.prompt(prompt, temperature=0.7)
        
        print(f"Model: {response.model}")
        print(f"Token usage: {response.usage}")
        print(f"\nResponse:\n{response.text}")
        
        # Save to markdown
        client.save_response_to_markdown(
            response, 
            "transformer_explanation.md",
            "Transformer Models Explained"
        )
        
        # Test conversation
        print("\nTesting conversation...")
        messages = [
            Message(role="system", content="You are a helpful programming assistant."),
            Message(role="user", content="How do I read a CSV file in Python?"),
            Message(role="assistant", content="You can read a CSV file in Python using the `csv` module from the standard library or the more powerful `pandas` library."),
            Message(role="user", content="Show me an example with pandas.")
        ]
        
        response = client.chat(messages, temperature=0.7)
        
        print(f"Model: {response.model}")
        print(f"Token usage: {response.usage}")
        print(f"\nResponse:\n{response.text}")
        
        # Save to markdown
        client.save_response_to_markdown(
            response, 
            "pandas_csv_example.md",
            "Reading CSV Files with Pandas"
        )
        
    except Exception as e:
        print(f"Error testing Ollama: {str(e)}")

def test_llama_cpp():
    """Test using the LLaMA.cpp provider."""
    print("\n==== Testing LLaMA.cpp Provider ====")
    
    # Example path - replace with your actual path to the model
    model_path = "/path/to/your/model.gguf"
    
    # Check if model exists before attempting
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        print("Please update the model_path variable with your correct model path.")
        return
    
    try:
        provider = LlamaProvider(model_path=model_path)
        client = LLMClient(provider)
        
        # Simple prompt test
        print(f"\nTesting prompt with local LLaMA model...")
        prompt = "Write a short poem about artificial intelligence."
        
        response = client.prompt(prompt, temperature=0.7)
        
        print(f"Model: {response.model}")
        print(f"Token usage: {response.usage}")
        print(f"\nResponse:\n{response.text}")
        
        # Save to markdown
        client.save_response_to_markdown(
            response, 
            "ai_poem.md",
            "AI-Generated Poem"
        )
        
    except Exception as e:
        print(f"Error testing LLaMA.cpp: {str(e)}")

if __name__ == "__main__":
    print("Local LLM Framework Test")
    print("========================")
    
    # Test Ollama (more common and easier to set up)
    test_ollama()
    
    # Uncomment to test LLaMA.cpp if you have it installed
    # test_llama_cpp()
    
    print("\nTests completed!")
