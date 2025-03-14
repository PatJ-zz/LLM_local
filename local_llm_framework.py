# -----------------------------------------------------------------------------
# File: local_llm_framework.py
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

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import subprocess
import tempfile
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Represents a message in a conversation with an LLM."""
    role: str
    content: str

@dataclass
class LLMResponse:
    """Structured response from an LLM."""
    text: str
    model: str
    usage: Dict[str, int]
    raw_response: Dict[str, Any]

class LocalLLMProvider:
    """Base class for local LLM providers."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
    
    def generate_text(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text from a simple prompt."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def generate_chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Generate a response from a list of messages."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def _count_tokens(self, text: str) -> int:
        """Simple token counting approximation."""
        # This is a very rough approximation
        # Real tokenizers would be more accurate
        return len(text.split())

class LlamaProvider(LocalLLMProvider):
    """Provider for running local LLaMA models."""
    
    def __init__(self, model_path: str, llama_cpp_path: Optional[str] = None):
        super().__init__(model_path)
        self.llama_cpp_path = llama_cpp_path or self._find_llama_cpp()
        if not self.llama_cpp_path:
            raise ValueError("Could not find llama.cpp executable. Please specify the path.")
    
    def _find_llama_cpp(self) -> Optional[str]:
        """Attempt to find llama.cpp executable in PATH."""
        for executable in ["llama-cli", "main"]:
            path = subprocess.run(["which", executable], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
            if path.returncode == 0:
                return path.stdout.decode('utf-8').strip()
        return None
    
    def generate_text(self, 
                      prompt: str, 
                      max_tokens: int = 1000,
                      temperature: float = 0.7,
                      **kwargs) -> LLMResponse:
        """Generate text using a local LLaMA model."""
        
        # Prepare the command
        cmd = [
            self.llama_cpp_path,
            "-m", self.model_path,
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "--prompt", prompt
        ]
        
        # Add additional parameters
        for key, value in kwargs.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error running llama.cpp: {result.stderr}")
                raise RuntimeError(f"llama.cpp failed with error: {result.stderr}")
            
            output = result.stdout.strip()
            
            # Remove the original prompt from the output
            if output.startswith(prompt):
                output = output[len(prompt):].strip()
            
            # Calculate token usage
            prompt_tokens = self._count_tokens(prompt)
            completion_tokens = self._count_tokens(output)
            
            return LLMResponse(
                text=output,
                model=os.path.basename(self.model_path),
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                },
                raw_response={"output": output}
            )
            
        except Exception as e:
            logger.error(f"Error running local LLM: {str(e)}")
            raise
    
    def generate_chat(self, 
                      messages: List[Message], 
                      max_tokens: int = 1000,
                      temperature: float = 0.7,
                      **kwargs) -> LLMResponse:
        """Convert chat messages to a prompt and generate a response."""
        
        # Convert messages to a prompt format LLaMA can understand
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"<|system|>\n{msg.content}\n"
            elif msg.role == "user":
                prompt += f"<|user|>\n{msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"<|assistant|>\n{msg.content}\n"
        
        # Add the final assistant prompt to trigger generation
        prompt += "<|assistant|>\n"
        
        return self.generate_text(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

class OllamaProvider(LocalLLMProvider):
    """Provider for running models through Ollama."""
    
    def __init__(self, model_name: str, ollama_host: str = "http://localhost:11434"):
        # For Ollama, model_path is just the model name
        self.model_name = model_name
        self.ollama_host = ollama_host
        self._check_ollama_installed()
    
    def check_server_availability(self) -> bool:
        """Check if the Ollama server is available and responding."""
        try:
            import requests
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error connecting to Ollama server: {str(e)}")
            return False
        
    def _check_ollama_installed(self):
        """Check if Ollama is installed and running."""
        try:
            import requests
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code != 200:
                raise ValueError(f"Ollama server not responding properly at {self.ollama_host}")
            
            # Check if our model exists
            models = response.json().get("models", [])
            model_names = [model.get("name") for model in models]
            if self.model_name not in model_names:
                logger.warning(f"Model '{self.model_name}' not found in Ollama. "
                              f"Available models: {', '.join(model_names)}")
                
        except ImportError:
            raise ImportError("Please install the requests library: pip install requests")
        except Exception as e:
            raise ValueError(f"Error connecting to Ollama: {str(e)}")
    
    def generate_text(self, 
                      prompt: str, 
                      max_tokens: int = 1000,
                      temperature: float = 0.7,
                      **kwargs) -> LLMResponse:
        """Generate text using Ollama API."""
        try:
            import requests
            
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }
            
            # Add any extra options
            for key, value in kwargs.items():
                if key not in data["options"]:
                    data["options"][key] = value
            
            logger.info(f"Sending request to Ollama for model {self.model_name}")
            response = requests.post(f"{self.ollama_host}/api/generate", json=data)
            response.raise_for_status()
            result = response.json()
            
            prompt_tokens = self._count_tokens(prompt)
            completion_tokens = self._count_tokens(result.get("response", ""))
            
            return LLMResponse(
                text=result.get("response", ""),
                model=self.model_name,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                },
                raw_response=result
            )
            
        except ImportError:
            raise ImportError("Please install the requests library: pip install requests")
        except Exception as e:
            logger.error(f"Error with Ollama API: {str(e)}")
            raise
    
    def generate_chat(self, 
                     messages: List[Message], 
                     max_tokens: int = 1000,
                     temperature: float = 0.7,
                     **kwargs) -> LLMResponse:
        """Generate a chat response using Ollama API."""
        try:
            import requests
            
            # Convert our messages to Ollama format
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            data = {
                "model": self.model_name,
                "messages": formatted_messages,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }
            
            # Add any extra options
            for key, value in kwargs.items():
                if key not in data["options"]:
                    data["options"][key] = value
            
            logger.info(f"Sending chat request to Ollama for model {self.model_name}")
            response = requests.post(f"{self.ollama_host}/api/chat", json=data)
            response.raise_for_status()
            result = response.json()
            
            # Calculate token usage
            prompt_text = "\n".join([m.content for m in messages])
            prompt_tokens = self._count_tokens(prompt_text)
            completion_tokens = self._count_tokens(result.get("message", {}).get("content", ""))
            
            return LLMResponse(
                text=result.get("message", {}).get("content", ""),
                model=self.model_name,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                },
                raw_response=result
            )
            
        except ImportError:
            raise ImportError("Please install the requests library: pip install requests")
        except Exception as e:
            logger.error(f"Error with Ollama API: {str(e)}")
            raise

class LLMClient:
    """Client for interacting with different LLM providers."""
    
    def __init__(self, provider: Union[LocalLLMProvider, Any]):
        self.provider = provider
    
    def chat(self, 
             messages: List[Union[Message, Dict[str, str]]], 
             **kwargs) -> LLMResponse:
        """Send a conversation to the LLM and get a response."""
        
        # Convert dict messages to Message objects if needed
        processed_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                processed_messages.append(Message(role=msg["role"], content=msg["content"]))
            else:
                processed_messages.append(msg)
        
        return self.provider.generate_chat(processed_messages, **kwargs)
    
    def prompt(self, prompt: str, **kwargs) -> LLMResponse:
        """Simple helper to send a single user prompt."""
        return self.chat([Message(role="user", content=prompt)], **kwargs)
    
    def save_response_to_markdown(self, response: LLMResponse, filename: str, 
                                 title: str = "LLM Generated Content"):
        """Save a response to a markdown file."""
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write(f"Generated using model: {response.model}\n\n")
            f.write("---\n\n")
            f.write(response.text)
        
        logger.info(f"Content saved to {filename}")
        return filename
