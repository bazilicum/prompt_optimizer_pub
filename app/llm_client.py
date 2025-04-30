import requests
import json
from config import LLM_TEMPERATURE,LLM_REPETITION_PENALTY,LLM_TOP_P,LLM_TIMEOUT

class LocalAIClient:
    """
    A simple client for LocalAI that calls the completions and chat compleations endpoints.
    """
    def __init__(self, host, port, model):
        self.base_url = f"http://{host}:{port}"
        self.model = model

    def chat_generate(self, prompt: str, message_list: dict, max_tokens: int = 500) -> str:
        message_list.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": message_list,
            "max_tokens": max_tokens,
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
            "repetition_penalty": LLM_REPETITION_PENALTY,
            "stop": ["<|im_end|>", "<|endoftext|>"]
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(self.base_url+"/v1/chat/completions", headers=headers, data=json.dumps(payload), timeout=LLM_TIMEOUT)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"LocalAI request failed: {e}")
            return ""
        
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
            "repetition_penalty": LLM_REPETITION_PENALTY,     
            "stop": ["<|im_end|>", "<|endoftext|>"]
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(self.base_url+"/v1/completions", headers=headers, data=json.dumps(payload), timeout=LLM_TIMEOUT)
            response.raise_for_status()
            return response.json()["choices"][0]["text"].strip()
        except Exception as e:
            print(f"LocalAI request failed: {e}")
            return ""


class OpenAIClient:
    """
    A stab for OpenAI client
    """
    def __init__(self, host, port, model):
        self.base_url = f"http://{host}:{port}"
        self.model = model

    def chat_generate(self, prompt: str, message_list: dict, max_tokens: int = 500) -> str:
        return ""
        
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        return ""



def get_llm_client(provider="LocalAI", **kwargs) -> LocalAIClient:
    if provider == "LocalAI":
        return LocalAIClient(**kwargs)
    elif provider=='OpenAI':  #This is just a stab. TBI
        return OpenAIClient(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}") 