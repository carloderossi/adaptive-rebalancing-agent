from typing import List, Dict, Any
import ollama

from config_loader import Config


class LLMClient:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        # point ollama to configured host
        ollama.Client(host=self.config.ollama_host)

    def chat(
        self,
        role: str,
        messages: List[Dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        model = self.config.model_for(role)
        temperature = temperature if temperature is not None else self.config.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.default_max_tokens

        response = ollama.chat(
            model=model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
        return response["message"]["content"]