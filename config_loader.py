import os
from typing import Any, Dict

import yaml


class Config:
    def __init__(self, path: str = "config/models.yaml"):
        self.path = path
        self._config = self._load()

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Config file not found: {self.path}")
        with open(self.path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @property
    def ollama_host(self) -> str:
        return self._config.get("ollama", {}).get("host", "http://localhost:11434")

    def model_for(self, role: str) -> str:
        models = self._config.get("models", {})
        if role not in models:
            raise KeyError(f"No model configured for role '{role}' in {self.path}")
        return models[role]

    @property
    def default_temperature(self) -> float:
        return float(self._config.get("defaults", {}).get("temperature", 0.2))

    @property
    def default_max_tokens(self) -> int:
        return int(self._config.get("defaults", {}).get("max_tokens", 1024))