"""
Registry serves as a global storage for all configs that are used in the project.

Usage:
from src.common.registry import Registry

# Register a config
Registry.register("trainer_config", trainer_config)
Registry.register("model_config", model_config)
Registry.register("dataset_config", dataset_config)

# Access a config
trainer_config = Registry.get("trainer_config")
model_config = Registry.get("model_config")
dataset_config = Registry.get("dataset_config")
"""

from typing import Any, Dict


class Registry:
    _registry: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, config: Any) -> None:
        cls._registry[name] = config

    @classmethod
    def get(cls, name: str) -> Any:
        return cls._registry[name]
