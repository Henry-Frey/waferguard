"""Component registry for dynamic construction."""

from __future__ import annotations

from typing import Any, Callable


class Registry:
    def __init__(self, name: str):
        self._name = name
        self._registry: dict[str, Callable] = {}

    def register(self, name: str | None = None) -> Callable:
        def decorator(cls: Callable) -> Callable:
            key = name or cls.__name__.lower()
            if key in self._registry:
                raise ValueError(f"{key} already registered in {self._name}")
            self._registry[key] = cls
            return cls
        return decorator

    def build(self, name: str, **kwargs: Any) -> Any:
        if name not in self._registry:
            available = ", ".join(sorted(self._registry))
            raise KeyError(f"Unknown {self._name} '{name}'. Available: {available}")
        return self._registry[name](**kwargs)


BACKBONES = Registry("backbone")
