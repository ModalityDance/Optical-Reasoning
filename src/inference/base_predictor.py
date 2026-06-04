from __future__ import annotations

from typing import Any


class BasePredictor:
    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def predict(self, input_data: dict[str, Any], prompt: str) -> dict[str, Any]:
        raise NotImplementedError

    def get_model_info(self) -> dict[str, str]:
        return {"model": self.model, "base_url": self.base_url}
