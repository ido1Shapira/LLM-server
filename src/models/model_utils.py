from typing import Dict, Any

from src.llm import LLM
from src.models import ModelPath, pathToModel


def load_model(path: ModelPath, params: Dict[str, Any] = None) -> LLM:
    params = params or {}
    return pathToModel[path](model_path=path.value, **params)
