from src.llm import LLM
from src.models import ModelPath, pathToModel


def load_model(path: ModelPath, **kwargs) -> LLM:
    return pathToModel[path](model_path=path.value, kwargs=kwargs)
