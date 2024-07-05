from typing import Dict, Any

from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

from src.models import ModelPath


def load_model(path: ModelPath, params: Dict[str, Any] = None) -> LlamaCpp:
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    return LlamaCpp(model_path=path.value, callback_manager=callback_manager, **params)
