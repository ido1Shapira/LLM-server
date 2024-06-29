from typing import List, Callable

from llama_cpp import Llama
from pydantic import BaseModel


def get_prompt(user_input: str) -> str:
    return f"<|user|>\n{user_input}<|end|>\n<|assistant|>"


def load_model(path: str, kwargs) -> Llama:
    return Llama(model_path=path, kwargs=kwargs)


class ModelRequest(BaseModel):
    prompt: str
    temperature: float = 0
    max_new_tokens: int = 256
    streaming: bool = False
    stop: List[str] = ["<|end|>"]


def get_response(llm: Callable, params: ModelRequest):
    return llm(
        get_prompt(params.prompt),
        max_tokens=params.max_new_tokens,
        stop=params.stop,
        temperature=params.temperature,
        stream=params.streaming
    )
