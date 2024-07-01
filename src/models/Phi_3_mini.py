from typing import Optional

from src.llm import LLM


class Phi3Mini(LLM):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)

    def get_prompt(self, user_input: str, system_prompt: Optional[str]) -> str:
        return f"<|user|>\n{user_input}<|end|>\n<|assistant|>"
