from typing import Optional

from src.llm import LLM


class Llama3Instruct(LLM):
    DEFAULT_SYSTEM_PROMPT: str = "You are Llama-3-instruct. AI agent"

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)

    def get_prompt(self, user_input: str, system_prompt: Optional[str]) -> str:
        return f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {system_prompt or self.DEFAULT_SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>
        {user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
