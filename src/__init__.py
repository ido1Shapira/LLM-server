from typing import List, Optional

from pydantic import BaseModel, Field


class ModelRequest(BaseModel):
    """
    A class representing a request for the LLM model.

    Attributes:
    - prompt (str): The input prompt for the LLM model.
    - system_prompt (Optional[str]): An optional system prompt for the LLM model. Default is None.
    - temperature (float): The temperature parameter for the LLM model. Default is 0.5.
    - max_new_tokens (int): The maximum number of new tokens to generate. Default is 256.
    - streaming (bool): A flag indicating whether to stream the response. Default is False.
    - stop (List[str]): A list of stop sequences. Default is ["<|end|>"]
    """
    prompt: str
    system_prompt: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.5)
    max_new_tokens: int = Field(default=256)
    streaming: bool = Field(default=False)
    stop: List[str] = Field(default=["<|end|>"])
