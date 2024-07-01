from enum import Enum
from typing import Dict, Type

from src.llm import LLM
from src.models import Phi3Mini, Llama3Instruct


class ModelPath(Enum):
    model_name = "<--pathToModel/model_name.gguf-->"
    # for example:
    phi_3_mini = r"C:\Users\Ido\.cache\huggingface\hub\Phi-3-mini-4k-instruct-fp16.gguf"
    llama_3_instruct = r"C:\Users\Ido\.cache\huggingface\hub\Llama-3-Instruct-8B-SPPO-Iter3-Q6_K_L.gguf"


pathToModel: Dict[ModelPath, Type[LLM]] = {
    ModelPath.model_name: LLM,
    # for example:
    ModelPath.phi_3_mini: Phi3Mini,
    ModelPath.llama_3_instruct: Llama3Instruct
}
