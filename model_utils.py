import configparser
import os
from typing import List, Callable, Iterator

from llama_cpp import Llama, CreateCompletionResponse, CreateCompletionStreamResponse
from pydantic import BaseModel

# Load configuration from config.ini file
config = configparser.ConfigParser()
config.read(os.path.join(os.getcwd(), 'config.ini'))


def extract_text(response: CreateCompletionResponse | Iterator[CreateCompletionStreamResponse]) -> str:
    """
    Extracts the text from the LLM response.

    Parameters:
    response (CreateCompletionResponse | Iterator[CreateCompletionStreamResponse]): The LLM response.

    Returns:
    str: The extracted text.
    """
    return response.get('choices')[0].get('text')


def get_prompt(user_input: str) -> str:
    """
    Prepares the prompt for the LLM model.

    Parameters:
    user_input (str): The user input.

    Returns:
    str: The prepared prompt.
    """
    return f"<|user|>\n{user_input}<|end|>\n<|assistant|>"


def load_model(path: str, kwargs) -> Llama:
    """
    Loads the LLM model.

    Parameters:
    path (str): The path to the LLM model.
    kwargs (dict): Additional keyword arguments for model initialization.

    Returns:
    Llama: The loaded LLM model.
    """
    return Llama(model_path=path, kwargs=kwargs)


class ModelRequest(BaseModel):
    """
    A class representing a request for the LLM model.

    Attributes:
    prompt (str): The input prompt for the LLM model.
    temperature (float): The temperature parameter for the LLM model. Default value is read from the config file.
    max_new_tokens (int): The maximum number of new tokens to generate. Default value is read from the config file.
    streaming (bool): A flag indicating whether to stream the response. Default value is False.
    stop (List[str]): A list of stop sequences. Default value is read from the config file.
    """
    prompt: str
    temperature: float = config.get('ModelDefaultRequest', 'temperature')
    max_new_tokens: int = config.get('ModelDefaultRequest', 'max_new_tokens')
    streaming: bool = False
    stop: List[str] = config.get('ModelDefaultRequest', 'stop')


def get_response(llm: Callable, params: ModelRequest) -> CreateCompletionResponse | Iterator[
    CreateCompletionStreamResponse]:
    """
    Gets the response from the LLM model.

    Parameters:
    llm (Callable): The LLM model.
    params (ModelRequest): The request parameters.

    Returns:
    CreateCompletionResponse | Iterator[CreateCompletionStreamResponse]: The LLM response.
    """
    return llm(
        get_prompt(params.prompt),
        max_tokens=params.max_new_tokens,
        stop=params.stop,
        temperature=params.temperature,
        stream=params.streaming
    )
