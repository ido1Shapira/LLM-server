from abc import ABC, abstractmethod
from typing import Iterator, Optional

from llama_cpp import Llama, CreateCompletionResponse, CreateCompletionStreamResponse
from pydantic import BaseModel, Field, ConfigDict

from src import ModelRequest


class LLM(ABC, BaseModel):
    """
    An abstract base class representing an LLM model.

    Attributes:
    DEFAULT_SYSTEM_PROMPT (Optional[str]): The default system prompt for the LLM model.

    Methods:
    __init__(self, model_path: str, **kwargs): Initializes the LLM model.
    extract_text(response: CreateCompletionResponse | Iterator[CreateCompletionStreamResponse]) -> str: Extracts the text from the LLM response.
    __get_prompt(self, user_input: str, system_prompt: Optional[str]) -> str: Prepares the prompt for the LLM model.
    __call__(self, params: ModelRequest) -> str: Gets the response from the LLM model.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    DEFAULT_SYSTEM_PROMPT: Optional[str] = Field(default=None)
    llm: Llama = None
    model_path: str = Field(default=None)

    def __init__(self, model_path: str, **kwargs):
        super().__init__()
        self.llm = Llama(model_path, kwargs=kwargs)
        self.model_path = model_path

    @staticmethod
    def extract_text(response: CreateCompletionResponse | Iterator[CreateCompletionStreamResponse]) -> str:
        """
        Extracts the text from the LLM response.

        Parameters:
        response (CreateCompletionResponse | Iterator[CreateCompletionStreamResponse]): The LLM response.

        Returns:
        str: The extracted text.
        """
        return response.get('choices')[0].get('text')

    @abstractmethod
    def get_prompt(self, user_input: str, system_prompt: Optional[str]) -> str:
        """
        Prepares the prompt for the LLM model.

        This method is an abstract method and should be implemented in the child classes.
        It takes user input and an optional system prompt as parameters and returns a string
        that represents the prompt to be used for the LLM model.

        Parameters:
        user_input (str): The user input. This is the main input that will be processed by the LLM model.
        system_prompt (Optional[str]): An optional system prompt that can provide additional context or instructions to the LLM model.

        Returns:
        str: The prepared prompt. This prompt should be in a format that the LLM model can understand and generate appropriate responses.

        Raises:
        NotImplementedError: If this method is not implemented in the child classes, a NotImplementedError will be raised.
        """
        raise NotImplementedError

    def get_response(self, params: ModelRequest) -> CreateCompletionResponse | Iterator[CreateCompletionStreamResponse]:
        """
        Prepares and sends a request to the LLM model.

        This method takes a ModelRequest object as input, prepares the prompt using the user input and system prompt,
        and sends a request to the LLM model. The response from the LLM model is then returned.

        Parameters:
        params (ModelRequest): An object containing the parameters for the LLM model request.

        Returns:
        CreateCompletionResponse | Iterator[CreateCompletionStreamResponse]: The response from the LLM model.
        If the streaming parameter in the ModelRequest is True, an iterator of CreateCompletionStreamResponse objects is returned.
        Otherwise, a single CreateCompletionResponse object is returned.
        """
        prompt = self.get_prompt(params.prompt, params.system_prompt or self.DEFAULT_SYSTEM_PROMPT)
        return self.llm(
            prompt=prompt,
            max_tokens=params.max_new_tokens,
            stop=params.stop,
            temperature=params.temperature,
            stream=params.streaming
        )

    def __call__(self, params: ModelRequest) -> str:
        """
        Calls the LLM model and extracts the text from the response.

        This method takes a ModelRequest object as input, calls the LLM model using the get_response method,
        extracts the text from the response, and returns the extracted text.

        Parameters:
        params (ModelRequest): An object containing the parameters for the LLM model request.

        Returns:
        str: The extracted text from the LLM model response.
        """
        response = self.get_response(params)
        return self.extract_text(response)
