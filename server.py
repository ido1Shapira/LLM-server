from starlette.responses import StreamingResponse

from src import ModelRequest
from src.llm import LLM


def server(app, llm: LLM):
    @app.post("/invoke")
    async def invoke(request: ModelRequest):
        """
        Invoke a response from the LLM model.

        Parameters:
        request (ModelRequest): The request parameters.

        Returns:
        str: The extracted text from the LLM response.
        """
        return llm(request)

    def get_response_generator(request: ModelRequest):
        """
        Get a generator for the LLM response.

        Parameters:
        request (ModelRequest): The request parameters.

        Returns:
        Generator[str, None, None]: A generator yielding the extracted text from each LLM response event.
        """
        request.streaming = True
        output_stream = llm.get_response(request)

        for event in output_stream:
            current_response = llm.extract_text(event)
            yield current_response

    @app.post("/stream")
    async def stream(request: ModelRequest):
        """
        Stream the LLM response.

        Parameters:
        request (ModelRequest): The request parameters.

        Returns:
        StreamingResponse: A streaming response containing the LLM response events.
        """
        return StreamingResponse(
            get_response_generator(request),
            media_type='text/event-stream',
        )
