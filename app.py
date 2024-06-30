import configparser
import os

import uvicorn
from fastapi import FastAPI
from starlette.responses import StreamingResponse

from model_utils import load_model, ModelRequest, get_response, extract_text

# Create a FastAPI application
app = FastAPI(
    title="LLM Server",
    description="A simple server for running LLM models using FastAPI and uvicorn.",
    version="1.0.0",
)

# Load configuration from config.ini file
config = configparser.ConfigParser()
config.read(os.path.join(os.getcwd(), 'config.ini'))

# Load the LLM model
model_path = config.get('Model', 'path')
verbose = bool(config.get('ModelParams', 'verbose'))
llm = load_model(path=model_path, kwargs={'verbose': verbose})


@app.post("/generate/")
async def generate(request: ModelRequest):
    """
    Generate a response from the LLM model.

    Parameters:
    request (ModelRequest): The request parameters.

    Returns:
    str: The extracted text from the LLM response.
    """
    output = get_response(llm, request)
    return extract_text(output)


def get_response_generator(request: ModelRequest):
    """
    Get a generator for the LLM response.

    Parameters:
    request (ModelRequest): The request parameters.

    Returns:
    Generator[str, None, None]: A generator yielding the extracted text from each LLM response event.
    """
    request.streaming = True
    output_stream = get_response(llm, request)

    for event in output_stream:
        current_response = extract_text(event)
        yield current_response


@app.post("/stream/")
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


@app.get("/model/")
async def get_model_name():
    """
    Get the name of the loaded LLM model.

    Returns:
    dict: A dictionary containing the model path.
    """
    return {"model_path": llm.model_path}


if __name__ == "__main__":
    # Run the FastAPI application using uvicorn
    host = config.get('Application', 'host')
    port = int(config.get('Application', 'port'))
    uvicorn.run("app:app", host=host, port=port, reload=True)