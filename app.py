import configparser
import os

import uvicorn
from fastapi import FastAPI
from starlette.responses import StreamingResponse

from src import ModelRequest
from src.models import ModelPath
from src.models.model_utils import load_model

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
model_name = config.get('Model', 'model_name')
verbose = bool(config.get('ModelParams', 'verbose'))
llm = load_model(path=ModelPath[model_name], verbose=verbose)


@app.post("/invoke/")
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