import configparser
import os

import uvicorn
from fastapi import FastAPI

from src.models import ModelPath

# Load configuration from config.ini file
config = configparser.ConfigParser()
config.read(os.path.join(os.getcwd(), 'config.ini'))
is_using_langchain = eval(config.get('Application', 'using_langchain'))
if is_using_langchain:
    from src.models.langchain_model_utils import load_model
else:
    from src.models.model_utils import load_model

# Create a FastAPI application
app = FastAPI(
    title="LLM Server",
    description="A simple server for running LLM models.",
    version="1.0.0",
)

# Load the LLM model
model_name = config.get('Model', 'model_name')
model_params = {
    "verbose": eval(config.get('ModelParams', 'verbose')),
    "n_ctx": eval(config.get('ModelParams', 'n_ctx')),
    "n_gpu_layers": eval(config.get('ModelParams', 'n_gpu_layers'))
}
llm = load_model(path=ModelPath[model_name], params=model_params)


@app.get("/model")
async def get_model_name():
    """
    Get the name of the loaded LLM model.

    Returns:
    dict: A dictionary containing the model path.
    """
    return {"model_path": llm.model_path}


if is_using_langchain:
    from langserve import add_routes
    llm.with_config({"run_name": "agent"})
    # Adds routes to the app for using the chain under:
    # /invoke
    # /batch
    # /stream
    # /stream_events
    add_routes(app, llm)
else:
    from server import server

    server(app, llm)

if __name__ == "__main__":
    # Run the FastAPI application using uvicorn
    host = config.get('Application', 'host')
    port = int(config.get('Application', 'port'))
    uvicorn.run("app:app", host=host, port=port)
