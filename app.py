# import configparser

import uvicorn
from fastapi import FastAPI
from starlette.responses import StreamingResponse

from model_utils import load_model, ModelRequest, get_response

app = FastAPI()
# config = configparser.ConfigParser()
# config.read('config.ini')
# model_path = config.get('Model', 'path')
model_path = r'C:\Users\Ido\.cache\huggingface\hub\Phi-3-mini-4k-instruct-fp16.gguf'

llm = load_model(path=model_path, kwargs={'verbose': True})


@app.post("/generate/")
async def generate(request: ModelRequest):
    output = get_response(llm, request)
    return output['choices'][0]['text']


def get_response_generator(request: ModelRequest):
    request.streaming = True
    output_stream = get_response(llm, request)

    for event in output_stream:
        current_response = event["choices"][0]['text']
        yield current_response


@app.post("/stream/")
async def stream(request: ModelRequest):
    return StreamingResponse(
        get_response_generator(request),
        media_type='text/event-stream')


@app.get("/model/")
async def get_model_name():
    return {"model_path": llm.model_path}


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=80, reload=True)
