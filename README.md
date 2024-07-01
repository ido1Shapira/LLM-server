# Simple LLM server

## Requirements:
* GPU device with cuda version 12.2 installed.
* Visual studio building tools for compiling c/c++ code.
* Python 3.10 or above.

## How to install?
```
$ conda create -n LLM-server python=3.10
$ conda activate LLM-server
$ pip install -r requirements.txt
``` 
Next, go to `YOUR__init__.py` inside `src.models` and update the location of your models.
Last thing go to `YOUR_config.int` file and update the model name (this must much the model name you defined in `YOUR__init__.py`). 
You Are ready to go.

### The following query format:
```python
class ModelRequest(BaseModel):
    prompt: str
    temperature: float = 0
    max_new_tokens: int = 256
    streaming: bool = False
    stop: List[str] = ["<|end|>"]
```

#### Query full example:
```python
header = {
    "Content-Type": "application/json"
}

params = {
    "prompt": "what you can do?",
    "max_new_tokens": 15,
    "temperature": 1.0
}

base_url = 'http://127.0.0.1:80/'
```

### Use one of the 3 following api methods:
1. `model/` - return the model path that have been loaded. You can change it in the `config.ini` file.
```python
import httpx
print(httpx.get(base_url + 'model/', headers=header).text)

```
2. `invoke/` - invoke llm response.

```python
import httpx
invoke_url = base_url + 'invoke/'
print(httpx.post(invoke_url, json=params, headers=header).text)
```

3. `stream/` - stream the llm response.
```python
import httpx
stream_url = base_url + 'stream/'
with httpx.stream('POST', stream_url, json=params, headers=header) as r:
    for chunk in r.iter_raw():
        print(chunk.decode(), end='')
```