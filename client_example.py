"""
This script demonstrates how to use the LLM server client to interact with the LLM model.

The client uses the `httpx` library to make HTTP requests to the LLM server.

The script performs the following actions:
1. Sends a GET request to the `/model/` endpoint to retrieve the name of the loaded LLM model.
2. Sends a POST request to the `/invoke/` endpoint to invoke an LLM response.
3. Sends a POST request to the `/stream/` endpoint to stream the LLM response.

The script prints the responses received from the server.

Note: Make sure to replace the `base_url` variable with the actual base URL of the LLM server.
"""

import httpx

timeout = 60.0

# Define the request headers
header = {
    "Content-Type": "application/json"
}

# Define the request parameters
params = {
    "prompt": "what you can do?",
    "max_new_tokens": 15,
    "temperature": 0.0
}

# Define the base URL of the LLM server
base_url = 'http://127.0.0.1:80/'

# Define the URLs for the invoke and stream endpoints
invoke_url = base_url + 'invoke/'
stream_url = base_url + 'stream/'

# Send a GET request to the model endpoint
print(httpx.get(base_url + 'model/', headers=header).text)

# Send a POST request to the invoke endpoint
print(httpx.post(invoke_url, json=params, headers=header, timeout=timeout).text)

# Send a POST request to the stream endpoint and print the streamed response
with httpx.stream('POST', stream_url, json=params, headers=header, timeout=timeout) as r:
    for chunk in r.iter_raw():
        print(chunk.decode(), end='')