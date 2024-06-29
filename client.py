import httpx

header = {
    "Content-Type": "application/json"
}

params = {
    "prompt": "what you can do?",
    "max_new_tokens": 15,
    "temperature": 1.0
}

base_url = 'http://127.0.0.1:80/'
generate_url = base_url + 'generate/'
stream_url = base_url + 'stream/'

print(httpx.get(base_url + 'model/', headers=header).text)
print(httpx.post(generate_url, json=params, headers=header).text)

with httpx.stream('POST', stream_url, json=params, headers=header) as r:
    for chunk in r.iter_raw():
        print(chunk.decode(), end='')

