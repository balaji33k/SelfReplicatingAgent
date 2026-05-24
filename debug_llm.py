import json
import urllib.request
import os
import time

api_key = os.getenv("GEMINI_API_KEY")
model = "gemini-flash-latest"
url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
prompt = "Solve this coding task. Respond ONLY with the python content. Task: Write a hello world script."
data = {"contents": [{"parts": [{"text": prompt}]}]}
req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req) as res:
        response = json.loads(res.read().decode('utf-8'))
        print("RESPONSE:", json.dumps(response, indent=2))
        text = response['candidates'][0]['content']['parts'][0]['text']
        print("TEXT:", text)
except Exception as e:
    print("ERROR:", e)
    if hasattr(e, 'read'):
        print("ERROR BODY:", e.read().decode())
