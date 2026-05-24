import json
import urllib.request
import os
import time

use_vertex = True
project = "codingagentproject-487605"
location = "us-central1"
model = "gemini-2.0-flash" 
token = os.getenv("GCLOUD_ACCESS_TOKEN")

url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent"
headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {token}'}

prompt = "Solve this coding task. Respond ONLY with the python content. Task: Write a hello world script."
# Vertex AI format for 'contents' is an array of objects
data = {
    "contents": [
        {
            "role": "user",
            "parts": [
                {"text": prompt}
            ]
        }
    ]
}

req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers)

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
