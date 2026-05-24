import os
import urllib.request
import json
import traceback

api_key = os.getenv("GEMINI_API_KEY")
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

try:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as res:
        data = json.loads(res.read().decode('utf-8'))
        print([m['name'] for m in data.get('models', []) if 'gemini' in m['name']])
except Exception as e:
    print("FAILED:", e)
    if hasattr(e, 'read'):
        print(e.read().decode('utf-8'))
