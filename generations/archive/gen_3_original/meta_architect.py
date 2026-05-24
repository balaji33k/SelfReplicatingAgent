import json
import logging
import os
import urllib.request
import time

logger = logging.getLogger("MetaArchitect")

class MetaArchitect:
    def __init__(self):
        self.model = "gemini-2.0-flash"
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def solve_task(self, task):
        # Support real problem statements and test cases
        description = task.get('problem_statement') or task.get('prompt') or task.get('question')
        test_info = task.get('test_cases', "Ensure logic corresponds to the problem.")
        
        if not description:
            description = f"Write a logic to verify {task.get('id')} with difficulty {task.get('difficulty')}."
            
        prompt = f"""Solve this coding task. 
Respond ONLY with the python content. 
Ensure the script includes verification/assertion logic (Test Cases) as part of the script execution.

Task: {description}
Verification Requirements: {test_info}
"""
        return self._call_llm(prompt)

    def synthesize_offspring(self, current_gen_num, results):
        context = self._prepare_context()
        prompt = f"""Architect Generation {current_gen_num+1}.
        
        CONSTRAINTS:
        1. MANDATORY: Respond ONLY with a pure JSON object.
        2. KEYS: run.py, meta_architect.py, env.py, spawner.py, topology (array), improvement_log (string).
        """
        response_text = self._call_llm(prompt)
        offspring_data = self._parse_json(response_text)
        
        manifest = {
            "GENERATION": current_gen_num + 1,
            "TOPOLOGY": offspring_data.get("topology", []),
            "IMPROVEMENT_LOG": offspring_data.get("improvement_log", "Standard evolutionary refinement.")
        }
        logic_files = {k: v for k, v in offspring_data.items() if k.endswith('.py')}
        return manifest, logic_files

    def _prepare_context(self):
        files = {}
        for f in ["run.py", "meta_architect.py"]:
            try:
                with open(f, "r") as src: files[f] = src.read()
            except: pass
        return files

    def _call_llm(self, prompt):
        use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI") == "TRUE"
        if use_vertex:
            project = os.getenv("GOOGLE_CLOUD_PROJECT", "codingagentproject-487605")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            token = os.getenv("GCLOUD_ACCESS_TOKEN")
            url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{self.model}:generateContent"
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {token}'}
            
            # Vertex AI JSON structure
            data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}]
                    }
                ]
            }
        else:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
            headers = {'Content-Type': 'application/json'}
            data = {"contents": [{"parts": [{"text": prompt}]}]}

        req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers)
        
        for attempt in range(7):
            try:
                with urllib.request.urlopen(req) as res:
                    resp_json = json.loads(res.read().decode('utf-8'))
                    return resp_json['candidates'][0]['content']['parts'][0]['text']
            except Exception as e:
                wait_time = (5 * (2 ** attempt))
                logger.error(f"LLM Call failed (Attempt {attempt+1}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        return ""

    def _parse_json(self, text):
        try:
            start = text.find('{'); end = text.rfind('}')
            return json.loads(text[start:end+1]) if (start != -1 and end != -1) else json.loads(text)
        except: return {}
