import json
import logging
import os
import sys
import urllib.request
import time

logger = logging.getLogger("MetaArchitect")

class MetaArchitect:
    def __init__(self):
        self.model = "gemini-2.0-flash"
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def solve_task(self, task):
        description = task.get('problem_statement') or task.get('prompt') or task.get('question')
        test_info = task.get('test_cases', "Ensure the logic is correct and handles edge cases.")

        if not description:
            description = f"Write a Python solution for task {task.get('id')} with difficulty {task.get('difficulty')}."

        prompt = f"""Solve this coding task. Respond ONLY with Python code — no markdown fences, no explanation.
The script must include assertion-based test cases that verify correctness when executed.
Think step by step before writing the solution.

Task: {description}
Verification Requirements: {test_info}
"""
        return self._call_llm(prompt)

    def _analyze_failures(self, results):
        total = len(results)
        passed = sum(1 for r in results if r['status'] == 'success')
        failed = [r for r in results if r['status'] != 'success']

        error_buckets = {}
        for r in failed:
            logs = r.get('logs', '')
            if 'ModuleNotFoundError' in logs:
                mod = 'unknown'
                if "No module named '" in logs:
                    mod = logs.split("No module named '")[1].split("'")[0]
                error_buckets.setdefault('import_error', []).append({
                    'id': r['id'], 'missing_module': mod
                })
            elif 'SyntaxError' in logs:
                error_buckets.setdefault('syntax_error', []).append({'id': r['id']})
            elif 'AssertionError' in logs:
                error_buckets.setdefault('assertion_error', []).append({'id': r['id']})
            elif r['status'] == 'timeout':
                error_buckets.setdefault('timeout', []).append({'id': r['id']})
            else:
                snippet = logs[-300:] if len(logs) > 300 else logs
                error_buckets.setdefault('runtime_error', []).append({
                    'id': r['id'], 'snippet': snippet
                })

        recommendations = []
        if 'import_error' in error_buckets:
            missing_mods = list({e['missing_module'] for e in error_buckets['import_error']})
            recommendations.append(
                f"AUTO-INSTALL missing modules at startup in run.py: {missing_mods}. "
                "Use subprocess to call pip install before the benchmark loop."
            )
        if 'syntax_error' in error_buckets:
            recommendations.append(
                "env.py must strip markdown fences (```python ... ```) from LLM output before execution. "
                "Add a _strip_markdown() method in the Sandbox class."
            )
        if 'assertion_error' in error_buckets:
            recommendations.append(
                "Solutions fail assertions — improve solve_task() to use chain-of-thought (CoT) prompting: "
                "ask the LLM to think step by step before writing code."
            )
        if 'timeout' in error_buckets:
            recommendations.append(
                "Some tasks timed out — increase the sandbox timeout limit and add a complexity guard "
                "in solve_task() to warn the LLM about time constraints."
            )
        if 'runtime_error' in error_buckets:
            recommendations.append(
                "Runtime errors detected — improve error reporting in env.py to capture full tracebacks "
                "and feed them back to the LLM for retry."
            )

        return {
            'pass_rate': f"{passed}/{total}",
            'pass_percentage': round(100 * passed / max(1, total), 1),
            'error_breakdown': error_buckets,
            'recommendations': recommendations,
            'sample_failures': [
                {
                    'id': r['id'],
                    'status': r['status'],
                    'error_tail': r['logs'][-400:] if r.get('logs') else ''
                }
                for r in failed[:5]
            ]
        }

    def synthesize_offspring(self, current_gen_num, results):
        failure_analysis = self._analyze_failures(results)
        current_source = self._prepare_context()

        prompt = f"""You are designing Generation {current_gen_num + 1} of a self-improving AI coding agent system.

## MISSION
Generation {current_gen_num} just finished executing coding benchmarks. Your job is to analyze
the failures and CREATE a genuinely improved architecture for the next generation.

## PERFORMANCE REPORT FROM GENERATION {current_gen_num}
{json.dumps(failure_analysis, indent=2)}

## CURRENT SOURCE CODE
Study the architecture below carefully. Understand how it works.
Then CREATE BETTER code that fixes the identified failures.
DO NOT copy this code — write improved versions of each file.

{json.dumps(current_source, indent=2)}

## REQUIREMENTS FOR GENERATION {current_gen_num + 1}
1. Fix EVERY root cause listed in the recommendations above
2. solve_task() must use chain-of-thought (CoT) prompting for better solutions
3. env.py Sandbox must handle all identified failure modes (import errors, syntax errors, timeouts)
4. synthesize_offspring() MUST include failure analysis in the next generation's synthesis prompt
5. The gen_num variable in run.py MUST be set to {current_gen_num + 1}
6. CRITICAL: Do NOT copy the current code. Write improved logic that addresses the failures.
7. If import_error was detected, add auto-install logic for missing packages at startup in run.py

## OUTPUT FORMAT
Respond ONLY with a valid JSON object. No text before or after. Keys must be exactly:
- "run.py": complete content of the new run.py
- "meta_architect.py": complete content of the new meta_architect.py
- "env.py": complete content of the new env.py
- "spawner.py": complete content of the new spawner.py
- "topology": array of objects with "id" and "label" keys describing the pipeline stages
- "improvement_log": string summarizing exactly what you changed and why
"""
        response_text = self._call_llm(prompt)
        offspring_data = self._parse_json(response_text)

        if not offspring_data:
            logger.warning("First synthesis attempt returned empty JSON. Retrying...")
            response_text = self._call_llm(prompt)
            offspring_data = self._parse_json(response_text)

        if not offspring_data:
            raise RuntimeError("LLM failed to return valid JSON for offspring after 2 attempts.")

        manifest = {
            "GENERATION": current_gen_num + 1,
            "TOPOLOGY": offspring_data.get("topology", []),
            "IMPROVEMENT_LOG": offspring_data.get("improvement_log", ""),
            "PARENT_PASS_RATE": failure_analysis['pass_rate'],
            "PARENT_ERRORS": list(failure_analysis['error_breakdown'].keys()),
        }
        logic_files = {k: v for k, v in offspring_data.items() if k.endswith('.py')}
        return manifest, logic_files

    def _prepare_context(self):
        gen_dir = os.path.dirname(os.path.abspath(__file__))
        files = {}
        for fname in ["run.py", "meta_architect.py", "env.py", "spawner.py"]:
            path = os.path.join(gen_dir, fname)
            try:
                with open(path, "r") as src:
                    files[fname] = src.read()
            except FileNotFoundError:
                logger.warning(f"_prepare_context: could not read {path}")
        return files

    def _call_llm(self, prompt):
        use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI") == "TRUE"
        if use_vertex:
            project = os.getenv("GOOGLE_CLOUD_PROJECT", "codingagentproject-487605")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            token = os.getenv("GCLOUD_ACCESS_TOKEN")
            url = (
                f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
                f"/locations/{location}/publishers/google/models/{self.model}:generateContent"
            )
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {token}'
            }
            data = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}]
            }
        else:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.model}:generateContent?key={self.api_key}"
            )
            headers = {'Content-Type': 'application/json'}
            data = {"contents": [{"parts": [{"text": prompt}]}]}

        req = urllib.request.Request(
            url, data=json.dumps(data).encode('utf-8'), headers=headers
        )

        for attempt in range(7):
            try:
                with urllib.request.urlopen(req) as res:
                    resp_json = json.loads(res.read().decode('utf-8'))
                    return resp_json['candidates'][0]['content']['parts'][0]['text']
            except Exception as e:
                wait_time = 5 * (2 ** attempt)
                logger.error(f"LLM call failed (attempt {attempt+1}/7): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        return ""

    def _parse_json(self, text):
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                return json.loads(text[start:end + 1])
            return json.loads(text)
        except Exception as e:
            logger.error(f"JSON parse failed: {e}. Text preview: {text[:300]!r}")
            return {}
