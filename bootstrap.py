"""
bootstrap.py — Seeds Generation 1 of the self-replicating agent system.

Reads seed_prompt.txt, calls the LLM, validates the response,
writes all generated source files to generations/gen_1/, then prints
the command to start the lineage.

Run once:
    python bootstrap.py
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

logging.basicConfig(level=logging.INFO, format="[Bootstrap] %(levelname)s %(message)s")
logger = logging.getLogger("Bootstrap")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GEN_1_DIR = os.path.join(PROJECT_ROOT, "generations", "gen_1")

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
MAX_BOOTSTRAP_ATTEMPTS = 2
MAX_LLM_RETRIES = 7


def load_seed_prompt() -> str:
    path = os.path.join(PROJECT_ROOT, "seed_prompt.txt")
    with open(path, "r") as f:
        content = f.read()
    logger.info(f"Seed prompt loaded ({len(content)} chars)")
    return content


def call_llm(prompt: str) -> str:
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI") == "TRUE"

    if use_vertex:
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "codingagentproject-487605")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        token = os.getenv("GCLOUD_ACCESS_TOKEN")
        url = (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
            f"/locations/{location}/publishers/google/models/{LLM_MODEL}:generateContent"
        )
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
        data = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    else:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "No API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{LLM_MODEL}:generateContent?key={api_key}"
        )
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": prompt}]}]}

    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers)

    for attempt in range(MAX_LLM_RETRIES):
        try:
            with urllib.request.urlopen(req) as res:
                resp = json.loads(res.read().decode())
                return resp["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            wait = 5 * (2 ** attempt)
            logger.error(f"LLM call failed (attempt {attempt+1}/{MAX_LLM_RETRIES}): {e}. Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"LLM call failed after {MAX_LLM_RETRIES} attempts.")


def parse_response(text: str) -> dict:
    """
    Parse the two-section LLM response:
      Section 1 — small JSON with entrypoint / topology / improvement_log (no code)
      Section 2 — Python files delimited by === BEGIN filename.py === / === END filename.py ===

    Falls back to legacy single-JSON format if delimiters are not found.
    """
    result = {}

    # ── Section 2: extract delimited file blocks ──────────────────────────────
    file_pattern = re.compile(
        r"=== BEGIN (.+?\.py) ===\n(.*?)\n=== END \1 ===",
        re.DOTALL,
    )
    for m in file_pattern.finditer(text):
        fname = m.group(1).strip()
        content = m.group(2)
        result[fname] = content
        logger.info(f"  Extracted file block: {fname} ({len(content)} chars)")

    # ── Section 1: extract JSON metadata ──────────────────────────────────────
    # Remove file blocks so braces inside code don't confuse the JSON parser
    meta_text = file_pattern.sub("", text)

    # Strip markdown fences
    if "```" in meta_text:
        lines = [l for l in meta_text.splitlines() if not l.strip().startswith("```")]
        meta_text = "\n".join(lines)

    start = meta_text.find("{")
    if start != -1:
        end = meta_text.rfind("}")
        if end != -1:
            json_str = meta_text[start:end + 1]
            for strict in (True, False):
                try:
                    meta = json.loads(json_str, strict=strict)
                    result.update({k: v for k, v in meta.items() if not k.endswith(".py")})
                    break
                except Exception as e:
                    if strict:
                        logger.warning(f"Strict JSON parse failed ({e}), retrying lenient...")
                    else:
                        logger.error(f"JSON parse failed: {e}. Metadata preview: {json_str[:300]!r}")

    # ── Legacy fallback: single JSON with code embedded ───────────────────────
    if not result.get("topology") and not any(k.endswith(".py") for k in result):
        logger.warning("Delimiter format not found — attempting legacy single-JSON parse")
        raw = text
        if "```" in raw:
            lines = [l for l in raw.splitlines() if not l.strip().startswith("```")]
            raw = "\n".join(lines)
        s = raw.find("{")
        if s != -1:
            e2 = raw.rfind("}")
            if e2 != -1:
                for strict in (True, False):
                    try:
                        result = json.loads(raw[s:e2 + 1], strict=strict)
                        break
                    except Exception:
                        pass

    return result


def detect_entrypoint(py_files: dict) -> str:
    for fname, content in py_files.items():
        if 'if __name__' in content and '__main__' in content:
            return fname
    return ""


def validate(data: dict) -> list:
    errors = []
    py_files = {k: v for k, v in data.items() if k.endswith(".py")}

    if not py_files:
        errors.append("No .py source files in LLM response")

    if not data.get("topology"):
        errors.append("'topology' is missing or empty — LLM must define pipeline nodes")

    if not data.get("improvement_log"):
        errors.append("'improvement_log' is missing")

    entrypoint = data.get("entrypoint") or detect_entrypoint(py_files)
    if not entrypoint:
        errors.append(
            "'entrypoint' not specified and could not be auto-detected. "
            "LLM must specify which file is the main entry point."
        )
    else:
        data["entrypoint"] = entrypoint
        if entrypoint not in py_files:
            errors.append(
                f"Entrypoint '{entrypoint}' is specified but not present in generated files. "
                f"Available files: {list(py_files.keys())}"
            )

    return errors


def compile_check(py_files: dict, target_dir: str) -> list:
    errors = []
    for fname in py_files:
        path = os.path.join(target_dir, fname)
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", path],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            errors.append(f"{fname}: {result.stderr.strip()}")
    return errors


def write_gen1(data: dict) -> tuple:
    py_files = {k: v for k, v in data.items() if k.endswith(".py")}

    os.makedirs(GEN_1_DIR, exist_ok=True)

    for fname, content in py_files.items():
        path = os.path.join(GEN_1_DIR, fname)
        with open(path, "w") as f:
            f.write(content)
        logger.info(f"  Written: {fname}")

    compile_errors = compile_check(py_files, GEN_1_DIR)
    if compile_errors:
        return False, compile_errors

    # Copy infrastructure files (not logic — LLM does not generate these)
    for infra_file in ("telemetry.py", "lineage_memory.py"):
        src = os.path.join(PROJECT_ROOT, infra_file)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(GEN_1_DIR, infra_file))
            logger.info(f"  Copied infrastructure: {infra_file}")

    # Copy enriched problem pool
    pool_src = os.path.join(PROJECT_ROOT, "data", "fixed_problem_pool.json")
    pool_dst = os.path.join(GEN_1_DIR, "problem_pool.json")
    if os.path.exists(pool_src):
        shutil.copy(pool_src, pool_dst)
        logger.info("  Copied: problem_pool.json (enriched)")
    else:
        logger.warning("  data/fixed_problem_pool.json not found — problem pool will be missing")

    # Write manifest
    manifest = {
        "GENERATION": 1,
        "TOPOLOGY": data.get("topology", []),
        "IMPROVEMENT_LOG": data.get("improvement_log", ""),
        "ENTRYPOINT": data.get("entrypoint", ""),
        "PARENT_PASS_RATE": "N/A",
        "PARENT_ERRORS": [],
        "SEEDED_FROM": "seed_prompt.txt",
    }
    with open(os.path.join(GEN_1_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)
    logger.info("  Written: manifest.json")

    return True, []


def main():
    logger.info("=" * 50)
    logger.info("  Self-Replicating Agent — Bootstrap")
    logger.info("=" * 50)

    if os.path.exists(GEN_1_DIR):
        logger.warning(f"generations/gen_1/ already exists. Remove it first to re-bootstrap.")
        sys.exit(1)

    seed_prompt = load_seed_prompt()

    for attempt in range(1, MAX_BOOTSTRAP_ATTEMPTS + 1):
        logger.info(f"Calling LLM (bootstrap attempt {attempt}/{MAX_BOOTSTRAP_ATTEMPTS})...")

        response_text = call_llm(seed_prompt)
        data = parse_response(response_text)

        if not data:
            logger.warning("LLM returned empty or unparseable JSON.")
            if attempt == MAX_BOOTSTRAP_ATTEMPTS:
                logger.error("Bootstrap failed: could not get valid JSON from LLM.")
                sys.exit(1)
            continue

        errors = validate(data)
        if errors:
            logger.warning(f"Validation failed ({len(errors)} issue(s)):")
            for e in errors:
                logger.warning(f"  - {e}")
            if attempt == MAX_BOOTSTRAP_ATTEMPTS:
                logger.error("Bootstrap failed: LLM response did not meet validation requirements.")
                sys.exit(1)
            continue

        logger.info(f"Writing Gen 1 to {GEN_1_DIR} ...")
        success, compile_errors = write_gen1(data)

        if not compile_errors:
            entrypoint = data["entrypoint"]
            topology_labels = [n.get("label", n.get("id")) for n in data.get("topology", [])]

            logger.info("")
            logger.info("Bootstrap complete.")
            logger.info(f"  Entrypoint : {entrypoint}")
            logger.info(f"  Topology   : {' -> '.join(topology_labels)}")
            logger.info(f"  Files      : {[k for k in data if k.endswith('.py')]}")
            logger.info("")
            logger.info("To start the lineage:")
            logger.info(f"  python {os.path.join(GEN_1_DIR, entrypoint)}")
            return

        logger.warning(f"Compile errors in generated code ({len(compile_errors)} file(s)):")
        for err in compile_errors:
            logger.warning(f"  - {err}")

        shutil.rmtree(GEN_1_DIR, ignore_errors=True)

        if attempt == MAX_BOOTSTRAP_ATTEMPTS:
            logger.error("Bootstrap failed: LLM generated code that does not compile.")
            sys.exit(1)

    logger.error("Bootstrap failed after all attempts.")
    sys.exit(1)


if __name__ == "__main__":
    main()
