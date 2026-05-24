import logging
import subprocess
import tempfile
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

from task_manager import Task

logger = logging.getLogger(__name__)

class ExecutionResult:
    """Dataclass to hold the result of code execution."""
    def __init__(self, task_id: str, success: bool, stdout: str, stderr: str, error_type: str, runtime: float, details: Dict[str, Any] = None):
        self.task_id = task_id
        self.success = success
        self.stdout = stdout
        self.stderr = stderr
        self.error_type = error_type
        self.runtime = runtime
        self.details = details if details is not None else {}

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error_type": self.error_type,
            "runtime": self.runtime,
            "details": self.details
        }

class SandboxExecutor:
    """
    Executes generated Python code in an isolated environment.
    This G1 implementation uses a basic subprocess with temporary files.
    Future generations could evolve to use Docker, specialized sandboxes, etc.
    """
    def __init__(self, timeout_seconds: int = 60, memory_limit_mb: int = 512):
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb # Not enforced in this G1 basic implementation
        self._temp_dir: Path = None

    def _setup_sandbox_environment(self) -> Path:
        """Creates a temporary directory for the sandbox."""
        self._temp_dir = Path(tempfile.mkdtemp(prefix="agent_sandbox_"))
        logger.debug(f"Sandbox temporary directory created at: {self._temp_dir}")
        return self._temp_dir

    def _cleanup_sandbox_environment(self):
        """Removes the temporary directory."""
        if self._temp_dir and self._temp_dir.exists():
            import shutil
            try:
                shutil.rmtree(self._temp_dir)
                logger.debug(f"Sandbox temporary directory removed: {self._temp_dir}")
            except OSError as e:
                logger.warning(f"Failed to remove sandbox directory {self._temp_dir}: {e}")
        self._temp_dir = None

    def _create_test_runner_script(self, task: Task, code: str, script_path: Path) -> None:
        """
        Creates a Python script that contains the generated code and executes its test cases.
        This script will dynamically execute the generated function with provided inputs
        and capture outputs for comparison.
        """
        # Guardrail: No Privilege Escalation in the Sandbox
        # The sandbox script should not allow arbitrary file access, network, etc.
        # This current implementation is still limited to what subprocess.run allows.
        
        # Function to be used by the test runner to compare results
        TEST_RUNNER_HELPER_CODE = """
import json
import sys

# Custom comparison for float or list of floats with tolerance
def compare_results(expected, actual):
    if isinstance(expected, float) and isinstance(actual, float):
        return abs(expected - actual) < 1e-6
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False
        if all(isinstance(e, float) for e in expected) and all(isinstance(a, float) for a in actual):
            return all(abs(e - a) < 1e-6 for e, a in zip(expected, actual))
    return expected == actual
"""
        
        # Find the function name. This is a heuristic and might need refinement.
        # For G1, we assume the solution defines a single function.
        # If a signature is provided, we can extract the name more robustly.
        func_name = "solve" # Default if not specified or extractable
        if task.signature:
            try:
                func_name = task.signature.split('def ')[1].split('(')[0].strip()
            except IndexError:
                logger.warning(f"Could not parse function name from signature: {task.signature}. Using default 'solve'.")
        else:
            # Attempt to find the first function definition if no signature
            import re
            match = re.search(r"def\s+(\w+)\s*\(", code)
            if match:
                func_name = match.group(1)
            else:
                logger.warning(f"Could not find function name in code for task {task.task_id}. Using default 'solve'.")


        script_content = f"{TEST_RUNNER_HELPER_CODE}\n{self._strip_markdown(code)}\n\n"
        script_content += f"if __name__ == '__main__':\n"
        script_content += f"    results = []\n"
        script_content += f"    test_cases_data = {json.dumps(task.test_cases)}\n"
        script_content += f"    for i, case in enumerate(test_cases_data):\n"
        script_content += f"        input_data = case.get('input')\n"
        script_content += f"        expected_output = case.get('expected')\n"
        script_content += f"        try:\n"
        script_content += f"            # Dynamically call the function with input_data (handle tuple vs single arg)\n"
        script_content += f"            if isinstance(input_data, list) or isinstance(input_data, tuple):\n"
        script_content += f"                actual_output = {func_name}(*input_data)\n"
        script_content += f"            else:\n"
        script_content += f"                actual_output = {func_name}(input_data)\n"
        script_content += f"            \n"
        script_content += f"            passed = compare_results(expected_output, actual_output)\n"
        script_content += f"            results.append({{'test_case': i + 1, 'input': input_data, 'expected': expected_output, 'actual': actual_output, 'passed': passed}})\n"
        script_content += f"        except Exception as e:\n"
        script_content += f"            results.append({{'test_case': i + 1, 'input': input_data, 'expected': expected_output, 'error': str(e), 'passed': False}})\n"
        script_content += f"    print(json.dumps(results))\n"

        try:
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
        except IOError as e:
            logger.error(f"Failed to write test runner script to {script_path}: {e}")
            raise

    @staticmethod
    def _strip_markdown(code: str) -> str:
        """Removes ```python / ``` fences that LLMs often wrap code in."""
        lines = code.splitlines()
        result = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                continue  # drop fence lines entirely
            result.append(line)
        return "\n".join(result)

    def execute_code(self, task: Task, code: str) -> ExecutionResult:
        """
        Executes the generated code within a sandbox and captures its output.
        """
        self._setup_sandbox_environment()
        
        script_file = self._temp_dir / "solution_runner.py"
        try:
            self._create_test_runner_script(task, code, script_file)
        except Exception as e:
            self._cleanup_sandbox_environment()
            return ExecutionResult(
                task_id=task.task_id, success=False, stdout="", stderr=f"Failed to prepare runner script: {e}",
                error_type="EnvironmentError", runtime=0.0
            )

        start_time = time.time()
        process = None
        try:
            # Use `sys.executable` to ensure the correct Python interpreter is used.
            # Guardrail: No Privilege Escalation in the Sandbox (basic enforcement by `subprocess`)
            # No network, filesystem access outside temp dir (not strictly enforced by Python)
            # More robust sandboxing (e.g., Docker, firejail) would be a G2+ improvement.
            process = subprocess.run(
                [sys.executable, str(script_file)],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False, # Don't raise CalledProcessError, we want to inspect stdout/stderr
                env={"PYTHONUNBUFFERED": "1"} # Ensure output is not buffered
            )
            end_time = time.time()
            runtime = end_time - start_time

            stdout = process.stdout
            stderr = process.stderr

            if process.returncode != 0 and not stderr:
                # If non-zero exit but no stderr, it might be due to Python interpreter issues or signal.
                stderr = f"Process exited with non-zero code {process.returncode} but no stderr captured."

            # Parse results from stdout (assuming the test runner prints JSON)
            test_results: List[Dict[str, Any]] = []
            try:
                test_results = json.loads(stdout.strip())
            except json.JSONDecodeError:
                # If stdout is not JSON, it implies a syntax error or a crash before test runner prints results
                logger.debug(f"Stdout not JSON for {task.task_id}. Raw stdout: {stdout.strip()}")
                if stderr:
                    return ExecutionResult(task_id=task.task_id, success=False, stdout=stdout, stderr=stderr,
                                        error_type="RuntimeError" if "Error" in stderr or "Exception" in stderr else "SyntaxError",
                                        runtime=runtime, details={"raw_output": stdout + stderr})
                else:
                    return ExecutionResult(task_id=task.task_id, success=False, stdout=stdout, stderr="Could not parse test results from stdout. Possible syntax error or silent crash.",
                                        error_type="SyntaxError", runtime=runtime, details={"raw_output": stdout})

            all_passed = all(res.get('passed', False) for res in test_results)
            
            error_type = None
            if not all_passed:
                # Check for runtime errors within test cases
                if any('error' in res for res in test_results if not res.get('passed')):
                    error_type = "RuntimeError" # e.g., ZeroDivisionError, IndexError inside the function
                else:
                    error_type = "LogicError" # Test cases failed due to incorrect logic

            return ExecutionResult(
                task_id=task.task_id,
                success=all_passed,
                stdout=stdout,
                stderr=stderr,
                error_type=error_type,
                runtime=runtime,
                details={"test_results": test_results}
            )

        except subprocess.TimeoutExpired:
            process.kill() # Ensure the process is terminated
            process.wait() # Wait for it to clean up
            runtime = time.time() - start_time
            return ExecutionResult(
                task_id=task.task_id, success=False, stdout="", stderr="Execution timed out.",
                error_type="TimeoutError", runtime=runtime
            )
        except FileNotFoundError:
            return ExecutionResult(
                task_id=task.task_id, success=False, stdout="", stderr="Python interpreter not found.",
                error_type="EnvironmentError", runtime=0.0
            )
        except Exception as e:
            # Catch any other unexpected errors during subprocess management
            logger.error(f"Unexpected error during sandbox execution for task {task.task_id}: {e}", exc_info=True)
            return ExecutionResult(
                task_id=task.task_id, success=False, stdout="", stderr=f"System error during execution: {e}",
                error_type="SystemError", runtime=0.0
            )
        finally:
            self._cleanup_sandbox_environment()

    def cleanup(self):
        """Ensures the sandbox environment is cleaned up."""
        self._cleanup_sandbox_environment()
