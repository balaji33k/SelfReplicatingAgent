"""
task_manager.py — Loads benchmark tasks from problem_pool.json.

Supports the enriched pool format:
  {"training": {"swe": [...], "lcb": [...]}, "test": {...}}

Each task is normalised into a Task object with:
  task_id, description, test_cases, signature, metadata
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class Task:
    """Represents a single coding benchmark task."""

    def __init__(
        self,
        task_id: str,
        description: str,
        test_cases: List[Dict],
        signature: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        self.task_id = task_id
        self.description = description
        self.test_cases = test_cases
        self.signature = signature
        self.metadata = metadata if metadata is not None else {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        task_id = data.get("id") or data.get("task_id", "unknown")
        description = (
            data.get("description")
            or data.get("problem_statement")
            or data.get("prompt")
            or ""
        )
        raw_cases = data.get("test_cases") or data.get("tests") or []
        # Normalise test cases: ensure list of {"input": ..., "expected": ...}
        # Handles both:
        #   {"input": {"nums": [1,2], "target": 3}, "expected": [0,1]}  ← CoderBreed format
        #   {"input": [1, 2], "expected": 3}                            ← flat list format
        normalised = []
        if isinstance(raw_cases, str):
            # String description of tests — skip, pipeline will use problem statement only
            logger.debug(f"Task {task_id}: test_cases is a string description, treating as empty")
        else:
            for case in raw_cases:
                if isinstance(case, dict) and ("input" in case or "expected" in case):
                    normalised.append(case)
                elif isinstance(case, dict):
                    normalised.append(case)
                else:
                    logger.warning(f"Task {task_id}: unexpected test case format, skipping: {type(case)}")
        # CoderBreed uses "function_signature"; fallback to "signature"
        signature = (
            data.get("function_signature")
            or data.get("signature")
        )
        return cls(
            task_id=task_id,
            description=description,
            test_cases=normalised,
            signature=signature,
            metadata={k: v for k, v in data.items() if k not in
                      ("id", "task_id", "description", "problem_statement",
                       "prompt", "test_cases", "tests", "signature",
                       "function_signature")},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.task_id,
            "description": self.description,
            "test_cases": self.test_cases,
            "signature": self.signature,
            "metadata": self.metadata,
        }


class TaskLoader:
    """
    Loads benchmark tasks from problem_pool.json.

    Expected pool format:
      {
        "training": {
          "swe": [ {task}, ... ],
          "lcb": [ {task}, ... ]
        },
        "test": { ... }
      }

    Falls back to a flat list if the file is a plain JSON array.
    """

    def __init__(self, pool_path: Path):
        # Accept either a path to problem_pool.json directly or a directory
        if pool_path.is_dir():
            pool_path = pool_path / "problem_pool.json"
        if not pool_path.exists():
            raise FileNotFoundError(f"Problem pool not found: {pool_path}")
        self.pool_path = pool_path

    def load_tasks(self) -> List[Task]:
        """Load and return all training tasks from the problem pool."""
        try:
            with open(self.pool_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {self.pool_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to read {self.pool_path}: {e}")
            return []

        raw_tasks: List[Dict] = []

        if isinstance(data, list):
            # Plain list format
            raw_tasks = data
        elif isinstance(data, dict):
            # Nested format: {"training": {"swe": [...], "lcb": [...]}}
            training = data.get("training", data)
            if isinstance(training, dict):
                for category, items in training.items():
                    if isinstance(items, list):
                        raw_tasks.extend(items)
                    else:
                        logger.warning(f"Unexpected format in training[{category!r}]: {type(items)}")
            elif isinstance(training, list):
                raw_tasks = training
        else:
            logger.error(f"Unrecognised problem pool format in {self.pool_path}")
            return []

        tasks = []
        for raw in raw_tasks:
            try:
                task = Task.from_dict(raw)
                if not task.test_cases:
                    logger.warning(f"Task {task.task_id} has no test cases — will likely fail execution.")
                tasks.append(task)
            except Exception as e:
                logger.error(f"Failed to load task from pool entry: {e}. Entry: {str(raw)[:100]}")

        # Guardrail: No Benchmark Tampering — load all tasks, skip none
        logger.info(f"Loaded {len(tasks)} tasks from {self.pool_path}")
        return tasks
