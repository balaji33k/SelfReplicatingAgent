import logging
import os
import json
from pathlib import Path
import sys
import importlib.util

def setup_logging(level_str: str, log_file: Path):
    """
    Sets up logging for the application.
    """
    level = getattr(logging, level_str.upper(), logging.INFO)
    
    # Ensure the directory for the log file exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Silence excessive logging from some libraries if needed
    logging.getLogger("httpx").setLevel(logging.WARNING) 
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_project_root() -> Path:
    """
    Returns the absolute path to the project's root directory.
    Assumes `main.py` is at `agent_root/gen_1/main.py` and project root is `agent_root`.
    """
    # This assumes we are running from within a generation directory, e.g., gen_1/
    # The project root is one level up from the current generation's directory.
    return Path(__file__).resolve().parent.parent

def get_current_generation_dir(generation_number: int) -> Path:
    """
    Returns the absolute path to the current generation's directory.
    """
    # If main.py is called directly, Path(__file__).resolve().parent will be the current generation's directory
    return Path(__file__).resolve().parent

def get_next_generation_dir(generation_number: int) -> Path:
    """
    Returns the absolute path where the next generation should be spawned.
    """
    project_root = get_project_root()
    return project_root / f"gen_{generation_number}"

def save_json(data, file_path: Path):
    """Saves data to a JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logging.getLogger(__name__).error(f"Failed to write JSON to {file_path}: {e}")
        raise

def load_json(file_path: Path):
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.getLogger(__name__).warning(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logging.getLogger(__name__).error(f"Failed to decode JSON from {file_path}: {e}")
        raise
    except IOError as e:
        logging.getLogger(__name__).error(f"Failed to read JSON from {file_path}: {e}")
        raise

def import_config_from_path(config_path: Path):
    """Dynamically imports the Config class from a given path."""
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    if spec is None:
        raise ImportError(f"Could not find module spec for {config_path}")
    
    config_module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Could not load module for {config_path}")

    sys.modules["config_module"] = config_module
    spec.loader.exec_module(config_module)
    
    if not hasattr(config_module, 'Config'):
        raise AttributeError(f"Module {config_module.__name__} at {config_path} does not contain a 'Config' class.")
    
    return config_module.Config

