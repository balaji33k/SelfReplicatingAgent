import os
import datetime


class AgentTopologyConfig:
    """
    Data-driven topology for the multi-agent pipeline.
    The LLM can evolve this config to change which agents are active,
    how many revision/debug cycles are allowed, and execution limits.

    This is the ONLY place pipeline topology is defined.
    The pipeline.py reads this; no if/else chains elsewhere.
    """

    def __init__(self):
        # Gen 1: minimal pipeline — Analyst → Coder → Executor only.
        # Rationale: establish a working baseline first. Each optional agent adds
        # LLM calls and failure points. Future generations enable agents based on
        # what the failure evidence shows:
        #   AssertionError dominates → enable critic + reviser (Gen 2)
        #   RuntimeError dominates   → enable debugger (Gen 2-3)
        #   Test coverage is the gap → enable test_writer (Gen 3+)
        #   Coder logic is bottleneck → enable architect (Gen 3+)
        self.enable_critic: bool = False       # Gen 2+ if assertion errors dominate
        self.enable_reviser: bool = False      # requires critic — keep in sync
        self.enable_test_writer: bool = False  # Gen 2+ if test coverage is the gap
        self.enable_debugger: bool = False     # Gen 2+ if runtime errors dominate

        # Cycle limits — prevents infinite loops when agents are enabled
        self.max_revise_cycles: int = 2        # max times Critic→Reviser runs
        self.max_debug_cycles: int = 1         # max times Debugger runs per task

        # Execution
        self.execution_timeout: int = 10       # seconds per test run

    def to_dict(self) -> dict:
        return {
            "enable_critic": self.enable_critic,
            "enable_reviser": self.enable_reviser,
            "enable_test_writer": self.enable_test_writer,
            "enable_debugger": self.enable_debugger,
            "max_revise_cycles": self.max_revise_cycles,
            "max_debug_cycles": self.max_debug_cycles,
            "execution_timeout": self.execution_timeout,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentTopologyConfig":
        cfg = cls()
        cfg.enable_critic = data.get("enable_critic", True)
        cfg.enable_reviser = data.get("enable_reviser", True)
        cfg.enable_test_writer = data.get("enable_test_writer", True)
        cfg.enable_debugger = data.get("enable_debugger", True)
        cfg.max_revise_cycles = data.get("max_revise_cycles", 2)
        cfg.max_debug_cycles = data.get("max_debug_cycles", 1)
        cfg.execution_timeout = data.get("execution_timeout", 10)
        return cfg


class LLMConfig:
    def __init__(self, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct", api_key_env_var: str = "GROQ_API_KEY", temperature: float = 0.7):
        self.model_name = model_name
        # api_key is kept for compatibility but LLMClient reads directly from env vars
        self.api_key = os.getenv(api_key_env_var) or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.temperature = temperature
        self.timeout_seconds = 300 # 5 minutes

class SandboxConfig:
    def __init__(self, timeout_seconds: int = 60, memory_limit_mb: int = 512):
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        # Future: path to isolated environment setup script (e.g., Dockerfile, virtualenv)

class Config:
    def __init__(self):
        self.generation_number: int = 1 # This will be updated by the Spawner for next generations
        self.timestamp: str = datetime.datetime.now().isoformat()
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

        self.llm_config = LLMConfig()
        self.sandbox_config = SandboxConfig()
        self.topology = AgentTopologyConfig()   # Multi-agent pipeline topology

        # Lineage metadata (set by the Spawner for subsequent generations)
        self.parent_pass_rate: float = None
        self.parent_error_types: dict = {}
        self.improvement_log: str = "Initial generation establishing the core framework."

    def to_dict(self):
        return {
            "generation_number": self.generation_number,
            "timestamp": self.timestamp,
            "log_level": self.log_level,
            "llm_config": {
                "model_name": self.llm_config.model_name,
                "temperature": self.llm_config.temperature,
                "timeout_seconds": self.llm_config.timeout_seconds
            },
            "sandbox_config": {
                "timeout_seconds": self.sandbox_config.timeout_seconds,
                "memory_limit_mb": self.sandbox_config.memory_limit_mb
            },
            "topology": self.topology.to_dict(),
            "parent_pass_rate": self.parent_pass_rate,
            "parent_error_types": self.parent_error_types,
            "improvement_log": self.improvement_log
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        config = cls()
        config.generation_number = data.get("generation_number", 1)
        config.timestamp = data.get("timestamp", datetime.datetime.now().isoformat())
        config.log_level = data.get("log_level", "INFO")

        llm_data = data.get("llm_config", {})
        config.llm_config = LLMConfig(
            model_name=llm_data.get("model_name", "meta-llama/llama-4-scout-17b-16e-instruct"),
            temperature=llm_data.get("temperature", 0.7)
            # api_key is always loaded from env var
        )
        config.llm_config.timeout_seconds = llm_data.get("timeout_seconds", 300)

        sandbox_data = data.get("sandbox_config", {})
        config.sandbox_config = SandboxConfig(
            timeout_seconds=sandbox_data.get("timeout_seconds", 60),
            memory_limit_mb=sandbox_data.get("memory_limit_mb", 512)
        )

        config.parent_pass_rate = data.get("parent_pass_rate")
        config.parent_error_types = data.get("parent_error_types", {})
        config.improvement_log = data.get("improvement_log", "Generated config.")
        topology_data = data.get("topology", {})
        config.topology = AgentTopologyConfig.from_dict(topology_data)
        return config
