import logging
from task_manager import Task
from llm_client import LLMClient
from config import LLMConfig
from prompts import PromptTemplates

logger = logging.getLogger(__name__)

class CodeGenerator:
    """
    Generates Python code solutions for tasks using an LLM.
    """
    def __init__(self, llm_config: LLMConfig):
        self.llm_client = LLMClient(llm_config)

    def _format_test_cases(self, test_cases: list) -> str:
        """Helper to format test cases for the prompt."""
        if not test_cases:
            return "No specific test cases provided. Consider common inputs and edge cases."
        
        formatted_cases = []
        for i, case in enumerate(test_cases):
            input_str = repr(case.get("input", "N/A"))
            expected_str = repr(case.get("expected", "N/A"))
            formatted_cases.append(f"Test Case {i+1}:\nInput: {input_str}\nExpected Output: {expected_str}")
        return "\n\n".join(formatted_cases)

    def generate_solution(self, task: Task) -> str:
        """
        Generates Python code for a given task using the configured LLM.
        """
        test_cases_description = self._format_test_cases(task.test_cases)
        
        system_message = "You are a professional Python software engineer tasked with writing robust and correct code."
        user_message = PromptTemplates.code_generation_prompt(
            task_description=task.description,
            test_cases_description=test_cases_description,
            function_signature=task.signature
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        try:
            raw_response = self.llm_client.chat_completion(messages)
            # Basic parsing: extract code block if present
            if "```python" in raw_response:
                code_block = raw_response.split("```python")[1].split("```")[0].strip()
            else:
                code_block = raw_response.strip() # Assume the whole response is code if no block marker
            
            return code_block
        except Exception as e:
            logger.error(f"Failed to generate code for task {task.task_id}: {e}")
            # Guardrail: No Silent Failure
            # Re-raise or return a distinct error state if generation fails
            raise RuntimeError(f"Code generation failed for task {task.task_id}") from e
