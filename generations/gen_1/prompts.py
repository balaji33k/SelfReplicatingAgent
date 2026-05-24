# This file contains the generic prompt templates used by the system.

class PromptTemplates:
    """
    Collection of generic prompt templates.
    These prompts instruct the LLM on HOW to reason and structure a solution,
    not WHAT the solution should look like in terms of specific APIs or domains.
    """

    @staticmethod
    def code_generation_prompt(task_description: str, test_cases_description: str, function_signature: str = None) -> str:
        """
        Generic prompt for generating Python code to solve a problem.
        The system derives all task-specific context from the task data itself at runtime.
        """
        signature_hint = f"\nYour function must match this signature: {function_signature}" if function_signature else ""
        
        return (
            "You are an expert Python programmer. Your goal is to write a Python function that solves the given problem.\n"
            "The problem description is provided below. You will also be given test cases, which serve as examples of inputs and their corresponding expected outputs.\n"
            "Your solution should be a single, self-contained Python function. Do not include example usage, print statements outside the function, or any code that is not part of the function definition itself. Do not import external libraries unless absolutely necessary and permitted by the problem context (which will be implied by the test cases if any advanced features are needed).\n"
            "Think step by step to arrive at the correct solution. First, understand the problem thoroughly. Then, consider the input constraints, edge cases, and potential pitfalls. Finally, devise an algorithm and implement it.\n"
            "Ensure your code is clean, efficient, and directly addresses the problem as described.\n\n"
            f"Problem Description:\n{task_description}\n\n"
            f"Input/Output Examples (Test Cases):\n{test_cases_description}\n"
            f"{signature_hint}\n\n"
            "Begin your Python function definition now. The function should be named as suggested by the task or context, but focus on the functionality."
        )

    @staticmethod
    def evolutionary_design_prompt(analysis_report_summary: str, current_config_json: str, improvement_log_parent: str) -> str:
        """
        Generic prompt for the EvolutionEngine to design the next generation.
        It receives a summary of the parent generation's performance and
        is instructed to propose changes based on generic failure patterns.
        """
        return (
            "You are an AI architect tasked with designing the next generation of a self-replicating AI coding agent.\n"
            "The current generation has completed its benchmark run. Here is a summary of its performance and the root causes of its failures:\n"
            f"--- Current Generation Analysis Report ---\n{analysis_report_summary}\n\n"
            f"--- Parent Generation Improvement Log (context for current generation's design) ---\n{improvement_log_parent}\n\n"
            "Based on this analysis, your goal is to propose improvements for the *next* generation.\n"
            "You must output a JSON object containing two keys: 'improvement_log_message' and 'new_files_content'.\n"
            "1. 'improvement_log_message': A string describing what was learned from the current generation's experience, how the solving strategy will change as a result, and what measurable improvement is expected in the next benchmark run. This message must be generic, focusing on reasoning strategies and robustness, not specific task solutions.\n"
            "2. 'new_files_content': A dictionary where keys are file paths (e.g., 'config.py', 'prompts.py') and values are the complete, updated Python source code for those files. You must only provide files that you intend to change. If a file is not listed, it will be carried over unchanged.\n"
            "   - If you modify `config.py`, ensure the output is a complete, valid Python file. You can modify any attributes within the `Config`, `LLMConfig`, or `SandboxConfig` classes.\n"
            "   - If you modify `prompts.py`, provide the full content of the updated `PromptTemplates` class.\n"
            "   - You may also propose changes to other existing files or create new ones, provided they adhere to the modular architecture.\n"
            "   - Each component must be redesigned based on failure evidence. If a component is not changed, you *must* justify why no improvement was necessary based on the analysis in the 'improvement_log_message'. Do not silently reproduce the same logic.\n\n"
            "When generating changes, focus on patterns. For example:\n"
            "- If many `SyntaxError`s: Improve prompting for code structure, syntax adherence.\n"
            "- If many `LogicError`s: Enhance reasoning instructions, emphasis on edge cases, test case interpretation.\n"
            "- If `TimeoutError`s or `MemoryError`s: Suggest sandbox tuning (e.g., higher limits if reasonable, or prompt for more efficient algorithms).\n"
            "- If `EnvironmentError`s: Propose changes to `sandbox_execution.py` for better isolation or dependency management.\n\n"
            "Remember, your output must be a valid JSON object. Do not include any narrative outside the JSON.\n"
            "Example of expected output structure:\n"
            "```json\n"
            "{\n"
            "  \"improvement_log_message\": \"What was learned... How strategy changed... Expected improvement...\",\n"
            "  \"new_files_content\": {\n"
            "    \"config.py\": \"# Updated config.py content\\n...\",\n"
            "    \"prompts.py\": \"# Updated prompts.py content\\n...\"\n"
            "  }\n"
            "}\n"
            "```\n"
        )