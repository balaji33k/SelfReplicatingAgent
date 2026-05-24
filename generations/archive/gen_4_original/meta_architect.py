import random

class MetaArchitect:
    def __init__(self):
        self.prompt = ""
        self.improvement_log = []

    def load_initial_prompt(self, filename):
        with open(filename, 'r') as f:
            self.prompt = f.read()

    def suggest_change(self, topology):
        # Simple suggestion: add a new component
        possible_components = ['firewall', 'load_balancer', 'database', 'web_server']
        component = random.choice(possible_components)
        action = random.choice(['add', 'remove'])

        if action == 'add':
            suggestion = f'Add a {component} to the topology.'
        else:
            if len(topology) > 0:
                remove_target = random.choice(topology)
                suggestion = f'Remove {remove_target["type"]} component with name {remove_target["name"]}.'
            else:
                suggestion = f'Add a {component} to the topology.' # Add if empty


        self.improvement_log.append(f'Suggested: {suggestion}')
        return suggestion

    def update(self, suggestion, evaluation_result):
        # Simple update: log the suggestion and result
        self.improvement_log.append(f'Suggestion: {suggestion}, Result: {evaluation_result}')

    def get_improvement_log(self):
        return '\n'.join(self.improvement_log)