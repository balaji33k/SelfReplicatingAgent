import random

class Spawner:
    def apply_change(self, topology, suggestion):
        # Apply the suggested change to the topology
        # This is a dummy implementation that randomly adds or removes components
        # Replace with actual deployment logic
        if 'add' in suggestion.lower():
            component_type = suggestion.split(' ')[2]  # Extract the component type
            new_component = {
                'type': component_type,
                'name': f'{component_type}_{len(topology)}',
                'config': {'param1': random.randint(1, 10), 'param2': random.random()}
            }
            topology = topology + [new_component]
        elif 'remove' in suggestion.lower():
            if len(topology) > 0:
                target_name = suggestion.split('name ')[1][:-1] # extract the target_name to remove
                topology = [comp for comp in topology if comp['name'] != target_name]

        return topology