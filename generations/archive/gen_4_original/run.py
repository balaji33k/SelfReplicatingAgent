import os
import subprocess
import json

from meta_architect import MetaArchitect
from env import Environment
from spawner import Spawner

# Load topology from JSON if it exists, otherwise start with an empty topology
topology_file = 'current_topology.json'
if os.path.exists(topology_file):
    with open(topology_file, 'r') as f:
        topology = json.load(f)
else:
    topology = []

# Initialize environment, meta-architect, and spawner
env = Environment()
env.setup()

meta_architect = MetaArchitect()
meta_architect.load_initial_prompt('initial_prompt.txt')

spawner = Spawner()

# Main loop: evolve the topology
for i in range(5): # Run for 5 iterations
    print(f'\nIteration: {i+1}')
    # 1. Suggest a change to the topology
    suggestion = meta_architect.suggest_change(topology)
    print(f'Suggestion: {suggestion}')

    # 2. Apply the suggested change using the spawner
    new_topology = spawner.apply_change(topology, suggestion)
    print(f'New Topology: {new_topology}')

    # 3. Evaluate the new topology in the environment
    evaluation_result = env.evaluate(new_topology)
    print(f'Evaluation Result: {evaluation_result}')

    # 4. Update the meta-architect with the evaluation result
    meta_architect.update(suggestion, evaluation_result)

    # 5. Update the current topology
    if evaluation_result > env.best_result:
        topology = new_topology
        env.best_result = evaluation_result
        print("Topology improved!")
        env.save_best_topology(topology)
    else:
        print("Topology not improved.")

    # 6. Save the current topology to a file
    with open(topology_file, 'w') as f:
        json.dump(topology, f, indent=4)

print("\nEvolution finished.")
print(f"Best result: {env.best_result}")