import random
import json

class Environment:
    def __init__(self):
        self.best_result = float('-inf')

    def setup(self):
        # Setup the environment (e.g., create directories, initialize resources)
        print("Environment setup complete.")

    def evaluate(self, topology):
        # Evaluate the topology (replace with actual evaluation logic)
        # This is a dummy implementation that returns a random number
        # Higher is better
        score = random.uniform(0, 10) # Simulate some performance score
        print(f"Evaluating topology: {topology} -> score = {score}")
        return score

    def save_best_topology(self, topology):
        # Save the best topology to a file
        with open('best_topology.json', 'w') as f:
            json.dump(topology, f, indent=4)

        print("Best topology saved.")