import json
from pathlib import Path

class ModelRegistry:

    def __init__(self):
        self.path = Path("registry/model_registry.json")

    def get_active_model_path(self):

        with open(self.path, "r") as f:
            data = json.load(f)

        return Path("artifacts") / data["production_model"]

    def promote(self, new_model_name):

        with open(self.path, "r") as f:
            data = json.load(f)

        previous = data["production_model"]

        data["previous_model"] = previous
        data["production_model"] = new_model_name

        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def rollback(self):

        with open(self.path, "r") as f:
            data = json.load(f)

        data["production_model"] = data["previous_model"]

        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)