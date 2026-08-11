import json
import logging
from pathlib import Path

# Setup logging for this module
logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Simple JSON-based model registry to track production and previous models.
    """

    def __init__(self, registry_path: str = "registry/model_registry.json"):
        """
        Args:
            registry_path: Path to the registry JSON file (relative to project root).
        """
        self.registry_path = Path(registry_path)
        self.artifacts_dir = Path("models/artifacts")  # Where models are stored
        self._ensure_registry_exists()

    def _ensure_registry_exists(self):
        """Create a default registry file if it does not exist."""
        if not self.registry_path.exists():
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            default_registry = {
                "production_model": None,
                "previous_model": None
            }
            with open(self.registry_path, "w") as f:
                json.dump(default_registry, f, indent=2)
            logger.info(f"Created default model registry at {self.registry_path}")

    def _load_registry(self) -> dict:
        """Load and return the registry dictionary."""
        with open(self.registry_path, "r") as f:
            return json.load(f)

    def _save_registry(self, data: dict):
        """Save the registry dictionary to disk."""
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_active_model_path(self) -> Path:
        """
        Returns the full path to the currently active production model.
        If no production model is set, returns None.
        """
        data = self._load_registry()
        model_name = data.get("production_model")
        if model_name is None:
            logger.warning("No production model set in registry.")
            return None
        return self.artifacts_dir / model_name

    def promote(self, new_model_name: str):
        """
        Promote a new model to production.
        The current production model becomes the previous model.
        """
        data = self._load_registry()
        current_prod = data.get("production_model")
        data["previous_model"] = current_prod
        data["production_model"] = new_model_name
        self._save_registry(data)
        logger.info(f"Promoted model '{new_model_name}' to production. Previous: {current_prod}")

    def rollback(self):
        """
        Rollback to the previous production model.
        Does nothing if there is no previous model.
        """
        data = self._load_registry()
        previous = data.get("previous_model")
        if previous is None:
            logger.warning("No previous model available for rollback.")
            return
        data["production_model"] = previous
        # Optionally keep the previous pointer unchanged for another rollback
        self._save_registry(data)
        logger.info(f"Rolled back to previous model: {previous}")

    def get_previous_model_path(self) -> Path:
        """Returns the full path to the previous model, or None if not set."""
        data = self._load_registry()
        model_name = data.get("previous_model")
        if model_name is None:
            return None
        return self.artifacts_dir / model_name

    def get_all_models(self) -> list:
        """
        Returns a list of all model files found in the artifacts directory.
        Useful for debugging or manual inspection.
        """
        if not self.artifacts_dir.exists():
            return []
        return [str(p.name) for p in self.artifacts_dir.glob("*.pkl")]    