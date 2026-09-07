import os
from pathlib import Path
from ruamel.yaml import YAML



# Find the absolute path to config.yaml relative to this settings file
CONFIG_FILE_PATH = Path(__file__).parent / "config.yaml"
yaml = YAML(typ='safe')
yaml.preserve_quotes = True  # Keeps your string formatting clean
def load_config() -> dict:
    """Reads the YAML file and returns it as a dictionary."""
    if not CONFIG_FILE_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_FILE_PATH}")
    with open(CONFIG_FILE_PATH, 'r') as f:
        return yaml.load(f)
def update_config_value(category: str, key: str, value):
    """Updates a single value in the YAML file and permanently saves it."""
    config_data = load_config()
    
    # Update the value in memory
    config_data[category][key] = value
    
    # Save it back to the file system persistently
    with open(CONFIG_FILE_PATH, 'w') as f:
        yaml.dump(config_data, f)
