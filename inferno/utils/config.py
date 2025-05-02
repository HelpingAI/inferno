"""
Configuration management for Inferno
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

# Default configuration
default_config: Dict[str, Any] = {
    "models_dir": "~/.inferno/models",
    "api_host": "127.0.0.1",
    "api_port": 8000,
    "default_context_length": 4096,
    "default_gpu_layers": -1,  # -1 means use all available GPU layers
}

class Config:
    """
    Configuration manager for Inferno.
    Handles loading, saving, and accessing configuration values.
    """
    config_dir: Path
    config_file: Path
    models_dir: Path
    config: Dict[str, Any]

    def __init__(self) -> None:
        self.config_dir = Path(os.path.expanduser("~/.inferno"))
        self.config_file = self.config_dir / "config.json"
        self.models_dir = Path(os.path.expanduser(default_config["models_dir"]))
        self._ensure_dirs()
        self._load_config()

    def _ensure_dirs(self) -> None:
        """Ensure configuration and models directories exist."""
        self.config_dir.mkdir(exist_ok=True, parents=True)
        self.models_dir.mkdir(exist_ok=True, parents=True)

    def _load_config(self) -> None:
        """Load configuration from file or create default."""
        if not self.config_file.exists():
            self._save_config(default_config)
            self.config = default_config.copy()
        else:
            with open(self.config_file, "r") as f:
                self.config = json.load(f)

    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file."""
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key."""
        self.config[key] = value
        self._save_config(self.config)

    def get_model_path(self, model_name: str) -> Path:
        """Get the path to a model directory by model name."""
        return self.models_dir / model_name

    def list_models(self) -> List[str]:
        """List all downloaded model names."""
        if not self.models_dir.exists():
            return []
        return [d.name for d in self.models_dir.iterdir() if d.is_dir() and ":" not in d.name]

# Global configuration instance
config: Config = Config()
