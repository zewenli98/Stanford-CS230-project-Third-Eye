"""
Configuration loader and validator.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration class for loading and accessing training configs."""

    def __init__(self, config_path: str = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent / "training_config.yaml"

        self.config_path = Path(config_path)
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.

        Args:
            key: Configuration key (e.g., 'model.name')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set configuration value by dot-notation key.

        Args:
            key: Configuration key (e.g., 'model.name')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, output_path: str = None):
        """
        Save configuration to YAML file.

        Args:
            output_path: Output path (uses original path if None)
        """
        if output_path is None:
            output_path = self.config_path

        try:
            with open(output_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise

    def validate(self) -> bool:
        """
        Validate configuration values.

        Returns:
            True if valid, raises exception otherwise
        """
        # Check required fields
        required_fields = [
            'model.name',
            'data.yaml_path',
            'training.epochs',
            'training.batch_size',
        ]

        for field in required_fields:
            if self.get(field) is None:
                raise ValueError(f"Required configuration field missing: {field}")

        # Validate ranges
        if self.get('training.epochs') <= 0:
            raise ValueError("training.epochs must be positive")

        if self.get('training.batch_size') <= 0:
            raise ValueError("training.batch_size must be positive")

        if not (0 <= self.get('validation.conf_thres') <= 1):
            raise ValueError("validation.conf_thres must be between 0 and 1")

        if not (0 <= self.get('validation.iou_thres') <= 1):
            raise ValueError("validation.iou_thres must be between 0 and 1")

        logger.info("Configuration validation passed")
        return True

    def __repr__(self) -> str:
        """String representation."""
        return f"Config(path={self.config_path})"

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style assignment."""
        self.set(key, value)


def load_config(config_path: str = None) -> Config:
    """
    Convenience function to load configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Config object
    """
    return Config(config_path)


if __name__ == "__main__":
    # Test configuration loading
    config = Config()
    print(f"Model name: {config.get('model.name')}")
    print(f"Epochs: {config.get('training.epochs')}")
    print(f"Batch size: {config.get('training.batch_size')}")

    # Validate
    config.validate()
    print("Configuration is valid!")
