from typing import Any, Dict, Optional

import yaml


class Config:
    """
    A class to handle loading and accessing configuration parameters from a YAML file.

    Attributes:
        config_path (str): Path to the configuration file.
        config (dict): Dictionary containing the configuration parameters.
    """

    def __init__(self, config_path: str = "challenge/config.yaml") -> None:
        """
        Initialize the Config class.

        Args:
            config_path (str): Path to the configuration file. Defaults to
                "challenge/config.yaml".
        """
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """
        Load the configuration from the YAML file.

        Returns:
            Dict[str, Any]: Dictionary containing the configuration parameters.
        """
        with open(self.config_path, "r") as file:
            return yaml.safe_load(file)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get a configuration value by key.

        Args:
            key (str): The key of the configuration parameter.
            default (Optional[Any]): The default value to return if the key is not
                found. Defaults to None.

        Returns:
            Any: The value of the configuration parameter, or the default value if the
                key is not found.
        """
        return self.config.get(key, default)
