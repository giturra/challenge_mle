import yaml


class Config:
    def __init__(self, config_path: str = "challenge/config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> dict:
        with open(self.config_path, "r") as file:
            return yaml.safe_load(file)

    def get(self, key: str, default=None):
        return self.config.get(key, default)
