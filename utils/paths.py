import yaml
from pathlib import Path

def load_paths(config_file=Path(__file__).resolve().parents[1]/"config/paths.yaml"):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    root = Path(config["project_root"])

    # remove project_root from nested path dict
    config.pop("project_root")

    def join_paths(d):
        return {k: (root / v if isinstance(v, str) else join_paths(v)) for k, v in d.items()}

    return join_paths(config)
