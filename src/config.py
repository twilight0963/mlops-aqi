"""Load params.yaml and resolve paths relative to the repo root."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARAMS = REPO_ROOT / "params.yaml"


def load_params(path=DEFAULT_PARAMS):
    """Return params.yaml as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def repo_path(relative):
    """Absolute path for a repo-relative path from params.yaml."""
    p = Path(relative)
    return p if p.is_absolute() else REPO_ROOT / p
