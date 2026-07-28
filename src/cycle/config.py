"""YAML configuration loading with dot-path CLI overrides."""

import copy

import yaml


class Config(dict):
    """Dict with attribute access, recursively applied."""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return value

    @classmethod
    def wrap(cls, obj):
        if isinstance(obj, dict):
            return cls({k: cls.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [cls.wrap(v) for v in obj]
        return obj


def load_config(path, overrides=None):
    """Load a YAML config file and apply ``key.subkey=value`` overrides."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"Override must look like key.subkey=value, got: {item}")
        value = yaml.safe_load(raw)
        if isinstance(value, str):
            # YAML 1.1 parses dot-less scientific notation like "2e-5" as a
            # string; coerce numeric-looking overrides
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return Config.wrap(cfg)
