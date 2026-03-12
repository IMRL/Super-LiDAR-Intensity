import os
import yaml
from argparse import Namespace


def load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _flat_namespace(cfg)


def _flat_namespace(d):
    if isinstance(d, dict):
        return Namespace(**{k: _flat_namespace(v) for k, v in d.items()})
    return d


def resolve_data_paths(cfg, mode=None):
    data = cfg.data
    view = cfg.view_type
    if mode is None:
        mode = "depth" if view == "virtual_camera" else "reflectance"
    if data.coursedic and data.finedic:
        return data.coursedic, data.finedic
    if data.data_root:
        root = data.data_root
        if view == "virtual_camera":
            return os.path.join(root, mode), os.path.join(root, mode)
        return data.data_root, data.data_root
    return None, None
