# utils/common.py

import hashlib

def generate_param_hash(params: dict) -> str:
    """
    Create a short hash from a dictionary of hyperparameters to uniquely identify a config.
    """
    sorted_items = sorted(params.items())  # Sort for consistent ordering
    param_str = "_".join(f"{k}={v}" for k, v in sorted_items)
    return hashlib.md5(param_str.encode()).hexdigest()[:8]