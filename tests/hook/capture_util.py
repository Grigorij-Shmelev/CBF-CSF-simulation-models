"""Helpers shared by the capture hook (sitecustomize.py) and tests/direct.py."""
import hashlib

import numpy as np


def digest(*parts):
    """SHA-256 of arrays: dtype, shape and bytes, so equal digests mean bit-identical arrays."""
    h = hashlib.sha256()
    for a in parts:
        a = np.ascontiguousarray(a)
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def flatten(obj, prefix, arrays, texts):
    """Split a nested result into numeric arrays and JSON texts, keyed by path."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, f"{prefix}/{k}", arrays, texts)
    elif isinstance(obj, (list, tuple)) and not (obj and all(isinstance(v, (int, float, np.number)) and not
                                                             isinstance(v, bool) for v in obj)):
        for i, v in enumerate(obj):
            flatten(v, f"{prefix}/{i}", arrays, texts)
    elif isinstance(obj, (np.ndarray, np.generic, int, float, bool, list, tuple)):
        a = np.asarray(obj)
        if a.dtype.kind in "biufc":
            arrays[prefix] = a
        else:
            texts[prefix] = repr(obj)
    elif obj is None or isinstance(obj, str):
        texts[prefix] = obj
    else:
        texts[prefix] = f"<{type(obj).__name__}>"
