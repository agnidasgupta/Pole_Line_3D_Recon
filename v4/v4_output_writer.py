"""Single ordered CPU writer with a bounded queue and explicit array ownership."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np


def snapshot_stage1_output(item, pred, metadata):
    """Own the eight artifact arrays before another inference can reuse buffers.

    Only CPU NumPy arrays are accepted. Preserve absent optional fields so the
    production writer supplies exactly its existing defaults.
    """
    def copy_fields(source, names):
        result = {}
        for name in names:
            if name in source:
                if not isinstance(source[name], np.ndarray):
                    raise TypeError(f'{name} must be a CPU NumPy array')
                result[name] = source[name].copy()
        return result
    return (copy_fields(item, ('coords', 'dist_values', 'source_rows', 'raw_labels')),
            copy_fields(pred, ('pole', 'line', 'semantic', 'objectness')),
            deepcopy(metadata))


class OrderedOutputWriter:
    """At most one submitted write; caller may compute the next slice meanwhile.

    A write includes artifact durability and its ordered manifest update. Call
    drain before reporting completion or touching the manifest on another thread.
    Errors propagate at the next submit/drain and on normal context exit.
    """
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='v4-output')
        self.pending = None

    def drain(self):
        if self.pending is not None:
            result = self.pending.result()
            self.pending = None
            return result

    def submit(self, write, *args):
        self.drain()
        self.pending = self.pool.submit(write, *args)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.drain()
        finally:
            self.pool.shutdown(wait=True, cancel_futures=True)
