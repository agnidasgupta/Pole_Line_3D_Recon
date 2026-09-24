"""Bounded, ordered CPU read/parse/preparation ahead of Stage 1 inference.

GPU calls remain on the caller; input workers never write outputs. The caller
must close the iterator on early exit; workers join and errors surface in order.
"""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
import os
import time
import pandas as pd

@dataclass
class LoadedStage1Input:
    frame: object
    item: dict
    read_ms: float
    prepare_ms: float
    schedule: object = None

def _range(name):
    if os.environ.get('POLELINE_NVTX') == '1':
        import torch
        return torch.cuda.nvtx.range(name)
    return nullcontext()

def load_stage1_input(path, grid_size=(400,400,200)):
    from v4_realtime_core import build_sparse_item_from_dataframe
    t=time.perf_counter()
    with _range('input/read_decompress_parse'):
        frame=pd.read_csv(path)
    read_ms=(time.perf_counter()-t)*1000
    t=time.perf_counter()
    with _range('input/sparse_prepare'):
        item=build_sparse_item_from_dataframe(frame,grid_size)
    return LoadedStage1Input(frame,item,read_ms,(time.perf_counter()-t)*1000)

def iter_prefetched(rows, load, *, depth=1, workers=1):
    """Yield ordered inputs with at most ``depth`` future inputs pending.

    Input row discovery must be cheap and independent of current inference.
    Never mutate queued input rows. No GPU tensors may be shared with the worker.
    """
    if not isinstance(depth, int) or depth < 1:
        raise ValueError('prefetch depth must be a positive integer')
    if not isinstance(workers, int) or workers < 1 or workers > depth:
        raise ValueError('prefetch workers must be between 1 and depth')
    iterator=iter(rows);sentinel=object();current=next(iterator,sentinel)
    if current is sentinel:return
    pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='v4-input')
    pending=deque()
    try:
        pending.append((current,pool.submit(load,current)))
        while pending:
            current,future=pending.popleft()
            t=time.perf_counter()
            with _range("input/wait"):
                value=future.result()
            wait_ms=(time.perf_counter()-t)*1000
            del future
            while len(pending) < depth:
                following=next(iterator,sentinel)
                if following is sentinel:break
                pending.append((following,pool.submit(load,following)))
            yield current,value,wait_ms
            del value
    finally:
        for _,future in pending:future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)
