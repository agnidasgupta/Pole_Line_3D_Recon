"""Exact core scheduling with one scalar key and one gather offset per row."""
import math
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PreparedCoreSchedule:
    """CPU-only schedule bound to a coordinate snapshot and grid configuration.

    Treat the arrays/groups as immutable. Validation rejects stale coordinates;
    plans must not be mutated concurrently with inference.
    """
    coords: np.ndarray
    grid_size: tuple
    core_size: int
    groups: list

    def validate(self, coords, grid_size, core_size):
        if tuple(map(int, grid_size)) != self.grid_size or int(core_size) != self.core_size:
            raise ValueError('Prepared schedule grid/core configuration changed')
        if not np.array_equal(np.asarray(coords, dtype=np.int32), self.coords):
            raise ValueError('Prepared schedule coordinates changed')
        return self.groups


def prepare_core_schedule(coords, grid_size=(400,400,200), core_size=48):
    """Prepare on a CPU worker while an independent slice executes on the GPU."""
    snapshot = np.array(coords, dtype=np.int32, copy=True)
    groups = active_core_groups_with_offsets(snapshot, grid_size, core_size)
    snapshot.flags.writeable = False
    for group in groups:
        for key in ('origin', 'center', 'rows', '_flat_core'):
            group[key].flags.writeable = False
    return PreparedCoreSchedule(snapshot, tuple(map(int, grid_size)), int(core_size), groups)


def iter_prepared_core_schedules(items, grid_size=(400,400,200), core_size=48):
    """Yield ordered (item, schedule) pairs with one CPU preparation ahead.

    Only scheduling is overlapped; loading items and GPU inference stay with the
    caller. Do not mutate submitted items. Close the iterator on early exit so
    its worker is joined. The first preparation is still paid by the caller.
    """
    from concurrent.futures import ThreadPoolExecutor
    iterator = iter(items)
    sentinel = object()
    current = next(iterator, sentinel)
    if current is sentinel:
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='v4-prepare') as pool:
        pending = pool.submit(prepare_core_schedule, current['coords'], grid_size, core_size)
        while True:
            plan = pending.result()
            following = next(iterator, sentinel)
            if following is not sentinel:
                pending = pool.submit(prepare_core_schedule, following['coords'], grid_size, core_size)
            yield current, plan
            if following is sentinel:
                break
            current = following


def active_core_groups_with_offsets(coords, grid_size=(400,400,200), core_size=48):
    c=np.asarray(coords,dtype=np.int32)
    if not len(c):return []
    gx,gy,gz=map(int,grid_size);csz=int(core_size)
    if csz<=0:raise ValueError('core_size must be positive')
    if np.any(c<0) or np.any(c[:,0]>=gx) or np.any(c[:,1]>=gy) or np.any(c[:,2]>=gz):
        raise RuntimeError('active-core scheduling received an out-of-grid coordinate')
    nx=int(math.ceil(gx/csz));ny=int(math.ceil(gy/csz));nz=int(math.ceil(gz/csz))
    dtype=np.int32 if max(nx*ny*nz,csz**3)<=np.iinfo(np.int32).max else np.int64
    x,y,z=np.ascontiguousarray(c.T,dtype=dtype)
    qx=x//csz;qy=y//csz;qz=z//csz
    flat=(qz*ny+qy)*nx+qx
    order=np.argsort(flat,kind='stable').astype(np.int64,copy=False)
    sorted_flat=flat[order]
    starts=np.r_[0,np.flatnonzero(sorted_flat[1:]!=sorted_flat[:-1])+1]
    stops=np.r_[starts[1:],len(order)]
    # Identical integer expression to the accepted origin subtraction. Valid
    # grid coordinates make modulo equal to each coordinate minus core origin.
    # Reuse the quotients above instead of dividing every coordinate again.
    offsets=(((z-qz*csz)*csz+y-qy*csz)*csz+x-qx*csz)[order].astype(np.int64,copy=False)
    groups=[]
    for start,stop in zip(starts,stops):
        keyid=int(sorted_flat[start]);kx=keyid%nx;ky=(keyid//nx)%ny;kz=keyid//(nx*ny)
        key=(kx,ky,kz);origin=np.asarray(key,np.int64)*csz
        groups.append({'key':key,'origin':origin,'center':origin+csz//2,
                       'rows':order[start:stop],'_flat_core':offsets[start:stop]})
    return groups
