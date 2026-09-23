"""Guarded dispatcher for the retained Kernel Factory cached-input copy kernel.

Campaign rdjabvhxps61s60qvkw671b0zc, candidate
poleline_cached_copy_triton_plane2048. No output values are cached.
"""
import torch


def try_copy_cached_input(data, x, y, z, output):
    tensors = (data, x, y, z, output)
    if (data.shape != (12, 2, 64, 64, 64)
            or output.shape != (12, 5, 64, 64, 64)
            or any(t.shape != (12, 64) for t in (x, y, z))
            or any(not t.is_cuda or t.dtype != torch.float32 for t in tensors)
            or any(t.device != data.device for t in tensors)
            or any(not t.is_contiguous() for t in (data, x, y, z))
            or not output.is_contiguous(memory_format=torch.channels_last_3d)):
        return False
    # The caller supplies a distinct output allocation.
    if any(output.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
           for t in (data, x, y, z)):
        return False
    try:
        from v4_kernel_factory_input_copy import run
    except ImportError:
        return False
    run(data, x, y, z, output.permute(0, 2, 3, 4, 1))
    return True
