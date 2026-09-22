import triton
import triton.language as tl


@triton.jit
def _copy_ndhwc(data, xline, yline, zline, result, BLOCK: tl.constexpr):
    plane_block = tl.program_id(0)
    d = tl.program_id(1)
    b = tl.program_id(2)
    out_in_plane = plane_block * BLOCK + tl.arange(0, BLOCK)
    voxel = out_in_plane // 5
    channel = out_in_plane - voxel * 5
    h = (voxel >> 6) & 63
    w = voxel & 63
    data_offset = b * 524288 + d * 4096 + voxel
    line_offset = b * 64
    src = tl.where(channel == 0, data + data_offset, tl.where(channel == 1, xline + line_offset + w, tl.where(channel == 2, yline + line_offset + h, tl.where(channel == 3, zline + line_offset + d, data + data_offset + 262144))))
    value = tl.load(src)
    result_offset = (b * 64 + d) * 20480 + out_in_plane
    tl.store(result + result_offset, value)


def run(data, xline, yline, zline, result):
    _copy_ndhwc[(10, 64, 12)](data, xline, yline, zline, result, BLOCK=2048, num_warps=8, num_stages=1)
