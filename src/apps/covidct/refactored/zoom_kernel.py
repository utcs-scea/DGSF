import os
import cupy
from cupy._core import _carray
from cupy._core import internal


@cupy._util.memoize(for_each_device=True)
def get_zoom_function(cubin_dir):
    fpath = os.path.join(cubin_dir, 'zoom_kernel.cubin').strip()
    print(f"<<< reading zoom from: >{fpath}<", )
    print("exists? ", os.path.isfile(fpath))

    kern_mod = cupy.RawModule(path=fpath)
    kern = kern_mod.get_function('interpolate_zoom_order3_nearest_3d_y302_390_390')
    return kern


def zoom_kernel(x, zoom, _raw_y, cubin_dir):
    block_size = 128
    shape_arr = _raw_y.shape
    shape = [shape_arr[0] * shape_arr[1] * shape_arr[2]]
    shape_arr = cupy.asarray(shape_arr, dtype=cupy.int32)
    idx_size = internal.prod(shape)
    indexer = _carray.Indexer(shape, idx_size)
    inout_args = [x, zoom, _raw_y, shape_arr, indexer]

    kern = get_zoom_function(cubin_dir=cubin_dir)
    gridx = min(0x7fffffff, (indexer.size + block_size - 1) // block_size)
    blockx = min(block_size, indexer.size)
    kern((gridx, 1, 1), (blockx, 1, 1), inout_args, shared_mem=0)
