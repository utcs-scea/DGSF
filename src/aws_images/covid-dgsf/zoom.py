import os
import warnings
from timeit import default_timer as timer
import numpy
import cupy
# from cupy._core import internal
import _util
from scipy.ndimage import spline_filter
# from cupyx.scipy.ndimage import _interp_kernels

_prod = cupy._core.internal.prod


def _check_parameter(func_name, order, mode):
    if order is None:
        warnings.warn(f'Currently the default order of {func_name} is 1. In a '
                      'future release this may change to 3 to match '
                      'scipy.ndimage ')
    elif order < 0 or 5 < order:
        raise ValueError('spline order is not supported')

    if mode not in ('constant', 'grid-constant', 'nearest', 'mirror',
                    'reflect', 'grid-mirror', 'wrap', 'grid-wrap', 'opencv',
                    '_opencv_edge'):
        raise ValueError('boundary mode ({}) is not supported'.format(mode))


def _prepad_for_spline_filter(input, mode, cval):
    if mode in ['nearest', 'grid-constant']:
        # these modes need padding to get accurate boundary values
        npad = 12  # empirical factor chosen by SciPy
        if mode == 'grid-constant':
            kwargs = dict(mode='constant', constant_values=cval)
        else:
            kwargs = dict(mode='edge')
        padded = numpy.pad(input, npad, **kwargs)
    else:
        npad = 0
        padded = input
    return padded, npad


def _filter_input_cpu(image, prefilter, mode, cval, order):
    if not prefilter or order < 2:
        return (cupy.ascontiguousarray(image), 0)
    padded, npad = _prepad_for_spline_filter(image, mode, cval)
    float_dtype = numpy.promote_types(image.dtype, numpy.float32)
    filtered = spline_filter(padded, order, output=float_dtype, mode=mode)
    return filtered, npad


def zoom(input, zoom, output=None, order=3, mode='constant', cval=0.0,
         prefilter=True, *, grid_mode=False,
         cubin_dir=os.path.dirname(os.path.realpath(__file__))):
    """Zoom an array.
    The array is zoomed using spline interpolation of the requested order.
    Args:
        input (numpy.ndarray): The input array.
        zoom (float or sequence): The zoom factor along the axes. If a float,
            ``zoom`` is the same for each axis. If a sequence, ``zoom`` should
            contain one value for each axis.
        output (cupy.ndarray or ~cupy.dtype): The array in which to place the
            output, or the dtype of the returned array.
        order (int): The order of the spline interpolation, default is 3. Must
            be in the range 0-5.
        mode (str): Points outside the boundaries of the input are filled
            according to the given mode (``'constant'``, ``'nearest'``,
            ``'mirror'``, ``'reflect'``, ``'wrap'``, ``'grid-mirror'``,
            ``'grid-wrap'``, ``'grid-constant'`` or ``'opencv'``).
        cval (scalar): Value used for points outside the boundaries of
            the input if ``mode='constant'`` or ``mode='opencv'``. Default is
            0.0
        prefilter (bool): It is not used yet. It just exists for compatibility
            with :mod:`scipy.ndimage`.
        grid_mode (bool, optional): If False, the distance from the pixel
            centers is zoomed. Otherwise, the distance including the full pixel
            extent is used. For example, a 1d signal of length 5 is considered
            to have length 4 when ``grid_mode`` is False, but length 5 when
            ``grid_mode`` is True. See the following visual illustration:
            .. code-block:: text
                    | pixel 1 | pixel 2 | pixel 3 | pixel 4 | pixel 5 |
                         |<-------------------------------------->|
                                            vs.
                    |<----------------------------------------------->|
            The starting point of the arrow in the diagram above corresponds to
            coordinate location 0 in each mode.
    Returns:
        cupy.ndarray or None:
            The zoomed input.
    .. seealso:: :func:`scipy.ndimage.zoom`
    """

    _check_parameter('zoom', order, mode)

    zoom = _util._fix_sequence_arg(zoom, input.ndim, 'zoom', float)

    output_shape = []
    for s, z in zip(input.shape, zoom):
        output_shape.append(int(round(s * z)))
    output_shape = tuple(output_shape)
    zk_t = 0

    if mode == 'opencv':
        zoom = []
        offset = []
        for in_size, out_size in zip(input.shape, output_shape):
            if out_size > 1:
                zoom.append(float(in_size) / out_size)
                offset.append((zoom[-1] - 1) / 2.0)
            else:
                zoom.append(0)
                offset.append(0)
        mode = 'nearest'

        output = affine_transform(
            input,
            cupy.asarray(zoom),
            offset,
            output_shape,
            output,
            order,
            mode,
            cval,
            prefilter,
        )
    else:
        from zoom_kernel import zoom_kernel
        if grid_mode:

            # warn about modes that may have surprising behavior
            suggest_mode = None
            if mode == 'constant':
                suggest_mode = 'grid-constant'
            elif mode == 'wrap':
                suggest_mode = 'grid-wrap'
            if suggest_mode is not None:
                warnings.warn(
                    f'It is recommended to use mode = {suggest_mode} instead '
                    f'of {mode} when grid_mode is True.')

        zoom = []
        for in_size, out_size in zip(input.shape, output_shape):
            if grid_mode and out_size > 0:
                zoom.append(in_size / out_size)
            elif out_size > 1:
                zoom.append((in_size - 1) / (out_size - 1))
            else:
                zoom.append(0)

        output = _util._get_output(output, input, shape=output_shape)
        if input.dtype.kind in 'iu':
            input = input.astype(numpy.float32)
        filtered, nprepad = _filter_input_cpu(input, prefilter, mode, cval, order)
        filtered = cupy.ascontiguousarray(cupy.asarray(filtered))
        integer_output = output.dtype.kind in 'iu'
        _util._check_cval(mode, cval, integer_output)
        # large_int = max(_prod(input.shape), _prod(output_shape)) > 1 << 31
        zoom = cupy.asarray(zoom, dtype=cupy.float64)
        zk_s = timer()
        zoom_kernel(filtered, zoom, output, cubin_dir)
        zk_e = timer()
        zk_t = zk_e - zk_s
        # kern = _interp_kernels._get_zoom_kernel(
        #     input.ndim, large_int, output_shape, mode, order=order,
        #     integer_output=integer_output, grid_mode=grid_mode,
        #     nprepad=nprepad)
        # kern(filtered, zoom, output)
    return output, zk_t
