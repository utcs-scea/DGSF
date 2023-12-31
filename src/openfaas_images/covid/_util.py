import warnings
import numpy
import cupy


def _check_cval(mode, cval, integer_output):
    if mode == 'constant' and integer_output and not cupy.isfinite(cval):
        raise NotImplementedError("Non-finite cval is not supported for "
                                  "outputs with integer dtype.")


def _get_inttype(input):
    # The integer type to use for indices in the input array
    # The indices actually use byte positions and we can't just use
    # input.nbytes since that won't tell us the number of bytes between the
    # first and last elements when the array is non-contiguous
    nbytes = sum((x-1)*abs(stride) for x, stride in
                 zip(input.shape, input.strides)) + input.dtype.itemsize
    return 'int' if nbytes < (1 << 31) else 'ptrdiff_t'


def _get_output(output, input, shape=None, complex_output=False):
    shape = input.shape if shape is None else shape
    if output is None:
        if complex_output:
            _dtype = cupy.promote_types(input.dtype, cupy.complex64)
        else:
            _dtype = input.dtype
        output = cupy.zeros(shape, dtype=_dtype)
    elif isinstance(output, (type, cupy.dtype)):
        if complex_output and cupy.dtype(output).kind != 'c':
            warnings.warn("promoting specified output dtype to complex")
            output = cupy.promote_types(output, cupy.complex64)
        output = cupy.zeros(shape, dtype=output)
    elif isinstance(output, str):
        output = numpy.sctypeDict[output]
        if complex_output and cupy.dtype(output).kind != 'c':
            raise RuntimeError("output must have complex dtype")
        output = cupy.zeros(shape, dtype=output)
    elif output.shape != shape:
        raise RuntimeError("output shape not correct")
    elif complex_output and output.dtype.kind != 'c':
        raise RuntimeError("output must have complex dtype")
    return output


def _fix_sequence_arg(arg, ndim, name, conv=lambda x: x):
    if isinstance(arg, str):
        return [conv(arg)] * ndim
    try:
        arg = iter(arg)
    except TypeError:
        return [conv(arg)] * ndim
    lst = [conv(x) for x in arg]
    if len(lst) != ndim:
        msg = "{} must have length equal to input rank".format(name)
        raise RuntimeError(msg)
    return lst


def _generate_boundary_condition_ops(mode, ix, xsize, int_t="int",
                                     float_ix=False):
    min_func = "fmin" if float_ix else "min"
    max_func = "fmax" if float_ix else "max"
    if mode in ['reflect', 'grid-mirror']:
        ops = '''
        if ({ix} < 0) {{
            {ix} = - 1 -{ix};
        }}
        {ix} %= {xsize} * 2;
        {ix} = {min}({ix}, 2 * {xsize} - 1 - {ix});'''.format(
            ix=ix, xsize=xsize, min=min_func)
    elif mode == 'mirror':
        ops = '''
        if ({xsize} == 1) {{
            {ix} = 0;
        }} else {{
            if ({ix} < 0) {{
                {ix} = -{ix};
            }}
            {ix} = 1 + ({ix} - 1) % (({xsize} - 1) * 2);
            {ix} = {min}({ix}, 2 * {xsize} - 2 - {ix});
        }}'''.format(ix=ix, xsize=xsize, min=min_func)
    elif mode == 'nearest':
        ops = '''
        {ix} = {min}({max}({ix}, 0), {xsize} - 1);'''.format(
            ix=ix, xsize=xsize, min=min_func, max=max_func)
    elif mode == 'grid-wrap':
        ops = '''
        {ix} %= {xsize};
        if ({ix} < 0) {{
            {ix} += {xsize};
        }}'''.format(ix=ix, xsize=xsize)
    elif mode == 'wrap':
        ops = '''
        if ({ix} < 0) {{
            {ix} += ({sz} - 1) * (({int_t})(-{ix} / ({sz} - 1)) + 1);
        }} else if ({ix} > ({sz} - 1)) {{
            {ix} -= ({sz} - 1) * ({int_t})({ix} / ({sz} - 1));
        }};'''.format(ix=ix, sz=xsize, int_t=int_t)
    elif mode in ['constant', 'grid-constant']:
        ops = '''
        if (({ix} < 0) || {ix} >= {xsize}) {{
            {ix} = -1;
        }}'''.format(ix=ix, xsize=xsize)
    return ops