typedef double W;
typedef float X;
typedef short Y;

// workaround for HIP: line begins with #include
#include <assert.h>
#include <cupy/carray.cuh>
#include <cupy/math_constants.h>

extern "C" __global__ void interpolate_zoom_order3_nearest_3d_y302_390_390(
    const CArray<float, 3, 1, 1> x, const CArray<double, 1, 1, 1> zoom,
    CArray<short, 1, 1, 1> _raw_y, const CArray<int, 1, 1, 1> shape, CIndexer<1> _ind) {
  CUPY_FOR(i, _ind.size()) {
    _ind.set(i);
    Y &y = _raw_y[_ind.get()];
    double out = 0.0;
    const int xsize_0 = x.shape()[0];
    const int xsize_1 = x.shape()[1];
    const int xsize_2 = x.shape()[2];
    const unsigned int sx_2 = 1;
    const unsigned int sx_1 = sx_2 * xsize_2;
    const unsigned int sx_0 = sx_1 * xsize_1;

    unsigned int in_coord[3];
    unsigned int s, t, idx = i;

    s = shape[2];
    t = idx / s;
    in_coord[2] = idx - t * s;
    idx = t;

    s = shape[1];
    t = idx / s;
    in_coord[1] = idx - t * s;
    idx = t;

    in_coord[0] = idx;

    W c_0 = zoom[0] * (W)in_coord[0] + (W)12;

    W c_1 = zoom[1] * (W)in_coord[1] + (W)12;

    W c_2 = zoom[2] * (W)in_coord[2] + (W)12;

    W wx, wy;
    int start;

    W weights_0[4];

    wx = c_0 - floor(3 & 1 ? c_0 : c_0 + 0.5);
    wy = 1.0 - wx;
    weights_0[1] = (wx * wx * (wx - 2.0) * 3.0 + 4.0) / 6.0;
    weights_0[2] = (wy * wy * (wy - 2.0) * 3.0 + 4.0) / 6.0;
    weights_0[0] = wy * wy * wy / 6.0;
    weights_0[3] = 1.0 - weights_0[0] - weights_0[1] - weights_0[2];

    start = (int)floor((double)c_0) - 1;
    int ci_0[4];

    ci_0[0] = start + 0;

    ci_0[0] = min(max(ci_0[0], 0), xsize_0 - 1);

    ci_0[1] = start + 1;

    ci_0[1] = min(max(ci_0[1], 0), xsize_0 - 1);

    ci_0[2] = start + 2;

    ci_0[2] = min(max(ci_0[2], 0), xsize_0 - 1);

    ci_0[3] = start + 3;

    ci_0[3] = min(max(ci_0[3], 0), xsize_0 - 1);

    W w_0;
    int ic_0;
    for (int k_0 = 0; k_0 <= 3; k_0++) {
      w_0 = weights_0[k_0];
      ic_0 = ci_0[k_0] * sx_0;

      W weights_1[4];

      wx = c_1 - floor(3 & 1 ? c_1 : c_1 + 0.5);
      wy = 1.0 - wx;
      weights_1[1] = (wx * wx * (wx - 2.0) * 3.0 + 4.0) / 6.0;
      weights_1[2] = (wy * wy * (wy - 2.0) * 3.0 + 4.0) / 6.0;
      weights_1[0] = wy * wy * wy / 6.0;
      weights_1[3] = 1.0 - weights_1[0] - weights_1[1] - weights_1[2];

      start = (int)floor((double)c_1) - 1;
      int ci_1[4];

      ci_1[0] = start + 0;

      ci_1[0] = min(max(ci_1[0], 0), xsize_1 - 1);

      ci_1[1] = start + 1;

      ci_1[1] = min(max(ci_1[1], 0), xsize_1 - 1);

      ci_1[2] = start + 2;

      ci_1[2] = min(max(ci_1[2], 0), xsize_1 - 1);

      ci_1[3] = start + 3;

      ci_1[3] = min(max(ci_1[3], 0), xsize_1 - 1);

      W w_1;
      int ic_1;
      for (int k_1 = 0; k_1 <= 3; k_1++) {
        w_1 = weights_1[k_1];
        ic_1 = ci_1[k_1] * sx_1;

        W weights_2[4];

        wx = c_2 - floor(3 & 1 ? c_2 : c_2 + 0.5);
        wy = 1.0 - wx;
        weights_2[1] = (wx * wx * (wx - 2.0) * 3.0 + 4.0) / 6.0;
        weights_2[2] = (wy * wy * (wy - 2.0) * 3.0 + 4.0) / 6.0;
        weights_2[0] = wy * wy * wy / 6.0;
        weights_2[3] = 1.0 - weights_2[0] - weights_2[1] - weights_2[2];

        start = (int)floor((double)c_2) - 1;
        int ci_2[4];

        ci_2[0] = start + 0;

        ci_2[0] = min(max(ci_2[0], 0), xsize_2 - 1);
        ci_2[1] = start + 1;
        ci_2[1] = min(max(ci_2[1], 0), xsize_2 - 1);
        ci_2[2] = start + 2;
        ci_2[2] = min(max(ci_2[2], 0), xsize_2 - 1);
        ci_2[3] = start + 3;
        ci_2[3] = min(max(ci_2[3], 0), xsize_2 - 1);
        W w_2;
        int ic_2;
        for (int k_2 = 0; k_2 <= 3; k_2++) {
          w_2 = weights_2[k_2];
          ic_2 = ci_2[k_2] * sx_2;

          double val = (double)x[ic_0 + ic_1 + ic_2];
          out += val * (double)(w_0 * w_1 * w_2);
        }
      }
    }
    y = (Y)rint((double)out);
    ;
  };
}
