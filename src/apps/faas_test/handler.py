

def handle(a, b):
    return {"result": "it worked!"}

# from numba import cuda
# import numpy as np
# import math

# @cuda.jit
# def g_add_one(inp, out):
#   idx = cuda.grid(1)
#   out[idx] = inp[idx] + 1

# def main():
#     data = np.ones(256)
#     out = np.ones(256)

#     threadsperblock = 256
#     blockspergrid = math.ceil(data.shape[0] / threadsperblock)
#     print("launching kernel")
#     g_add_one[blockspergrid, threadsperblock](data, out)

#     if out[0] != 2:
#         print("didnt sum one: ", out[0])
#     else:
#         print("all good")

# def handle(a, b):
#     main()
#     return {"result": "it worked!"}

if __name__ == "__main__":
    print("main in")
    #main()
    print("main out")
