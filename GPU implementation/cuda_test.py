from numba import cuda

def cpu_print(N):
    for i in range(0, N):
        print(i)

@cuda.jit
def gpu_print(N):
    idx = cuda.threadIdx.x+cuda.blockIdx.x*cuda.blockDim.x
    if (idx<N):
        print(idx)

# def main():
#     gpu_print[128,8](1024)
#     # cpu_print(8)

gpu_print[128,8](1024)