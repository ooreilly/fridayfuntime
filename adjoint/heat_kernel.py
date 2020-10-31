import numpy as np
from pycuda.compiler import SourceModule
from pycuda.autoinit import context
import pycuda.driver as cuda


def heat_equation(v, u, a, dt, h, deviceID=0, options=["--use_fast_math", "--ptxas-options=-v"],
        block=(1,1,1), update=1.0):
    mp = cuda.device_attribute.MULTIPROCESSOR_COUNT
    num_blocks = 2 * cuda.Device(deviceID).get_attribute(mp)
    source = open("heat_kernel.cu").read()
    mod = SourceModule(source, options=options)
    fcn = mod.get_function("heat_kernel")
    return lambda v, u : fcn(v.device, u.device, np.int32(u.shape[0]), np.float64(update), a.device, np.float64(dt) , np.float64(h), block=block,
            grid=(num_blocks, 1, 1))

def periodic_bc(u):
    source = open("heat_kernel.cu").read()
    mod = SourceModule(source)
    fcn = mod.get_function("test_periodic_bc")
    fcn(u.device, np.int32(u.shape[0]), block=(32, 1, 1), grid=(1, 1, 1))


class Array():

    def __init__(self, host_array, device_array=None):
        self.device = device_array
        self.dtype = host_array.dtype
        self.shape = host_array.shape
        self.nbytes = host_array.nbytes

    def get(self):
        out = np.ndarray(self.shape, self.dtype)
        cuda.memcpy_dtoh(out, self.device)
        return out

def array(u):
    out = cuda.mem_alloc(u.nbytes)
    cuda.memcpy_htod(out, u)
    return Array(u, out)


