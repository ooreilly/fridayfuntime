import heat_kernel as hk
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


def test_heat_kernel():
    n = 11
    # stencil radius
    r = 4
    a = np.ones(n)
    dt = np.float(0.1)
    x = np.linspace(0, 1, n)
    u = x ** 2
    h = x[1] - x[0]
    d_a = hk.array(a)
    d_u = hk.array(u)
    d_v = hk.array(u)
    f = hk.heat_equation(d_v, d_u, d_a, dt, h, update=0.0)
    f(d_v, d_u)
    v = d_v.get()
    assert np.all(np.isclose(2.0 * dt + 0 * v[r:-r], v[r:-r]))

def test_periodic_bc():
    n = 12
    r = 4
    x = np.arange(n)
    u = np.tile([0, 1, -1, 0], 3)
    d_u = hk.array(u)
    hk.periodic_bc(d_u)
    v = d_u.get()
    assert np.all(np.isclose(u,v))
